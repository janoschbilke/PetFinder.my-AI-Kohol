import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from src.data import get_data_root

DEFAULT_BACKBONE = "alexnet"
CACHE_DIR = Path("cache")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_backbone(name: str) -> tuple[torch.nn.Module, int]:
    if name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        model.classifier = model.classifier[:6]
        embedding_size = 4096
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        embedding_size = 2048
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        embedding_size = 1280
    else:
        raise ValueError(f"Unknown backbone: {name}")

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model, embedding_size


def embed_pet(pet_id: str, images_dir: Path, model: torch.nn.Module, device: torch.device) -> np.ndarray | None:
    image_vecs = []
    idx = 1
    while True:
        img_path = images_dir / f"{pet_id}-{idx}.jpg"
        if not img_path.exists():
            break
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = model(tensor)
            image_vecs.append(vec.squeeze().cpu().numpy())
        except Exception:
            pass
        idx += 1

    if not image_vecs:
        return None
    return np.mean(image_vecs, axis=0)


def compute_embeddings(pet_ids: list[str], images_dir: Path, model: torch.nn.Module, device: torch.device, embedding_size: int, backbone: str = DEFAULT_BACKBONE) -> np.ndarray:
    embeddings = np.zeros((len(pet_ids), embedding_size), dtype=np.float32)
    for i, pet_id in enumerate(tqdm(pet_ids, desc=f"Embedding ({backbone})")):
        vec = embed_pet(pet_id, images_dir, model, device)
        if vec is not None:
            embeddings[i] = vec
    return embeddings


def run(force: bool = False, backbone: str = DEFAULT_BACKBONE) -> None:
    """Compute and cache CNN embeddings for all pets.

    Args:
        force: Recompute even if cache exists
        backbone: CNN backbone to use ("alexnet", "resnet50", "efficientnet_b0")
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Backbone-specific cache files so multiple backbones can coexist
    train_emb_out = CACHE_DIR / f"train_embeddings_{backbone}.npy"
    train_ids_out = CACHE_DIR / f"train_pet_ids_{backbone}.npy"
    test_emb_out = CACHE_DIR / f"test_embeddings_{backbone}.npy"
    test_ids_out = CACHE_DIR / f"test_pet_ids_{backbone}.npy"

    if not force and all(p.exists() for p in [train_emb_out, train_ids_out, test_emb_out, test_ids_out]):
        print(f"Embedding cache found ({backbone}), skipping. Use --force to recompute.")
        return

    data_root = get_data_root()
    train_images_dir = data_root / "train_images"
    test_images_dir = data_root / "test_images"
    train_csv = data_root / "train" / "train.csv"
    test_csv = data_root / "test" / "test.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Backbone: {backbone}")

    model, embedding_size = build_backbone(backbone)
    model = model.to(device)

    import pandas as pd
    train_pet_ids = pd.read_csv(train_csv)["PetID"].tolist()
    test_pet_ids = pd.read_csv(test_csv)["PetID"].tolist()

    print(f"Computing train embeddings ({len(train_pet_ids)} pets)...")
    train_embeddings = compute_embeddings(train_pet_ids, train_images_dir, model, device, embedding_size, backbone)

    print(f"Computing test embeddings ({len(test_pet_ids)} pets)...")
    test_embeddings = compute_embeddings(test_pet_ids, test_images_dir, model, device, embedding_size, backbone)

    np.save(train_emb_out, train_embeddings)
    np.save(train_ids_out, np.array(train_pet_ids))
    np.save(test_emb_out, test_embeddings)
    np.save(test_ids_out, np.array(test_pet_ids))

    print(f"Train embeddings: {train_embeddings.shape}  -> {train_emb_out}")
    print(f"Test embeddings:  {test_embeddings.shape}  -> {test_emb_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=["alexnet", "resnet50", "efficientnet_b0"],
        default=DEFAULT_BACKBONE,
        help=f"CNN backbone to use (default: {DEFAULT_BACKBONE})",
    )
    args = parser.parse_args()
    run(force=args.force, backbone=args.backbone)
