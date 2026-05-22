# PetFinder Adoption Speed Prediction

## Problem Statement

-> multiclass ordinal classification problem, `AdoptionSpeed` has 5 discrete ordered categories:

| Label | Meaning |
| ------- | --------- |
| 0 | Adopted on the same day as listed |
| 1 | Adopted within 1–7 days |
| 2 | Adopted within 8–30 days |
| 3 | Adopted within 31–90 days |
| 4 | Not adopted after 100 days |

**Evaluation metrics:**

- Quadratic Weighted Kappa (QWK)
- Accuracy
- F1 macro and weighted
- Per-class classification report
- Confusion matrix

---

## Inputs

- `train/train.csv`
- `train_images/`
- `train_metadata/`
- `train_sentiment/`

---

## Full Feature Matrix

Every pet is represented as a single flat row built from four blocks:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FULL FEATURE VECTOR (1 row per pet)                  │
├──────────────────────────┬──────────────────────────────────────────────────┤
│ BLOCK A — Tabular        │  Numeric (kept as-is): 5 (6) cols                │
│ (from train.csv)         │    Age, Quantity,                                │
│                          │    Fee, VideoAmt, PhotoAmt                       │
│                          │    AdoptionSpeed (label)                         │
│                          │                                                  │
│                          │    One-hot encoded: 33 cols                      │
│                          │    Type          → 1 cols                        │
│                          │    Gender        → 2 cols                        │
│                          │    State         → 14 cols                       │
│                          │    Vaccinated    → 2 cols                        │
│                          │    Dewormed      → 2 cols                        │
│                          │    Sterilized    → 2 cols                        │
│                          │    Health        → 3 cols                        │
│                          │    MaturitySize  → 4 cols                        │
│                          │    FurLength     → 3 cols                        │
│                          │                                                  │
│                          │    Mutli-hot endoced: 314 cols                   │
│                          │    Breed   → 307 cols                            │
│                          │    Color   → 7 cols                              │
│                          │                                                  │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ BLOCK B — Sentiment      │ 5 columns                                        │
│ (from train_sentiment/)  │                                                  │
│                          │    sentiment_score                               │
│                          │    sentiment_magnitude                           │
│                          │    avg_sentence_score                            │
│                          │    avg_sentence_magnitude                        │
│                          │    num_sentences                                 │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ BLOCK C — Image Metadata │ ~8 columns (aggregated across all images)        │
│ (from train_metadata/)   │                                                  │
│                          │    meta_top_label_score   (max label confidence) │
│                          │    meta_mean_label_score  (mean label confidence)│
│                          │    meta_num_labels        (# detected labels)    │
│                          │    meta_dom_color_R       (dominant colour red)  │
│                          │    meta_dom_color_G       (dominant colour green)│
│                          │    meta_dom_color_B       (dominant colour blue) │
│                          │    meta_dom_color_frac    (pixel fraction)       │
│                          │    meta_crop_confidence   (crop hint confidence) │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ BLOCK D — CNN Embeddings │ embedding_size columns (depends on backbone)     │
│ (from train_images/)     │                                                  │
│                          │  Mean-pooled across all images per pet           │
│                          │  embed_0, embed_1, …, embed_{N-1}                │
└──────────────────────────┴──────────────────────────────────────────────────┘

TOTAL WIDTH (AlexNet):  5 + 33 + 307 + 5 + 8 + 4096  ≈  4,454 columns per pet
```

-> LightGBM with L1 regularisation (`reg_alpha > 0`) for feature selection

---

## Step-by-Step Plan for the training

### Step 1 — Preprocessing (`src/preprocessing.py`)

-> produces Blocks A + B + C

_Drop irrelevant / leaky columns:_

- `PetID`, unique identifier, not a feature (kept only as a join key)
- `Name`, free text, too sparse to be useful
- `Description`, raw text; replaced by pre-computed sentiment features
- `RescuerID`is a discussion point as to which extend this is needed, will hopefully be shown via L1

**Block A — Encoding:**

| Column | Treatment |
| -------- | ----------- |
| `Age`, `Breed1`, `Breed2`, `Quantity`, `Fee`, `VideoAmt`, `PhotoAmt` | Keep numeric |
| `RescuerID` | Keep as a numeric feature (label-encoded); L1 will determine if it's useful |
| `Type`, `Gender`, `Color1/2/3`, `State` | One-hot encode |
| `Vaccinated`, `Dewormed`, `Sterilized`, `Health`, `MaturitySize`, `FurLength` | One-hot encode |

**Block B — Sentiment features** (from `train_sentiment/{PetID}.json`):

- `sentiment_score`, `sentiment_magnitude` — document-level
- `avg_sentence_score`, `avg_sentence_magnitude` — mean across sentences
- `num_sentences`

**Block C — Image metadata features** (from `train_metadata/{PetID}-{n}.json`,
aggregated across all images for the same pet):

- `meta_top_label_score`, `meta_mean_label_score`, `meta_num_labels`
- `meta_dom_color_R/G/B`, `meta_dom_color_frac`, `meta_crop_confidence`

**Output:** `cache/train_features.parquet` and `cache/test_features.parquet`

---

### Step 2: Image Embeddings (`src/image_embeddings.py`)

Produces Block D, raw visual features from the actual photos -> creates embeddings (size depending on model)

-> CNN backbone, we want a swappable design. Start with AlexNet (from AI Lecture) and maybe try others later on

AI says good choice could be:

| Backbone | `BACKBONE` value | Embedding size |
| ---------- | ----------------- | --------------- |
| **AlexNet** | `"alexnet"` | 4096 |
| ResNet-50 | `"resnet50"` | 2048 |
| EfficientNet-B0 | `"efficientnet_b0"` | 1280 |

**Output:** `cache/train_embeddings.npy`, `cache/test_embeddings.npy`,
`cache/train_pet_ids.npy`, `cache/test_pet_ids.npy`

---

### Step 3: Model Training & Prediction (`src/model.py`)

-> Load and assemble full feature matrix: ~4,189 columns (AlexNet)

AI says good Classifier could be LightGBM with:

```text
- `reg_alpha > 0` (L1) drives unimportant features towards zero
- Balanced class weights via `compute_sample_weight("balanced", ...)`
- Early stopping (50 rounds) per fold
```

**Saved outputs:**

- `cache/model.pkl` — best fold model (by QWK), loadable independently
- `cache/oof_predictions.npy` — out-of-fold predicted labels for hyperparameter validation
- `cache/oof_labels.npy` — true labels in OOF order
- `cache/col_names.npy` — feature column names for importance plots
- `submission.csv` — test set predictions

---

### Step 4: Evaluation (`src/evaluate.py`)

Runs independently by loading `cache/model.pkl` and OOF arrays.

**Metrics computed on OOF predictions:**

- Quadratic Weighted Kappa (QWK)
- Accuracy
- F1 macro and weighted
- Per-class precision / recall / F1 (classification report)

**Plots saved:**

- `confusion_matrix.png` — side-by-side counts and row-% confusion matrices
- `feature_importance.png` — top 40 features by LightGBM split importance
