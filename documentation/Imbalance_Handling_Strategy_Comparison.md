# Imbalance Strategy Comparison — Mode: `all_multiclass`

Feature set: `resnet50` / PCA `64` | 15 Optuna trials | 5-fold CV | Mode: `all_multiclass`

Class distribution (approx): `{0: 410, 1: 3090, 2: 4037, 3: 3259, 4: 4197}`

## Summary Table (sorted by QWK)

| Strategy | QWK | Accuracy | F1 macro | F1 weighted |
| ---------- | ----- | ---------- | ---------- | ------------- |
| SMOTE oversampling (in-fold) | 0.3961 | 0.4173 | 0.3343 | 0.4024 |
| Custom class_0 weight (Optuna) | 0.3945 | 0.4209 | 0.3417 | 0.4038 |
| Baseline (balanced class_weight) | 0.3936 | 0.4202 | 0.3248 | 0.4000 |

## Detailed Results

---

### SMOTE oversampling (in-fold)  (id: `_resnet50_pca64_smote`)

- **Strategy:**      `balanced`
- **SMOTE:**         `True`
- **QWK:**           0.3961
- **Accuracy:**      0.4173
- **F1 macro:**      0.3343
- **F1 weighted:**   0.4024

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) SMOTE

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Sterilized_2` | 45244.1 | 4.1% |
| 2 | `Age` | 34691.9 | 3.2% |
| 3 | `Breed_PCA_1` | 32218.5 | 3.0% |
| 4 | `Breed_PCA_0` | 29064.0 | 2.7% |
| 5 | `FurLength_1` | 27236.4 | 2.5% |
| 6 | `FurLength_2` | 25287.5 | 2.3% |
| 7 | `MaturitySize_2` | 24505.8 | 2.2% |
| 8 | `Color_1` | 18072.0 | 1.7% |
| 9 | `Gender_1` | 17739.5 | 1.6% |
| 10 | `embed_pca_5` | 16698.2 | 1.5% |
| 11 | `meta_num_labels` | 16022.3 | 1.5% |
| 12 | `Dewormed_1` | 15374.6 | 1.4% |
| 13 | `embed_pca_0` | 14725.4 | 1.3% |
| 14 | `meta_top_label_score` | 12941.1 | 1.2% |
| 15 | `embed_pca_3` | 12782.2 | 1.2% |
| 16 | `PhotoAmt` | 12597.1 | 1.2% |
| 17 | `meta_mean_label_score` | 12287.1 | 1.1% |
| 18 | `Color_2` | 11936.4 | 1.1% |
| 19 | `Color_5` | 11883.0 | 1.1% |
| 20 | `Breed_PCA_3` | 11699.4 | 1.1% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) SMOTE

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1524 |
| 2 | `Sterilized_2` | 0.0777 |
| 3 | `embed_pca_5` | 0.0758 |
| 4 | `embed_pca_0` | 0.0700 |
| 5 | `Gender_1` | 0.0644 |
| 6 | `Breed_PCA_0` | 0.0564 |
| 7 | `embed_pca_3` | 0.0542 |
| 8 | `FurLength_1` | 0.0488 |
| 9 | `num_sentences` | 0.0488 |
| 10 | `MaturitySize_2` | 0.0487 |
| 11 | `meta_num_labels` | 0.0465 |
| 12 | `embed_pca_17` | 0.0457 |
| 13 | `Quantity` | 0.0428 |
| 14 | `embed_pca_28` | 0.0426 |
| 15 | `Breed_PCA_1` | 0.0421 |
| 16 | `meta_mean_label_score` | 0.0408 |
| 17 | `PhotoAmt` | 0.0379 |
| 18 | `Color_2` | 0.0378 |
| 19 | `embed_pca_56` | 0.0372 |
| 20 | `embed_pca_38` | 0.0356 |

#### Feature Group Impact (Mean |SHAP| summed per group) SMOTE

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 64 | 1.5453 | 50.3% |
| Categorical (OHE) | 33 | 0.4737 | 15.4% |
| Breed (PCA) | 15 | 0.3029 | 9.9% |
| Numeric | 5 | 0.2750 | 9.0% |
| Image Metadata (Google Vision) | 8 | 0.1913 | 6.2% |
| Color (multi-hot) | 7 | 0.1487 | 4.8% |
| Sentiment (NLP) | 5 | 0.1332 | 4.3% |
<!-- feature-analysis-end -->

---

### Custom class_0 weight (Optuna)  (id: `_resnet50_pca64_custom_weights`)

- **Strategy:**      `custom_weights`
- **SMOTE:**         `False`
- **QWK:**           0.3945
- **Accuracy:**      0.4209
- **F1 macro:**      0.3417
- **F1 weighted:**   0.4038

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) Optuna

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 18659.8 | 2.5% |
| 2 | `embed_pca_0` | 15801.1 | 2.1% |
| 3 | `embed_pca_5` | 13838.0 | 1.9% |
| 4 | `Breed_PCA_1` | 13228.9 | 1.8% |
| 5 | `meta_top_label_score` | 12163.9 | 1.6% |
| 6 | `embed_pca_3` | 12123.9 | 1.6% |
| 7 | `embed_pca_17` | 11676.5 | 1.6% |
| 8 | `embed_pca_28` | 11288.3 | 1.5% |
| 9 | `meta_num_labels` | 10731.6 | 1.4% |
| 10 | `Breed_PCA_3` | 10414.9 | 1.4% |
| 11 | `meta_mean_label_score` | 10234.6 | 1.4% |
| 12 | `embed_pca_12` | 9843.1 | 1.3% |
| 13 | `embed_pca_15` | 9674.8 | 1.3% |
| 14 | `sentiment_magnitude` | 9368.9 | 1.3% |
| 15 | `embed_pca_8` | 9329.6 | 1.3% |
| 16 | `embed_pca_19` | 9283.9 | 1.3% |
| 17 | `embed_pca_13` | 9251.1 | 1.2% |
| 18 | `embed_pca_48` | 9061.4 | 1.2% |
| 19 | `embed_pca_30` | 8888.5 | 1.2% |
| 20 | `embed_pca_9` | 8783.9 | 1.2% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) Optuna

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1544 |
| 2 | `embed_pca_0` | 0.0830 |
| 3 | `embed_pca_5` | 0.0668 |
| 4 | `embed_pca_3` | 0.0656 |
| 5 | `Sterilized_2` | 0.0605 |
| 6 | `embed_pca_17` | 0.0570 |
| 7 | `meta_num_labels` | 0.0523 |
| 8 | `Breed_PCA_1` | 0.0498 |
| 9 | `Quantity` | 0.0481 |
| 10 | `meta_top_label_score` | 0.0479 |
| 11 | `Breed_PCA_3` | 0.0464 |
| 12 | `meta_mean_label_score` | 0.0449 |
| 13 | `sentiment_magnitude` | 0.0426 |
| 14 | `embed_pca_28` | 0.0417 |
| 15 | `num_sentences` | 0.0409 |
| 16 | `embed_pca_10` | 0.0398 |
| 17 | `embed_pca_13` | 0.0393 |
| 18 | `embed_pca_48` | 0.0390 |
| 19 | `embed_pca_8` | 0.0387 |
| 20 | `embed_pca_12` | 0.0364 |

#### Feature Group Impact (Mean |SHAP| summed per group) Optuna

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 64 | 1.8390 | 62.6% |
| Breed (PCA) | 15 | 0.2505 | 8.5% |
| Numeric | 5 | 0.2498 | 8.5% |
| Image Metadata (Google Vision) | 8 | 0.2159 | 7.4% |
| Categorical (OHE) | 33 | 0.1916 | 6.5% |
| Sentiment (NLP) | 5 | 0.1403 | 4.8% |
| Color (multi-hot) | 7 | 0.0485 | 1.7% |
<!-- feature-analysis-end -->

---

### Baseline (balanced class_weight)  (id: `_resnet50_pca64_baseline`)

- **Strategy:**      `balanced`
- **SMOTE:**         `False`
- **QWK:**           0.3936
- **Accuracy:**      0.4202
- **F1 macro:**      0.3248
- **F1 weighted:**   0.4000

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) balanced

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 13907.1 | 4.6% |
| 2 | `meta_top_label_score` | 6143.8 | 2.0% |
| 3 | `embed_pca_5` | 5880.4 | 1.9% |
| 4 | `embed_pca_0` | 5609.9 | 1.8% |
| 5 | `embed_pca_3` | 5508.2 | 1.8% |
| 6 | `embed_pca_17` | 5045.8 | 1.7% |
| 7 | `meta_num_labels` | 4893.8 | 1.6% |
| 8 | `meta_mean_label_score` | 4690.5 | 1.5% |
| 9 | `Sterilized_2` | 4283.9 | 1.4% |
| 10 | `embed_pca_13` | 4031.7 | 1.3% |
| 11 | `embed_pca_8` | 3963.8 | 1.3% |
| 12 | `embed_pca_19` | 3900.1 | 1.3% |
| 13 | `Breed_PCA_3` | 3873.0 | 1.3% |
| 14 | `embed_pca_10` | 3851.0 | 1.3% |
| 15 | `embed_pca_9` | 3821.2 | 1.3% |
| 16 | `embed_pca_15` | 3799.1 | 1.2% |
| 17 | `embed_pca_48` | 3793.8 | 1.2% |
| 18 | `embed_pca_14` | 3733.1 | 1.2% |
| 19 | `embed_pca_2` | 3700.9 | 1.2% |
| 20 | `embed_pca_11` | 3538.4 | 1.2% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) balanced

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1513 |
| 2 | `embed_pca_0` | 0.0861 |
| 3 | `embed_pca_5` | 0.0643 |
| 4 | `embed_pca_3` | 0.0529 |
| 5 | `embed_pca_17` | 0.0495 |
| 6 | `Sterilized_2` | 0.0471 |
| 7 | `Quantity` | 0.0467 |
| 8 | `meta_num_labels` | 0.0455 |
| 9 | `Breed_PCA_1` | 0.0444 |
| 10 | `meta_top_label_score` | 0.0421 |
| 11 | `num_sentences` | 0.0405 |
| 12 | `embed_pca_28` | 0.0385 |
| 13 | `meta_mean_label_score` | 0.0377 |
| 14 | `Breed_PCA_3` | 0.0345 |
| 15 | `embed_pca_10` | 0.0345 |
| 16 | `embed_pca_13` | 0.0318 |
| 17 | `embed_pca_48` | 0.0310 |
| 18 | `embed_pca_2` | 0.0297 |
| 19 | `embed_pca_12` | 0.0297 |
| 20 | `Fee` | 0.0285 |

#### Feature Group Impact (Mean |SHAP| summed per group) balanced

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 64 | 1.4307 | 59.3% |
| Numeric | 5 | 0.2526 | 10.5% |
| Breed (PCA) | 15 | 0.2234 | 9.3% |
| Categorical (OHE) | 33 | 0.1794 | 7.4% |
| Image Metadata (Google Vision) | 8 | 0.1792 | 7.4% |
| Sentiment (NLP) | 5 | 0.1078 | 4.5% |
| Color (multi-hot) | 7 | 0.0399 | 1.7% |
<!-- feature-analysis-end -->