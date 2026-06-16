# Experiment Results — Mode: `all_4class`

Backbone × PCA grid, each with 15 Optuna trials, 5-fold CV.

## Summary Table (sorted by QWK)

| Backbone | PCA | QWK | Accuracy | F1 macro | F1 weighted | Embed Variance |
| ---------- | ----- | ----- | ---------- | ---------- | ------------- | ---------------- |
| resnet50 | 64 | 0.4095 | 0.4345 | 0.4072 | 0.4187 | 79.7% |
| resnet50 | 16 | 0.4046 | 0.4390 | 0.4124 | 0.4234 | 56.0% |
| efficientnet_b0 | 64 | 0.4023 | 0.4343 | 0.4093 | 0.4203 | 65.8% |
| efficientnet_b0 | 16 | 0.4023 | 0.4343 | 0.4087 | 0.4195 | 44.6% |
| alexnet | 16 | 0.3820 | 0.4260 | 0.4006 | 0.4112 | 38.2% |
| alexnet | 64 | 0.3815 | 0.4241 | 0.3956 | 0.4074 | 58.9% |
| none | 0 | 0.3806 | 0.4213 | 0.3948 | 0.4057 | — |

## Detailed Results

---

### backbone=resnet50  pca=64  (id: `_4class_resnet50_pca64`)

- **QWK:**           0.4095
- **Accuracy:**      0.4345
- **F1 macro:**      0.4072
- **F1 weighted:**   0.4187
- **Embed PCA:**     2048 dims → 64 components (79.7% variance explained)

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) resnet50_64

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 35373.9 | 5.2% |
| 2 | `meta_top_label_score` | 14111.4 | 2.1% |
| 3 | `embed_pca_3` | 13116.3 | 1.9% |
| 4 | `embed_pca_5` | 12978.1 | 1.9% |
| 5 | `meta_num_labels` | 12833.0 | 1.9% |
| 6 | `embed_pca_17` | 12221.4 | 1.8% |
| 7 | `meta_mean_label_score` | 11098.3 | 1.6% |
| 8 | `embed_pca_0` | 10311.3 | 1.5% |
| 9 | `Breed_PCA_3` | 10019.1 | 1.5% |
| 10 | `Sterilized_2` | 9705.3 | 1.4% |
| 11 | `embed_pca_8` | 9385.3 | 1.4% |
| 12 | `embed_pca_19` | 9095.9 | 1.3% |
| 13 | `embed_pca_10` | 8814.7 | 1.3% |
| 14 | `embed_pca_16` | 8760.8 | 1.3% |
| 15 | `embed_pca_14` | 8598.2 | 1.3% |
| 16 | `embed_pca_15` | 8153.6 | 1.2% |
| 17 | `Quantity` | 8025.0 | 1.2% |
| 18 | `embed_pca_23` | 7905.4 | 1.2% |
| 19 | `embed_pca_12` | 7892.4 | 1.2% |
| 20 | `embed_pca_48` | 7831.8 | 1.2% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) resnet50_64

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1781 |
| 2 | `embed_pca_3` | 0.0714 |
| 3 | `embed_pca_5` | 0.0607 |
| 4 | `meta_num_labels` | 0.0509 |
| 5 | `Sterilized_2` | 0.0509 |
| 6 | `Quantity` | 0.0505 |
| 7 | `embed_pca_17` | 0.0482 |
| 8 | `embed_pca_0` | 0.0463 |
| 9 | `meta_top_label_score` | 0.0442 |
| 10 | `num_sentences` | 0.0394 |
| 11 | `meta_mean_label_score` | 0.0388 |
| 12 | `embed_pca_8` | 0.0381 |
| 13 | `embed_pca_16` | 0.0356 |
| 14 | `Breed_PCA_3` | 0.0344 |
| 15 | `Breed_PCA_0` | 0.0330 |
| 16 | `embed_pca_14` | 0.0319 |
| 17 | `embed_pca_10` | 0.0308 |
| 18 | `embed_pca_41` | 0.0305 |
| 19 | `Gender_1` | 0.0305 |
| 20 | `embed_pca_48` | 0.0287 |

#### Feature Group Impact (Mean |SHAP| summed per group) resnet50_64

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 64 | 1.3096 | 57.4% |
| Numeric | 5 | 0.2755 | 12.1% |
| Categorical (OHE) | 33 | 0.2178 | 9.6% |
| Image Metadata (Google Vision) | 8 | 0.1834 | 8.0% |
| Breed (PCA) | 15 | 0.1751 | 7.7% |
| Sentiment (NLP) | 5 | 0.1026 | 4.5% |
| Color (multi-hot) | 7 | 0.0164 | 0.7% |
<!-- feature-analysis-end -->

---

### backbone=resnet50  pca=16  (id: `_4class_resnet50_pca16`)

- **QWK:**           0.4046
- **Accuracy:**      0.4390
- **F1 macro:**      0.4124
- **F1 weighted:**   0.4234
- **Embed PCA:**     2048 dims → 16 components (56.0% variance explained)

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) resnet50_16

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 38479.3 | 7.5% |
| 2 | `embed_pca_3` | 19592.8 | 3.8% |
| 3 | `meta_top_label_score` | 19399.6 | 3.8% |
| 4 | `embed_pca_5` | 18312.3 | 3.6% |
| 5 | `meta_mean_label_score` | 16228.9 | 3.2% |
| 6 | `meta_num_labels` | 16070.6 | 3.1% |
| 7 | `embed_pca_8` | 14567.9 | 2.8% |
| 8 | `embed_pca_0` | 14567.6 | 2.8% |
| 9 | `embed_pca_10` | 14044.0 | 2.7% |
| 10 | `embed_pca_12` | 13922.2 | 2.7% |
| 11 | `embed_pca_2` | 13817.8 | 2.7% |
| 12 | `embed_pca_9` | 13006.0 | 2.5% |
| 13 | `embed_pca_13` | 12856.1 | 2.5% |
| 14 | `embed_pca_6` | 12792.7 | 2.5% |
| 15 | `embed_pca_14` | 12674.2 | 2.5% |
| 16 | `embed_pca_7` | 12591.0 | 2.5% |
| 17 | `embed_pca_15` | 12504.3 | 2.4% |
| 18 | `embed_pca_11` | 12150.2 | 2.4% |
| 19 | `Breed_PCA_3` | 11578.4 | 2.3% |
| 20 | `sentiment_magnitude` | 11531.2 | 2.2% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) resnet50_16

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1791 |
| 2 | `embed_pca_3` | 0.0748 |
| 3 | `embed_pca_5` | 0.0607 |
| 4 | `meta_top_label_score` | 0.0530 |
| 5 | `meta_num_labels` | 0.0527 |
| 6 | `Sterilized_2` | 0.0516 |
| 7 | `Quantity` | 0.0490 |
| 8 | `embed_pca_0` | 0.0462 |
| 9 | `meta_mean_label_score` | 0.0411 |
| 10 | `Breed_PCA_3` | 0.0381 |
| 11 | `embed_pca_8` | 0.0381 |
| 12 | `Breed_PCA_0` | 0.0366 |
| 13 | `embed_pca_2` | 0.0329 |
| 14 | `num_sentences` | 0.0328 |
| 15 | `embed_pca_10` | 0.0315 |
| 16 | `Gender_1` | 0.0310 |
| 17 | `embed_pca_14` | 0.0306 |
| 18 | `embed_pca_13` | 0.0278 |
| 19 | `embed_pca_9` | 0.0273 |
| 20 | `embed_pca_15` | 0.0266 |

#### Feature Group Impact (Mean |SHAP| summed per group) resnet50_16

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 16 | 0.5115 | 34.0% |
| Numeric | 5 | 0.2730 | 18.2% |
| Categorical (OHE) | 33 | 0.2183 | 14.5% |
| Image Metadata (Google Vision) | 8 | 0.2050 | 13.6% |
| Breed (PCA) | 15 | 0.1768 | 11.8% |
| Sentiment (NLP) | 5 | 0.1023 | 6.8% |
| Color (multi-hot) | 7 | 0.0173 | 1.1% |
<!-- feature-analysis-end -->

---

### backbone=efficientnet_b0  pca=64  (id: `_4class_efficientnet_b0_pca64`)

- **QWK:**           0.4023
- **Accuracy:**      0.4343
- **F1 macro:**      0.4093
- **F1 weighted:**   0.4203
- **Embed PCA:**     1280 dims → 64 components (65.8% variance explained)

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) efficientnet_b0_64

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 14098.7 | 3.8% |
| 2 | `embed_pca_5` | 6831.9 | 1.8% |
| 3 | `embed_pca_4` | 6626.1 | 1.8% |
| 4 | `meta_top_label_score` | 6398.7 | 1.7% |
| 5 | `meta_mean_label_score` | 5765.9 | 1.5% |
| 6 | `meta_num_labels` | 5534.6 | 1.5% |
| 7 | `embed_pca_27` | 5341.8 | 1.4% |
| 8 | `embed_pca_9` | 4995.0 | 1.3% |
| 9 | `embed_pca_2` | 4867.0 | 1.3% |
| 10 | `embed_pca_19` | 4778.7 | 1.3% |
| 11 | `embed_pca_13` | 4665.6 | 1.2% |
| 12 | `embed_pca_60` | 4623.8 | 1.2% |
| 13 | `embed_pca_8` | 4570.4 | 1.2% |
| 14 | `embed_pca_7` | 4499.2 | 1.2% |
| 15 | `embed_pca_12` | 4456.2 | 1.2% |
| 16 | `embed_pca_42` | 4425.4 | 1.2% |
| 17 | `embed_pca_0` | 4424.5 | 1.2% |
| 18 | `embed_pca_57` | 4417.7 | 1.2% |
| 19 | `embed_pca_41` | 4412.4 | 1.2% |
| 20 | `embed_pca_15` | 4317.8 | 1.2% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) efficientnet_b0_64

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1837 |
| 2 | `embed_pca_4` | 0.0779 |
| 3 | `embed_pca_5` | 0.0758 |
| 4 | `Sterilized_2` | 0.0542 |
| 5 | `meta_num_labels` | 0.0505 |
| 6 | `Quantity` | 0.0472 |
| 7 | `meta_top_label_score` | 0.0453 |
| 8 | `embed_pca_27` | 0.0453 |
| 9 | `meta_mean_label_score` | 0.0419 |
| 10 | `embed_pca_2` | 0.0406 |
| 11 | `embed_pca_9` | 0.0390 |
| 12 | `num_sentences` | 0.0358 |
| 13 | `embed_pca_0` | 0.0339 |
| 14 | `Breed_PCA_3` | 0.0339 |
| 15 | `embed_pca_13` | 0.0335 |
| 16 | `Gender_1` | 0.0325 |
| 17 | `embed_pca_8` | 0.0325 |
| 18 | `embed_pca_63` | 0.0321 |
| 19 | `embed_pca_19` | 0.0313 |
| 20 | `embed_pca_60` | 0.0302 |

#### Feature Group Impact (Mean |SHAP| summed per group) efficientnet_b0_64

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 64 | 1.5369 | 59.3% |
| Numeric | 5 | 0.2884 | 11.1% |
| Categorical (OHE) | 33 | 0.2321 | 9.0% |
| Image Metadata (Google Vision) | 8 | 0.2031 | 7.8% |
| Breed (PCA) | 15 | 0.1878 | 7.2% |
| Sentiment (NLP) | 5 | 0.1199 | 4.6% |
| Color (multi-hot) | 7 | 0.0218 | 0.8% |
<!-- feature-analysis-end -->

---

### backbone=efficientnet_b0  pca=16  (id: `_4class_efficientnet_b0_pca16`)

- **QWK:**           0.4023
- **Accuracy:**      0.4343
- **F1 macro:**      0.4087
- **F1 weighted:**   0.4195
- **Embed PCA:**     1280 dims → 16 components (44.6% variance explained)

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) efficientnet_b0_16

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 15199.0 | 5.4% |
| 2 | `embed_pca_5` | 10909.4 | 3.9% |
| 3 | `embed_pca_4` | 10152.4 | 3.6% |
| 4 | `meta_top_label_score` | 9553.5 | 3.4% |
| 5 | `embed_pca_2` | 8487.5 | 3.0% |
| 6 | `meta_mean_label_score` | 8380.9 | 3.0% |
| 7 | `embed_pca_13` | 8369.5 | 3.0% |
| 8 | `embed_pca_9` | 8356.6 | 3.0% |
| 9 | `embed_pca_15` | 8040.4 | 2.9% |
| 10 | `embed_pca_8` | 7850.2 | 2.8% |
| 11 | `embed_pca_7` | 7789.7 | 2.8% |
| 12 | `embed_pca_12` | 7783.9 | 2.8% |
| 13 | `embed_pca_0` | 7582.8 | 2.7% |
| 14 | `meta_num_labels` | 7537.5 | 2.7% |
| 15 | `embed_pca_6` | 7494.6 | 2.7% |
| 16 | `embed_pca_3` | 7474.2 | 2.7% |
| 17 | `meta_dom_color_frac` | 7185.9 | 2.6% |
| 18 | `embed_pca_14` | 7132.7 | 2.5% |
| 19 | `embed_pca_11` | 6976.7 | 2.5% |
| 20 | `embed_pca_1` | 6919.9 | 2.5% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) efficientnet_b0_16

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1913 |
| 2 | `embed_pca_4` | 0.0870 |
| 3 | `embed_pca_5` | 0.0866 |
| 4 | `meta_top_label_score` | 0.0583 |
| 5 | `Sterilized_2` | 0.0553 |
| 6 | `meta_num_labels` | 0.0552 |
| 7 | `Quantity` | 0.0538 |
| 8 | `embed_pca_2` | 0.0492 |
| 9 | `embed_pca_0` | 0.0487 |
| 10 | `embed_pca_9` | 0.0471 |
| 11 | `embed_pca_13` | 0.0434 |
| 12 | `meta_mean_label_score` | 0.0428 |
| 13 | `num_sentences` | 0.0416 |
| 14 | `embed_pca_8` | 0.0407 |
| 15 | `embed_pca_15` | 0.0395 |
| 16 | `Breed_PCA_3` | 0.0378 |
| 17 | `Gender_1` | 0.0375 |
| 18 | `embed_pca_7` | 0.0372 |
| 19 | `embed_pca_6` | 0.0354 |
| 20 | `avg_sentence_score` | 0.0343 |

#### Feature Group Impact (Mean |SHAP| summed per group) efficientnet_b0_16

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 16 | 0.6838 | 35.4% |
| Numeric | 5 | 0.3091 | 16.0% |
| Categorical (OHE) | 33 | 0.2836 | 14.7% |
| Image Metadata (Google Vision) | 8 | 0.2574 | 13.3% |
| Breed (PCA) | 15 | 0.2091 | 10.8% |
| Sentiment (NLP) | 5 | 0.1433 | 7.4% |
| Color (multi-hot) | 7 | 0.0435 | 2.3% |
<!-- feature-analysis-end -->

---

### backbone=alexnet  pca=16  (id: `_4class_alexnet_pca16`)

- **QWK:**           0.3820
- **Accuracy:**      0.4260
- **F1 macro:**      0.4006
- **F1 weighted:**   0.4112
- **Embed PCA:**     4096 dims → 16 components (38.2% variance explained)

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) alexnet_16

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 9387.8 | 8.0% |
| 2 | `meta_top_label_score` | 4349.0 | 3.7% |
| 3 | `meta_mean_label_score` | 4315.9 | 3.7% |
| 4 | `embed_pca_6` | 3688.6 | 3.2% |
| 5 | `meta_num_labels` | 3665.5 | 3.1% |
| 6 | `embed_pca_8` | 3441.0 | 2.9% |
| 7 | `embed_pca_9` | 3310.1 | 2.8% |
| 8 | `Breed_PCA_3` | 3304.4 | 2.8% |
| 9 | `embed_pca_1` | 3250.7 | 2.8% |
| 10 | `embed_pca_14` | 3041.1 | 2.6% |
| 11 | `embed_pca_13` | 3029.9 | 2.6% |
| 12 | `embed_pca_11` | 2986.9 | 2.6% |
| 13 | `embed_pca_4` | 2948.0 | 2.5% |
| 14 | `embed_pca_15` | 2866.0 | 2.4% |
| 15 | `embed_pca_10` | 2861.8 | 2.4% |
| 16 | `embed_pca_3` | 2646.0 | 2.3% |
| 17 | `embed_pca_2` | 2617.5 | 2.2% |
| 18 | `embed_pca_7` | 2600.5 | 2.2% |
| 19 | `embed_pca_5` | 2597.4 | 2.2% |
| 20 | `embed_pca_0` | 2565.3 | 2.2% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) alexnet_16

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1948 |
| 2 | `embed_pca_6` | 0.0555 |
| 3 | `Sterilized_2` | 0.0538 |
| 4 | `meta_mean_label_score` | 0.0525 |
| 5 | `meta_num_labels` | 0.0522 |
| 6 | `meta_top_label_score` | 0.0518 |
| 7 | `Quantity` | 0.0516 |
| 8 | `embed_pca_1` | 0.0516 |
| 9 | `Breed_PCA_3` | 0.0460 |
| 10 | `embed_pca_8` | 0.0409 |
| 11 | `num_sentences` | 0.0390 |
| 12 | `embed_pca_13` | 0.0370 |
| 13 | `Gender_1` | 0.0359 |
| 14 | `embed_pca_5` | 0.0356 |
| 15 | `embed_pca_15` | 0.0343 |
| 16 | `embed_pca_4` | 0.0343 |
| 17 | `Sterilized_1` | 0.0316 |
| 18 | `Breed_PCA_0` | 0.0294 |
| 19 | `embed_pca_12` | 0.0288 |
| 20 | `embed_pca_9` | 0.0284 |

#### Feature Group Impact (Mean |SHAP| summed per group) alexnet_16

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 16 | 0.5177 | 30.8% |
| Numeric | 5 | 0.3020 | 18.0% |
| Categorical (OHE) | 33 | 0.2500 | 14.9% |
| Image Metadata (Google Vision) | 8 | 0.2376 | 14.1% |
| Breed (PCA) | 15 | 0.2154 | 12.8% |
| Sentiment (NLP) | 5 | 0.1273 | 7.6% |
| Color (multi-hot) | 7 | 0.0324 | 1.9% |
<!-- feature-analysis-end -->

---

### backbone=alexnet  pca=64  (id: `_4class_alexnet_pca64`)

- **QWK:**           0.3815
- **Accuracy:**      0.4241
- **F1 macro:**      0.3956
- **F1 weighted:**   0.4074
- **Embed PCA:**     4096 dims → 64 components (58.9% variance explained)

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) alexnet_64

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 11283.0 | 6.3% |
| 2 | `meta_top_label_score` | 4049.1 | 2.3% |
| 3 | `Sterilized_2` | 3447.5 | 1.9% |
| 4 | `meta_num_labels` | 3287.8 | 1.8% |
| 5 | `meta_mean_label_score` | 3050.4 | 1.7% |
| 6 | `embed_pca_6` | 2807.8 | 1.6% |
| 7 | `embed_pca_43` | 2735.4 | 1.5% |
| 8 | `Breed_PCA_0` | 2670.2 | 1.5% |
| 9 | `embed_pca_8` | 2499.7 | 1.4% |
| 10 | `Quantity` | 2418.4 | 1.4% |
| 11 | `embed_pca_4` | 2291.4 | 1.3% |
| 12 | `embed_pca_27` | 2260.6 | 1.3% |
| 13 | `embed_pca_9` | 2258.8 | 1.3% |
| 14 | `Breed_PCA_3` | 2195.7 | 1.2% |
| 15 | `embed_pca_1` | 2188.6 | 1.2% |
| 16 | `embed_pca_36` | 2155.9 | 1.2% |
| 17 | `embed_pca_55` | 2145.2 | 1.2% |
| 18 | `PhotoAmt` | 2075.1 | 1.2% |
| 19 | `embed_pca_5` | 2043.9 | 1.1% |
| 20 | `embed_pca_31` | 1998.6 | 1.1% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) alexnet_64

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.1796 |
| 2 | `Sterilized_2` | 0.0542 |
| 3 | `Quantity` | 0.0468 |
| 4 | `meta_num_labels` | 0.0409 |
| 5 | `meta_mean_label_score` | 0.0407 |
| 6 | `meta_top_label_score` | 0.0406 |
| 7 | `Breed_PCA_0` | 0.0390 |
| 8 | `embed_pca_6` | 0.0366 |
| 9 | `num_sentences` | 0.0335 |
| 10 | `PhotoAmt` | 0.0329 |
| 11 | `embed_pca_43` | 0.0327 |
| 12 | `embed_pca_1` | 0.0313 |
| 13 | `Breed_PCA_1` | 0.0308 |
| 14 | `embed_pca_8` | 0.0302 |
| 15 | `Breed_PCA_3` | 0.0264 |
| 16 | `Sterilized_1` | 0.0244 |
| 17 | `embed_pca_4` | 0.0229 |
| 18 | `embed_pca_27` | 0.0220 |
| 19 | `Fee` | 0.0217 |
| 20 | `Breed_PCA_2` | 0.0200 |

#### Feature Group Impact (Mean |SHAP| summed per group) alexnet_64

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Image Embeddings (CNN PCA) | 64 | 0.9522 | 50.8% |
| Numeric | 5 | 0.2810 | 15.0% |
| Categorical (OHE) | 33 | 0.1960 | 10.5% |
| Breed (PCA) | 15 | 0.1911 | 10.2% |
| Image Metadata (Google Vision) | 8 | 0.1627 | 8.7% |
| Sentiment (NLP) | 5 | 0.0843 | 4.5% |
| Color (multi-hot) | 7 | 0.0073 | 0.4% |
<!-- feature-analysis-end -->

---

### backbone=none  pca=0  (id: `_4class_noembed`)

- **QWK:**           0.3806
- **Accuracy:**      0.4213
- **F1 macro:**      0.3948
- **F1 weighted:**   0.4057
- **Embed PCA:**     none (no embeddings)

<!-- feature-analysis-start -->
#### Top-20 Feature Importance (by gain) none

| Rank | Feature | Gain | % of Total |
| ------ | --------- | ------ | ------------ |
| 1 | `Age` | 22360.9 | 10.3% |
| 2 | `meta_top_label_score` | 14506.0 | 6.6% |
| 3 | `meta_mean_label_score` | 13274.0 | 6.1% |
| 4 | `meta_num_labels` | 10775.7 | 4.9% |
| 5 | `meta_dom_color_frac` | 10457.9 | 4.8% |
| 6 | `avg_sentence_score` | 9626.2 | 4.4% |
| 7 | `meta_dom_color_R` | 9464.1 | 4.3% |
| 8 | `meta_dom_color_B` | 8688.3 | 4.0% |
| 9 | `sentiment_magnitude` | 8615.1 | 3.9% |
| 10 | `avg_sentence_magnitude` | 8045.4 | 3.7% |
| 11 | `meta_dom_color_G` | 7596.5 | 3.5% |
| 12 | `Breed_PCA_3` | 7301.1 | 3.3% |
| 13 | `Breed_PCA_0` | 5340.1 | 2.4% |
| 14 | `num_sentences` | 5314.6 | 2.4% |
| 15 | `Sterilized_2` | 5080.6 | 2.3% |
| 16 | `Quantity` | 5056.6 | 2.3% |
| 17 | `PhotoAmt` | 4125.6 | 1.9% |
| 18 | `Breed_PCA_1` | 4067.5 | 1.9% |
| 19 | `Breed_PCA_12` | 3281.9 | 1.5% |
| 20 | `Fee` | 2788.9 | 1.3% |

#### Top-20 Features by Mean |SHAP| (subsample n=500) none

| Rank | Feature | Mean &#124;SHAP&#124; |
| ------ | --------- | ---------------------- |
| 1 | `Age` | 0.2107 |
| 2 | `meta_mean_label_score` | 0.0667 |
| 3 | `meta_top_label_score` | 0.0659 |
| 4 | `Sterilized_2` | 0.0588 |
| 5 | `Quantity` | 0.0577 |
| 6 | `meta_num_labels` | 0.0558 |
| 7 | `Breed_PCA_3` | 0.0469 |
| 8 | `num_sentences` | 0.0395 |
| 9 | `Gender_1` | 0.0394 |
| 10 | `Breed_PCA_0` | 0.0385 |
| 11 | `avg_sentence_score` | 0.0361 |
| 12 | `sentiment_magnitude` | 0.0360 |
| 13 | `meta_dom_color_frac` | 0.0314 |
| 14 | `Fee` | 0.0300 |
| 15 | `Sterilized_1` | 0.0300 |
| 16 | `meta_dom_color_B` | 0.0280 |
| 17 | `avg_sentence_magnitude` | 0.0273 |
| 18 | `PhotoAmt` | 0.0267 |
| 19 | `meta_dom_color_R` | 0.0256 |
| 20 | `Breed_PCA_1` | 0.0240 |

#### Feature Group Impact (Mean |SHAP| summed per group) none

| Group | # Features | Sum Mean |SHAP| | % of Total |
|-------|-----------|----------------|------------|
| Numeric | 5 | 0.3275 | 23.6% |
| Image Metadata (Google Vision) | 8 | 0.3028 | 21.9% |
| Categorical (OHE) | 33 | 0.2999 | 21.6% |
| Breed (PCA) | 15 | 0.2403 | 17.3% |
| Sentiment (NLP) | 5 | 0.1486 | 10.7% |
| Color (multi-hot) | 7 | 0.0660 | 4.8% |
<!-- feature-analysis-end -->
