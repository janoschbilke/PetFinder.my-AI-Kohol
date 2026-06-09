# Changelog

## 05.06.2026

1. Try LGBM instead of Random Forest
  - better with large / sparse feature vectors
  - hyperparameter optimization with optuna
    - num_leaves = 30, max_depth = 5 -> looks like stable result in tuning process
  - 2000 boosting rounds with 100 round early stopping
    - training process early stopped after ~700 rounds -> convergence?
  - result: QWK goes from 0.33 to 0.37

2. Apply PCA to breed vector
  - 307 one hot encoded cols -> 15 cols (80% explained variance)
  - tested with 80% to 95% explained variance -> 80% yielded best results in training
  - results: 
    - LGBM only jumps up to QWK 0.375
    - Random Forest improves more: 0.33 -> 0.357

### TODOs:
- instead of pca, maybe use target encoding with average adoption probability of breed per class (if training data is sufficient -> investigate)
- use smaller dataset for training first (only dogs)
- improve random forest implementation
- Use image embedding in addition to tabular data !!!

### Feedback
- simpler approach, simplify dataset (only dog, isolate certain output classes)
- take look at pca vector, if we can classify breeds from that (cat vs. dog)
- get more data understanding
- promising result first: ~80% precision