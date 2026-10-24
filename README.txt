UK-DALE V4: Adaptive Score Fusion

Architecture:
1) Preprocessing + six features
2) Layer 1 LSTM Autoencoder
3) Candidate screening
4) Layer 2 LSTM + self-attention
5) Fixed score fusion baseline
6) Hierarchical refinement
7) Proposed adaptive score fusion
8) Precision / Recall / F1 / FPR / PR-AUC / ROC-AUC

Important:
- Dynamic thresholding is removed from the proposed method.
- Context changes the relative weight of AE vs temporal score.
- Candidate threshold is calibrated on a synthetic validation set.
- Test is held out for final comparison.
- Synthetic anomaly labels are never used as model inputs.
