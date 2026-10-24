# UK-DALE Research Pipeline

## Architecture

UK-DALE
-> preprocessing
-> six energy/context features
-> 60-minute windows
-> Layer 1 LSTM Autoencoder
-> candidate selection
-> Layer 2 LSTM + Self-Attention
-> score fusion
-> Layer 3 Context-Aware Robust Dynamic Threshold
-> final anomaly decision

## Ablation models

M0:
LSTM Autoencoder

M1:
LSTM Autoencoder + Attention

M2:
Hierarchical candidate refinement

M3:
Proposed context-aware hierarchical dynamic threshold

## Fast experiment

The current configuration intentionally uses:

- 20,000 training windows
- 5,000 validation windows
- 30,000 test windows
- sequence length = 60
- batch size = 512
- up to 8 epochs with early stopping

This is for fast research iteration. Increase the window counts only after
the full pipeline is stable.

## Evaluation

`main.py` injects deterministic synthetic anomalies ONLY into the evaluation
copy of the test data. The models are trained only on normal training data.

The script reports:

- Precision
- Recall
- F1
- Accuracy
- False Positive Rate
- False Negative Rate
- candidate reduction
- runtime

Results are saved to:

- ablation_results.csv
- synthetic_anomaly_events.csv
- best_layer1.keras
- best_layer2.keras
