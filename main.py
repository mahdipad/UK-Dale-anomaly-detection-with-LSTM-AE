import os
import random
import time

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "1"
)

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocessing import (
    UKDALELoader,
    FEATURE_COLUMNS
)

from model import (
    build_initial_autoencoder,
    build_temporal_attention_autoencoder
)

from train import train_model

from detect import (
    reconstruction_error_by_feature,
    normalize_score,
    fuse_scores,
    adaptive_fusion,
    print_statistics,
    plot_error_distribution,
    plot_anomalies
)

from evaluate import (
    inject_synthetic_anomalies,
    endpoint_window_labels,
    binary_metrics,
    ranking_metrics,
    print_metrics,
    evaluation_table,
    tune_threshold,
    tune_candidate_percentile
)


# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = (
    r"C:\Users\Mahdi\PycharmProjects\PythonProject1"
    r"\dale dataset\ukdale\data"
)

HOUSE = 1
SEQUENCE_LENGTH = 60

# Fast research iteration.
TRAIN_WINDOWS = 24000
VALIDATION_WINDOWS = 6000
TEST_WINDOWS = 30000

EPOCHS = 8
BATCH_SIZE = 512

AE_WEIGHT = 0.5
TEMPORAL_WEIGHT = 0.5

ADAPTIVE_MIN_WEIGHT = 0.20
ADAPTIVE_MAX_WEIGHT = 0.80
ADAPTIVE_STRENGTH = 0.35

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# HELPERS
# ==========================================================

def make_windows(
    scaled_data,
    count,
    sequence_length
):
    max_start = (
        len(scaled_data)
        - sequence_length
    )

    if max_start < 0:
        raise ValueError(
            "Not enough rows."
        )

    starts = np.linspace(
        0,
        max_start,
        min(count, max_start + 1),
        dtype=np.int64
    )

    starts = np.unique(starts)

    return (
        loader.create_sequences_at_indices(
            scaled_data,
            starts,
            sequence_length
        ),
        starts
    )


def context_for_starts(
    df,
    starts,
    sequence_length
):
    ends = (
        starts
        + sequence_length
        - 1
    )

    return (
        df.iloc[
            ends
        ][FEATURE_COLUMNS]
        .reset_index(drop=True)
    )


def scale_eval_dataframe(
    scaler,
    df
):
    return (
        scaler
        .transform(
            df[FEATURE_COLUMNS]
            .to_numpy(
                dtype=np.float32
            )
        )
        .astype(np.float32)
    )


def add_ranking_metrics(
    metrics,
    y_true,
    score
):
    metrics.update(
        ranking_metrics(
            y_true,
            score
        )
    )
    return metrics


# ==========================================================
# START
# ==========================================================

total_start = time.perf_counter()

print(
    "\n=========================================="
)
print(
    "UK-DALE RESEARCH PIPELINE V4"
)
print(
    "Adaptive Score Fusion"
)
print(
    "=========================================="
)


# ==========================================================
# 1. DATA
# ==========================================================

loader = UKDALELoader(
    DATA_PATH
)

df = loader.load_aggregate(
    house=HOUSE
)

df = loader.resample_data(
    df,
    freq="1min",
    method="mean"
)

df = loader.handle_missing(
    df,
    method="ffill"
)

df = loader.add_features(
    df,
    rolling_window=5
)

print(
    "\nFeatures:",
    df.columns.tolist()
)

print(
    "Resampled:",
    df.shape
)


# ==========================================================
# 2. SPLIT
# ==========================================================

train_df, test_df = (
    loader.train_test_split(
        df,
        train_ratio=0.8
    )
)

train_size = int(
    len(train_df) * 0.9
)

train_part = (
    train_df.iloc[
        :train_size
    ].copy()
)

val_part = (
    train_df.iloc[
        train_size:
    ].copy()
)


# ==========================================================
# 3. SCALE
# ==========================================================

train_scaled, val_scaled, scaler = (
    loader.normalize(
        train_part,
        val_part
    )
)

test_scaled = (
    loader.scale_with_existing_scaler(
        scaler,
        test_df
    )
)


# ==========================================================
# 4. REPRESENTATIVE WINDOWS
# ==========================================================

X_train, train_starts = make_windows(
    train_scaled,
    TRAIN_WINDOWS,
    SEQUENCE_LENGTH
)

X_val, val_starts = make_windows(
    val_scaled,
    VALIDATION_WINDOWS,
    SEQUENCE_LENGTH
)

X_test_normal, test_starts = make_windows(
    test_scaled,
    TEST_WINDOWS,
    SEQUENCE_LENGTH
)

n_features = X_train.shape[2]

print(
    "\n========== WINDOWS =========="
)

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test_normal.shape
)


# ==========================================================
# 5. CONTEXT WINDOWS
# ==========================================================

val_context = context_for_starts(
    val_part,
    val_starts,
    SEQUENCE_LENGTH
)

test_context = context_for_starts(
    test_df,
    test_starts,
    SEQUENCE_LENGTH
)


# ==========================================================
# 6. SYNTHETIC VALIDATION
# ==========================================================

val_rows = (
    VALIDATION_WINDOWS
    + SEQUENCE_LENGTH
    - 1
)

val_eval_raw = (
    val_part.iloc[
        :val_rows
    ][["aggregate"]]
    .copy()
)

(
    val_eval_raw,
    val_point_labels,
    val_events
) = inject_synthetic_anomalies(
    val_eval_raw,
    n_events=10,
    seed=SEED + 1
)

val_eval_features = loader.add_features(
    val_eval_raw[["aggregate"]],
    rolling_window=5
)

val_eval_scaled = scale_eval_dataframe(
    scaler,
    val_eval_features
)

X_val_eval, val_eval_starts = make_windows(
    val_eval_scaled,
    VALIDATION_WINDOWS,
    SEQUENCE_LENGTH
)

y_val_true = endpoint_window_labels(
    val_point_labels,
    val_eval_starts,
    SEQUENCE_LENGTH
)

val_eval_context = context_for_starts(
    val_eval_features,
    val_eval_starts,
    SEQUENCE_LENGTH
)


# ==========================================================
# 7. SYNTHETIC TEST
# ==========================================================

test_rows = (
    TEST_WINDOWS
    + SEQUENCE_LENGTH
    - 1
)

test_eval_raw = (
    test_df.iloc[
        :test_rows
    ][["aggregate"]]
    .copy()
)

(
    test_eval_raw,
    test_point_labels,
    test_events
) = inject_synthetic_anomalies(
    test_eval_raw,
    n_events=16,
    seed=SEED + 2
)

test_eval_features = loader.add_features(
    test_eval_raw[["aggregate"]],
    rolling_window=5
)

test_eval_scaled = scale_eval_dataframe(
    scaler,
    test_eval_features
)

X_test_eval, test_eval_starts = make_windows(
    test_eval_scaled,
    TEST_WINDOWS,
    SEQUENCE_LENGTH
)

y_test_true = endpoint_window_labels(
    test_point_labels,
    test_eval_starts,
    SEQUENCE_LENGTH
)

test_eval_context = context_for_starts(
    test_eval_features,
    test_eval_starts,
    SEQUENCE_LENGTH
)

print(
    "\nValidation anomaly windows:",
    int(y_val_true.sum())
)

print(
    "Test anomaly windows:",
    int(y_test_true.sum())
)


# ==========================================================
# 8. LAYER 1
# ==========================================================

print(
    "\n=========================================="
)
print(
    "LAYER 1: LSTM AUTOENCODER"
)
print(
    "=========================================="
)

layer1 = build_initial_autoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=n_features
)

train_model(
    layer1,
    X_train,
    X_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    checkpoint_path="best_layer1_v4.keras"
)

val_error1 = (
    reconstruction_error_by_feature(
        layer1,
        X_val_eval
    )[:, 0]
)

test_error1 = (
    reconstruction_error_by_feature(
        layer1,
        X_test_eval
    )[:, 0]
)

m0_threshold, _ = tune_threshold(
    y_val_true,
    val_error1,
    min_precision=0.05
)

m0_pred = (
    test_error1
    >= m0_threshold
).astype(np.int8)

m0 = binary_metrics(
    y_test_true,
    m0_pred
)

m0 = add_ranking_metrics(
    m0,
    y_test_true,
    test_error1
)

print_metrics(
    "M0 - LSTM Autoencoder",
    m0
)

print(
    "M0 threshold:",
    m0_threshold
)


# ==========================================================
# 9. CANDIDATE SCREENING
# ==========================================================

candidate_calibration = (
    tune_candidate_percentile(
        y_val_true,
        val_error1,
        min_recall=0.90
    )
)

candidate_threshold = (
    candidate_calibration[
        "threshold"
    ]
)

candidate_mask = (
    test_error1
    >= candidate_threshold
)

candidate_indices = np.where(
    candidate_mask
)[0]

print(
    "\n========== CANDIDATE SCREENING =========="
)

print(
    "Candidate percentile:",
    candidate_calibration[
        "percentile"
    ]
)

print(
    "Candidate threshold:",
    candidate_threshold
)

print(
    "Candidate count:",
    len(candidate_indices)
)

print(
    "Candidate rate:",
    candidate_mask.mean() * 100
)

print(
    "Candidate validation recall:",
    candidate_calibration[
        "recall"
    ]
)


# ==========================================================
# 10. LAYER 2
# ==========================================================

print(
    "\n=========================================="
)
print(
    "LAYER 2: LSTM + ATTENTION"
)
print(
    "=========================================="
)

layer2 = build_temporal_attention_autoencoder(
    sequence_length=SEQUENCE_LENGTH,
    n_features=n_features
)

train_model(
    layer2,
    X_train,
    X_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    checkpoint_path="best_layer2_v4.keras"
)

val_error2 = (
    reconstruction_error_by_feature(
        layer2,
        X_val_eval
    )[:, 0]
)

test_error2 = (
    reconstruction_error_by_feature(
        layer2,
        X_test_eval
    )[:, 0]
)


# ==========================================================
# 11. FIXED FUSION BASELINE
# ==========================================================

val_ae_score, ae_scale = normalize_score(
    val_error1,
    val_error1
)

val_temporal_score, temporal_scale = (
    normalize_score(
        val_error2,
        val_error2
    )
)

test_ae_score, _ = normalize_score(
    test_error1,
    val_error1
)

test_temporal_score, _ = normalize_score(
    test_error2,
    val_error2
)

val_fixed_fused = fuse_scores(
    val_ae_score,
    val_temporal_score,
    AE_WEIGHT,
    TEMPORAL_WEIGHT
)

test_fixed_fused = fuse_scores(
    test_ae_score,
    test_temporal_score,
    AE_WEIGHT,
    TEMPORAL_WEIGHT
)

fixed_threshold, _ = tune_threshold(
    y_val_true,
    val_fixed_fused,
    min_precision=0.05
)

fixed_pred = (
    test_fixed_fused
    >= fixed_threshold
).astype(np.int8)

fixed_metrics = binary_metrics(
    y_test_true,
    fixed_pred
)

fixed_metrics = add_ranking_metrics(
    fixed_metrics,
    y_test_true,
    test_fixed_fused
)

print_metrics(
    "M1 - Fixed Fusion",
    fixed_metrics
)


# ==========================================================
# 12. HIERARCHICAL REFINEMENT
# ==========================================================

print(
    "\n=========================================="
)
print(
    "M2: HIERARCHICAL REFINEMENT"
)
print(
    "=========================================="
)

# Default Layer-1 score for every sample.
hierarchical_test_score = (
    test_ae_score.copy()
)

# Validation hierarchical score.
hierarchical_val_score = (
    val_ae_score.copy()
)

# Recompute validation candidate indices using the calibrated
# Layer-1 threshold.
val_candidate_mask = (
    val_error1
    >= candidate_threshold
)

val_candidate_indices = np.where(
    val_candidate_mask
)[0]


if len(candidate_indices) > 0:

    candidate_X = (
        X_test_eval[
            candidate_indices
        ]
    )

    candidate_error2 = (
        reconstruction_error_by_feature(
            layer2,
            candidate_X
        )[:, 0]
    )

    candidate_temporal_score, _ = (
        normalize_score(
            candidate_error2,
            val_error2
        )
    )

    candidate_ae_score = (
        test_ae_score[
            candidate_indices
        ]
    )

    candidate_fixed_fused = fuse_scores(
        candidate_ae_score,
        candidate_temporal_score,
        AE_WEIGHT,
        TEMPORAL_WEIGHT
    )

    hierarchical_test_score[
        candidate_indices
    ] = candidate_fixed_fused


if len(val_candidate_indices) > 0:

    val_candidate_X = (
        X_val_eval[
            val_candidate_indices
        ]
    )

    val_candidate_error2 = (
        reconstruction_error_by_feature(
            layer2,
            val_candidate_X
        )[:, 0]
    )

    val_candidate_temporal_score, _ = (
        normalize_score(
            val_candidate_error2,
            val_error2
        )
    )

    val_candidate_ae_score = (
        val_ae_score[
            val_candidate_indices
        ]
    )

    val_candidate_fused = fuse_scores(
        val_candidate_ae_score,
        val_candidate_temporal_score,
        AE_WEIGHT,
        TEMPORAL_WEIGHT
    )

    hierarchical_val_score[
        val_candidate_indices
    ] = val_candidate_fused


hierarchical_threshold, _ = tune_threshold(
    y_val_true,
    hierarchical_val_score,
    min_precision=0.05
)

m2_pred = (
    hierarchical_test_score
    >= hierarchical_threshold
).astype(np.int8)

m2 = binary_metrics(
    y_test_true,
    m2_pred
)

m2 = add_ranking_metrics(
    m2,
    y_test_true,
    hierarchical_test_score
)

print_metrics(
    "M2 - Hierarchical",
    m2
)

print(
    "M2 threshold:",
    hierarchical_threshold
)


# ==========================================================
# 13. M3 ADAPTIVE SCORE FUSION
# ==========================================================

print(
    "\n=========================================="
)
print(
    "M3: ADAPTIVE SCORE FUSION"
)
print(
    "=========================================="
)

# Adaptive fusion uses context only to adjust the
# relative weight of reconstruction and attention scores.
# No dynamic threshold is used.

# IMPORTANT:
# Adaptive fusion works on WINDOW-LEVEL scores. Therefore context
# must also be WINDOW-LEVEL (one row per evaluated window), not
# row-level feature data. Using val_eval_features/test_eval_features
# here creates 6059/30059 context rows for 6000/30000 scores.
val_context_features = (
    val_eval_context[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )
)

test_context_features = (
    test_eval_context[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )
)

val_adaptive_score, val_ae_weights, _ = (
    adaptive_fusion(
        ae_score=val_ae_score,
        temporal_score=val_temporal_score,
        context_features=val_context_features,
        validation_ae_score=val_ae_score,
        validation_temporal_score=val_temporal_score,
        base_weight=AE_WEIGHT,
        min_weight=ADAPTIVE_MIN_WEIGHT,
        max_weight=ADAPTIVE_MAX_WEIGHT,
        context_strength=ADAPTIVE_STRENGTH
    )
)

test_adaptive_score_all, test_ae_weights, _ = (
    adaptive_fusion(
        ae_score=test_ae_score,
        temporal_score=test_temporal_score,
        context_features=test_context_features,
        validation_ae_score=val_ae_score,
        validation_temporal_score=val_temporal_score,
        base_weight=AE_WEIGHT,
        min_weight=ADAPTIVE_MIN_WEIGHT,
        max_weight=ADAPTIVE_MAX_WEIGHT,
        context_strength=ADAPTIVE_STRENGTH
    )
)

# Hierarchical adaptive version:
hierarchical_adaptive_test_score = (
    test_adaptive_score_all.copy()
)

hierarchical_adaptive_val_score = (
    val_adaptive_score.copy()
)

if len(candidate_indices) > 0:

    candidate_adaptive_test = (
        test_adaptive_score_all[
            candidate_indices
        ]
    )

    hierarchical_adaptive_test_score[
        candidate_indices
    ] = candidate_adaptive_test


if len(val_candidate_indices) > 0:

    candidate_adaptive_val = (
        val_adaptive_score[
            val_candidate_indices
        ]
    )

    hierarchical_adaptive_val_score[
        val_candidate_indices
    ] = candidate_adaptive_val

adaptive_threshold, _ = tune_threshold(
    y_val_true,
    hierarchical_adaptive_val_score,
    min_precision=0.05
)

m3_pred = (
    hierarchical_adaptive_test_score
    >= adaptive_threshold
).astype(np.int8)

m3 = binary_metrics(
    y_test_true,
    m3_pred
)

m3 = add_ranking_metrics(
    m3,
    y_test_true,
    hierarchical_adaptive_test_score
)

print_metrics(
    "M3 - Proposed Adaptive Fusion",
    m3
)

print(
    "Adaptive threshold:",
    adaptive_threshold
)

print(
    "Mean AE weight:",
    float(
        test_ae_weights.mean()
    )
)

print(
    "Min AE weight:",
    float(
        test_ae_weights.min()
    )
)

print(
    "Max AE weight:",
    float(
        test_ae_weights.max()
    )
)


# ==========================================================
# 14. COMPUTATIONAL EFFICIENCY
# ==========================================================

reduction = (
    1.0
    -
    len(candidate_indices)
    /
    max(TEST_WINDOWS, 1)
) * 100.0

print(
    "\n========== COMPUTATIONAL EFFICIENCY =========="
)

print(
    "Total test windows:",
    TEST_WINDOWS
)

print(
    "Layer-1 candidates:",
    len(candidate_indices)
)

print(
    "Layer-2 workload reduction:",
    f"{reduction:.2f}%"
)


# ==========================================================
# 15. COMPARISON TABLE
# ==========================================================

results = {
    "M0 LSTM-AE": m0,
    "M1 Fixed Fusion": fixed_metrics,
    "M2 Hierarchical": m2,
    "M3 Adaptive Fusion": m3
}

table = evaluation_table(
    results
)

print(
    "\n=========================================="
)
print(
    "ABLATION / MODEL COMPARISON"
)
print(
    "=========================================="
)

print(
    table.to_string(
        index=False
    )
)

table.to_csv(
    "ablation_results_v4.csv",
    index=False
)

pd.DataFrame(
    val_events
).to_csv(
    "synthetic_validation_events_v4.csv",
    index=False
)

pd.DataFrame(
    test_events
).to_csv(
    "synthetic_test_events_v4.csv",
    index=False
)


# ==========================================================
# 16. PLOTS
# ==========================================================

plot_error_distribution(
    val_ae_score,
    hierarchical_adaptive_test_score,
    adaptive_threshold
)

plot_anomalies(
    test_eval_features[
        "aggregate"
    ].to_numpy()[
        SEQUENCE_LENGTH - 1:
    ],
    m3_pred
)


# ==========================================================
# 17. FINAL SUMMARY
# ==========================================================

elapsed = (
    time.perf_counter()
    - total_start
)

print(
    "\n=========================================="
)
print(
    "FINAL SUMMARY"
)
print(
    "=========================================="
)

print(
    "M0 F1:",
    m0["F1"]
)

print(
    "M1 F1:",
    fixed_metrics["F1"]
)

print(
    "M2 F1:",
    m2["F1"]
)

print(
    "M3 F1:",
    m3["F1"]
)

print(
    "\nM3 Precision:",
    m3["Precision"]
)

print(
    "M3 Recall:",
    m3["Recall"]
)

print(
    "M3 FPR:",
    m3["FPR"]
)

print(
    "M3 PR-AUC:",
    m3["PR_AUC"]
)

print(
    "M3 ROC-AUC:",
    m3["ROC_AUC"]
)

print(
    "\nCandidate count:",
    len(candidate_indices)
)

print(
    "Layer-2 workload reduction:",
    f"{reduction:.2f}%"
)

print(
    "Runtime:",
    f"{elapsed:.2f} seconds"
)

print(
    "\nSaved:"
)

print(
    "ablation_results_v4.csv"
)

print(
    "synthetic_validation_events_v4.csv"
)

print(
    "synthetic_test_events_v4.csv"
)

print(
    "best_layer1_v4.keras"
)

print(
    "best_layer2_v4.keras"
)

print(
    "\n=========================================="
)
print(
    "RESEARCH EXPERIMENT V4 FINISHED"
)
print(
    "=========================================="
)
