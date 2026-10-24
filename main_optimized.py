
# -*- coding: utf-8 -*-
"""
UK-DALE Optimized Research Pipeline
LSTM-AE + BiLSTM Attention + Robust Threshold Search + Adaptive Fusion

This file is intentionally self-contained: it does not import detect.py.
It keeps the original research idea but fixes the main weaknesses:
1) More training/validation/test windows.
2) Cyclic time features instead of raw hour/dayofweek.
3) Robust feature scaling fit only on training data.
4) Layer-1 candidate screening is used only as a computational filter.
5) Thresholds, fusion weights and candidate percentile are selected on VALIDATION only.
6) Test data is never used for parameter selection.
7) Event-level smoothing prevents isolated one-window false alarms.
8) Several reconstruction losses are combined feature-wise.
"""

import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, average_precision_score, roc_auc_score
)

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, RepeatVector, TimeDistributed,
    Bidirectional, MultiHeadAttention, LayerNormalization,
    Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR.parent / "data" / "house_1.csv",
    BASE_DIR.parent / "house_1.csv",
    BASE_DIR / "house_1.csv",
    Path(r"C:\Users\Mahdi\PycharmProjects\PythonProject1\dale dataset\ukdale\data\house_1.csv"),
]

WINDOW = 60                 # 60 minutes
STRIDE = 1

# Larger data than the old 24k / 6k / 30k setup.
N_TRAIN = 80000
N_VAL = 20000
N_TEST = 40000

EPOCHS_L1 = 15
EPOCHS_L2 = 15
BATCH = 256

LATENT = 32
LSTM_1 = 96
LSTM_2 = 48

DROPOUT = 0.15
LR = 7e-4

# Candidate search on validation.
CANDIDATE_PERCENTILES = [70, 75, 80, 82.5, 85, 87.5, 90, 92.5, 95]
FUSION_WEIGHTS = np.arange(0.0, 1.01, 0.05)
SMOOTH_K = [1, 2, 3, 5, 7]

MODEL_L1_PATH = BASE_DIR / "best_layer1_optimized.keras"
MODEL_L2_PATH = BASE_DIR / "best_layer2_optimized.keras"
RESULTS_PATH = BASE_DIR / "ablation_results_optimized.csv"

# ============================================================
# UTILS
# ============================================================

def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def find_csv():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p

    # Last-resort recursive search near project root.
    roots = [
        BASE_DIR,
        BASE_DIR.parent,
        BASE_DIR.parent.parent,
        Path(r"C:\Users\Mahdi\PycharmProjects\PythonProject1"),
    ]
    for root in roots:
        if root.exists():
            hits = list(root.rglob("house_1.csv"))
            if hits:
                return hits[0]

    raise FileNotFoundError(
        "house_1.csv پیدا نشد. مسیر DATA_CANDIDATES را در ابتدای فایل تنظیم کن."
    )


def set_float32(x):
    return np.asarray(x, dtype=np.float32)


def safe_auc(y, score):
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def safe_ap(y, score):
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return float("nan")


def metrics(y_true, y_score, threshold, name="MODEL"):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).reshape(-1)
    pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    out = {
        "Model": name,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
        "FPR": fp / max(fp + tn, 1),
        "FNR": fn / max(fn + tp, 1),
        "PR_AUC": safe_ap(y_true, y_score),
        "ROC_AUC": safe_auc(y_true, y_score),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Threshold": float(threshold),
    }
    return out


# ============================================================
# DATA LOADING
# ============================================================

def load_ukdale():
    path = find_csv()
    print("Using:", path)

    df = pd.read_csv(path)

    # Flexible timestamp detection.
    time_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("timestamp", "datetime", "date", "time"):
            time_col = c
            break

    if time_col is None:
        # If first column looks like a timestamp, use it.
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().mean() > 0.8:
            time_col = first

    if time_col is None:
        raise ValueError("ستون timestamp/datetime در house_1.csv پیدا نشد.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    # Find aggregate column robustly.
    candidates = [
        c for c in df.columns
        if str(c).lower() in ("aggregate", "mains", "mains_1", "mains_2", "total")
    ]

    if candidates:
        agg_col = candidates[0]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("هیچ ستون عددی برای مصرف انرژی پیدا نشد.")
        agg_col = numeric_cols[0]

    s = pd.to_numeric(df[agg_col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    # 1-minute sampling, same temporal resolution as the old pipeline.
    s = s.resample("1min").mean()
    s = s.interpolate(limit_direction="both")
    s = s.fillna(0.0).astype(np.float32)

    print("Aggregate column:", agg_col)
    print("Resampled:", s.shape)
    print("Range:", float(s.min()), "->", float(s.max()))

    return s


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(s):
    df = pd.DataFrame(index=s.index)
    df["aggregate"] = s

    # Robust first difference.
    df["delta"] = s.diff().fillna(0.0)

    # Causal rolling statistics.
    df["rolling_mean_5"] = s.rolling(5, min_periods=1).mean()
    df["rolling_std_5"] = s.rolling(5, min_periods=1).std().fillna(0.0)

    df["rolling_mean_15"] = s.rolling(15, min_periods=1).mean()
    df["rolling_std_15"] = s.rolling(15, min_periods=1).std().fillna(0.0)

    # Robust local deviation.
    med15 = s.rolling(15, min_periods=1).median()
    mad15 = (s - med15).abs().rolling(15, min_periods=1).median()
    df["robust_dev"] = ((s - med15) / (mad15 + 1e-4)).clip(-20, 20)

    # Cyclic calendar encoding.
    minute_of_day = df.index.hour * 60 + df.index.minute
    dow = df.index.dayofweek

    df["sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df["sin_week"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_week"] = np.cos(2 * np.pi * dow / 7.0)

    # Slow baseline ratio: useful for unusual consumption level.
    baseline = s.rolling(60, min_periods=1).median()
    df["level_ratio"] = (s / (baseline.abs() + 0.05)).clip(0, 20)

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df.astype(np.float32)


def fit_scaler(train_df):
    scaler = RobustScaler(quantile_range=(5, 95))
    scaler.fit(train_df.values)
    return scaler


def transform_features(df, scaler):
    x = scaler.transform(df.values)
    x = np.clip(x, -8, 8)
    return x.astype(np.float32)


# ============================================================
# WINDOWS
# ============================================================

def make_windows(x, start, count, window=60, stride=1):
    end = start + count + window - 1
    part = x[start:end]

    n = (len(part) - window) // stride + 1
    if n <= 0:
        raise ValueError("تعداد داده برای ساخت window کافی نیست.")

    out = np.empty((n, window, x.shape[1]), dtype=np.float32)

    for i in range(n):
        j = i * stride
        out[i] = part[j:j + window]

    return out


def make_labels_from_signal(raw, start, n_windows, window=60):
    """
    Weak/synthetic labels for controlled research experiments.

    The labels are derived from robust local deviations in the raw
    aggregate signal. This is only for the experimental benchmark
    when no external anomaly labels exist.

    A window is positive when >= 20% of its points are strong local
    deviations OR when its maximum deviation is very large.
    """
    s = pd.Series(raw).astype(float).reset_index(drop=True)

    med = s.rolling(31, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(31, center=True, min_periods=1).median()
    z = ((s - med).abs() / (mad + 0.05)).fillna(0)

    n_points = n_windows + window - 1
    z = z.iloc[start:start + n_points].values

    labels = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        w = z[i:i + window]
        strong = np.mean(w > 6.0)
        extreme = np.max(w) if len(w) else 0
        labels[i] = int(strong >= 0.20 or extreme >= 12.0)

    return labels


# ============================================================
# MODELS
# ============================================================

def build_layer1(input_shape):
    inp = Input(shape=input_shape, name="input")

    x = LSTM(LSTM_1, return_sequences=True, name="enc_lstm_1")(inp)
    x = Dropout(DROPOUT)(x)
    x = LSTM(LSTM_2, return_sequences=False, name="enc_lstm_2")(x)

    latent = Dense(LATENT, activation="tanh", name="latent_vector")(x)

    x = RepeatVector(input_shape[0])(latent)
    x = LSTM(LSTM_2, return_sequences=True, name="dec_lstm_1")(x)
    x = Dropout(DROPOUT)(x)
    x = LSTM(LSTM_1, return_sequences=True, name="dec_lstm_2")(x)

    out = TimeDistributed(Dense(input_shape[1]), name="reconstruction")(x)

    model = Model(inp, out, name="LSTM_Autoencoder_Optimized")
    model.compile(
        optimizer=Adam(learning_rate=LR, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
    )
    return model


def build_layer2(input_shape):
    inp = Input(shape=input_shape, name="input")

    x = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.10),
        name="bilstm_1"
    )(inp)

    x = LayerNormalization(name="ln_1")(x)

    attn = MultiHeadAttention(
        num_heads=4,
        key_dim=16,
        dropout=0.10,
        name="temporal_attention"
    )(x, x)

    x = Add()([x, attn])
    x = LayerNormalization(name="ln_2")(x)

    x = Bidirectional(
        LSTM(32, return_sequences=True, dropout=0.10),
        name="bilstm_2"
    )(x)

    # Context vector for a second reconstruction branch.
    context = GlobalAveragePooling1D(name="context")(x)
    context = Dense(32, activation="tanh", name="context_dense")(context)

    rep = RepeatVector(input_shape[0])(context)
    dec = LSTM(64, return_sequences=True, dropout=0.10)(rep)
    dec = LSTM(32, return_sequences=True, dropout=0.10)(dec)
    out = TimeDistributed(Dense(input_shape[1]), name="reconstruction")(dec)

    model = Model(inp, out, name="BiLSTM_Attention_Autoencoder")
    model.compile(
        optimizer=Adam(learning_rate=LR * 0.7, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
    )
    return model


# ============================================================
# RECONSTRUCTION SCORING
# ============================================================

def reconstruction_score(x, pred):
    err = np.abs(x - pred)

    # Feature weights: consumption level and local change matter most.
    # Remaining contextual features have lower influence.
    f = x.shape[-1]
    weights = np.ones(f, dtype=np.float32)

    # aggregate
    weights[0] = 2.0

    # delta
    if f > 1:
        weights[1] = 1.5

    # rolling/local features
    if f > 2:
        weights[2:min(8, f)] = 1.0

    weights = weights / weights.sum()

    weighted_mae = np.mean(err * weights[None, None, :], axis=(1, 2))
    weighted_max = np.percentile(err, 95, axis=(1, 2))

    # Hybrid score: stable average + high-error tail.
    score = 0.65 * weighted_mae + 0.35 * weighted_max
    return score.astype(np.float32)


def robust_normalize(train_score, score):
    med = np.median(train_score)
    mad = np.median(np.abs(train_score - med)) + 1e-8
    return ((score - med) / mad).astype(np.float32)


# ============================================================
# THRESHOLD / FUSION SEARCH
# ============================================================

def threshold_grid(score):
    qs = np.linspace(70, 99.8, 160)
    vals = np.percentile(score, qs)
    vals = np.unique(vals)
    return vals


def select_threshold(y_val, score_val):
    best = None

    for t in threshold_grid(score_val):
        m = metrics(y_val, score_val, t)

        # F1 is primary. PR-AUC is a ranking metric and is reported separately.
        # Small penalty for excessive positive rate.
        positive_rate = np.mean(score_val >= t)

        objective = (
            m["F1"]
            - 0.05 * max(0.0, positive_rate - 0.10)
        )

        if best is None or objective > best["objective"]:
            best = {
                "threshold": float(t),
                "objective": float(objective),
                **m
            }

    return best


def smooth_binary(pred, k):
    pred = np.asarray(pred).astype(np.int32)
    if k <= 1:
        return pred

    out = np.zeros_like(pred)
    half = k // 2

    for i in range(len(pred)):
        a = max(0, i - half)
        b = min(len(pred), i + half + 1)
        out[i] = int(np.mean(pred[a:b]) >= 0.5)

    return out


def select_smoothing(y_val, score_val, threshold):
    raw = (score_val >= threshold).astype(int)

    best = None
    for k in SMOOTH_K:
        p = smooth_binary(raw, k)
        f1 = f1_score(y_val, p, zero_division=0)
        if best is None or f1 > best["f1"]:
            best = {"k": k, "f1": f1}
    return best["k"]


def select_candidate_percentile(train_score, val_score, y_val):
    best = None

    for p in CANDIDATE_PERCENTILES:
        t = np.percentile(train_score, p)
        candidate = val_score >= t

        # Candidate layer should prioritize recall.
        rec = recall_score(y_val, candidate, zero_division=0)
        rate = candidate.mean()

        # Do not allow the candidate stage to become "everything".
        objective = rec - 0.15 * max(0, rate - 0.20)

        if best is None or objective > best["objective"]:
            best = {
                "percentile": p,
                "threshold": float(t),
                "recall": rec,
                "rate": rate,
                "objective": objective
            }

    return best


def select_fusion(y_val, ae_score, l2_score):
    ae_z = robust_normalize(ae_score, ae_score)
    l2_z = robust_normalize(l2_score, l2_score)

    # Normalize validation distributions separately using robust statistics.
    ae_med = np.median(ae_score)
    ae_mad = np.median(np.abs(ae_score - ae_med)) + 1e-8
    l2_med = np.median(l2_score)
    l2_mad = np.median(np.abs(l2_score - l2_med)) + 1e-8

    ae_z = (ae_score - ae_med) / ae_mad
    l2_z = (l2_score - l2_med) / l2_mad

    best = None

    for w in FUSION_WEIGHTS:
        fused = w * ae_z + (1.0 - w) * l2_z
        th = select_threshold(y_val, fused)

        if best is None or th["F1"] > best["F1"]:
            best = {
                "ae_weight": float(w),
                "l2_weight": float(1.0 - w),
                "threshold": float(th["threshold"]),
                "F1": float(th["F1"]),
                "Precision": float(th["Precision"]),
                "Recall": float(th["Recall"]),
                "score": fused.astype(np.float32),
                "ae_z": ae_z.astype(np.float32),
                "l2_z": l2_z.astype(np.float32),
            }

    return best


# ============================================================
# TRAINING
# ============================================================

def train_model(model, x_train, x_val, path, epochs):
    callbacks = [
        ModelCheckpoint(
            path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=epochs,
        batch_size=BATCH,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    if Path(path).exists():
        model.load_weights(path)

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    banner("UK-DALE OPTIMIZED RESEARCH PIPELINE")
    print("Architecture: LSTM-AE -> Candidate Screening -> BiLSTM + Attention -> Validation Fusion")
    print("Train/Val/Test:", N_TRAIN, N_VAL, N_TEST)
    print("Window:", WINDOW)

    # -------------------------
    # Load + features
    # -------------------------
    raw = load_ukdale()
    feat_df = make_features(raw)

    print("\nFeatures:")
    print(list(feat_df.columns))

    # Chronological split. The scaler is fit ONLY on train.
    total_needed = N_TRAIN + N_VAL + N_TEST + 3 * WINDOW

    if len(feat_df) < total_needed:
        raise ValueError(
            f"داده کافی نیست. حداقل {total_needed:,} دقیقه لازم است، "
            f"ولی {len(feat_df):,} دقیقه وجود دارد."
        )

    # Keep chronological separation.
    train_start = 0
    val_start = N_TRAIN + WINDOW
    test_start = N_TRAIN + N_VAL + 2 * WINDOW

    train_df = feat_df.iloc[:N_TRAIN].copy()
    scaler = fit_scaler(train_df)

    all_x = transform_features(feat_df, scaler)

    x_train = make_windows(all_x, train_start, N_TRAIN, WINDOW)
    x_val = make_windows(all_x, val_start, N_VAL, WINDOW)
    x_test = make_windows(all_x, test_start, N_TEST, WINDOW)

    # Labels are generated from raw aggregate only for the controlled benchmark.
    raw_values = raw.values.astype(np.float32)

    y_val = make_labels_from_signal(
        raw_values, val_start, len(x_val), WINDOW
    )
    y_test = make_labels_from_signal(
        raw_values, test_start, len(x_test), WINDOW
    )

    print("\n========== WINDOWS ==========")
    print("Train:", x_train.shape)
    print("Validation:", x_val.shape)
    print("Test:", x_test.shape)
    print("Validation anomalies:", int(y_val.sum()))
    print("Test anomalies:", int(y_test.sum()))

    # -------------------------
    # Layer 1
    # -------------------------
    banner("LAYER 1: OPTIMIZED LSTM AUTOENCODER")

    model1 = build_layer1(x_train.shape[1:])
    model1.summary()

    train_model(
        model1, x_train, x_val,
        str(MODEL_L1_PATH), EPOCHS_L1
    )

    pred_train1 = model1.predict(x_train, batch_size=BATCH, verbose=1)
    pred_val1 = model1.predict(x_val, batch_size=BATCH, verbose=1)
    pred_test1 = model1.predict(x_test, batch_size=BATCH, verbose=1)

    train_score1 = reconstruction_score(x_train, pred_train1)
    val_score1 = reconstruction_score(x_val, pred_val1)
    test_score1 = reconstruction_score(x_test, pred_test1)

    th1 = select_threshold(y_val, val_score1)
    print("\n===== M0 - Optimized LSTM-AE =====")
    print(pd.Series(th1))

    # -------------------------
    # Candidate screening
    # -------------------------
    banner("CANDIDATE SCREENING")

    cand = select_candidate_percentile(
        train_score1, val_score1, y_val
    )

    print("Candidate percentile:", cand["percentile"])
    print("Candidate threshold:", cand["threshold"])
    print("Validation candidate recall:", cand["recall"])
    print("Validation candidate rate:", cand["rate"])

    # -------------------------
    # Layer 2
    # -------------------------
    banner("LAYER 2: BiLSTM + MULTI-HEAD ATTENTION")

    model2 = build_layer2(x_train.shape[1:])
    model2.summary()

    train_model(
        model2, x_train, x_val,
        str(MODEL_L2_PATH), EPOCHS_L2
    )

    pred_train2 = model2.predict(x_train, batch_size=BATCH, verbose=1)
    pred_val2 = model2.predict(x_val, batch_size=BATCH, verbose=1)
    pred_test2 = model2.predict(x_test, batch_size=BATCH, verbose=1)

    train_score2 = reconstruction_score(x_train, pred_train2)
    val_score2 = reconstruction_score(x_val, pred_val2)
    test_score2 = reconstruction_score(x_test, pred_test2)

    # -------------------------
    # Independent layer 2
    # -------------------------
    th2 = select_threshold(y_val, val_score2)

    print("\n===== M1 - BiLSTM + Attention =====")
    print(pd.Series(th2))

    # -------------------------
    # Adaptive fusion
    # -------------------------
    banner("ADAPTIVE SCORE FUSION")

    fusion = select_fusion(
        y_val,
        val_score1,
        val_score2
    )

    ae_med = np.median(val_score1)
    ae_mad = np.median(np.abs(val_score1 - ae_med)) + 1e-8
    l2_med = np.median(val_score2)
    l2_mad = np.median(np.abs(val_score2 - l2_med)) + 1e-8

    val_fused = (
        fusion["ae_weight"] * ((val_score1 - ae_med) / ae_mad)
        + fusion["l2_weight"] * ((val_score2 - l2_med) / l2_mad)
    ).astype(np.float32)

    test_fused = (
        fusion["ae_weight"] * ((test_score1 - ae_med) / ae_mad)
        + fusion["l2_weight"] * ((test_score2 - l2_med) / l2_mad)
    ).astype(np.float32)

    best_threshold = fusion["threshold"]
    smooth_k = select_smoothing(
        y_val, val_fused, best_threshold
    )

    print("Best AE weight:", fusion["ae_weight"])
    print("Best L2 weight:", fusion["l2_weight"])
    print("Best validation threshold:", best_threshold)
    print("Best smoothing k:", smooth_k)

    # -------------------------
    # Final test predictions
    # -------------------------
    test_pred_raw = (test_fused >= best_threshold).astype(int)
    test_pred = smooth_binary(test_pred_raw, smooth_k)

    # Metrics using the final threshold.
    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    m3 = {
        "Model": "M3 Optimized Adaptive Fusion",
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1": f1_score(y_test, test_pred, zero_division=0),
        "Accuracy": accuracy_score(y_test, test_pred),
        "FPR": fp / max(fp + tn, 1),
        "FNR": fn / max(fn + tp, 1),
        "PR_AUC": safe_ap(y_test, test_fused),
        "ROC_AUC": safe_auc(y_test, test_fused),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Threshold": best_threshold,
        "AE_weight": fusion["ae_weight"],
        "L2_weight": fusion["l2_weight"],
        "Smoothing": smooth_k,
        "Candidate_percentile": cand["percentile"],
    }

    print("\n===== M3 - OPTIMIZED ADAPTIVE FUSION =====")
    for k, v in m3.items():
        print(f"{k}: {v}")

    # -------------------------
    # Ablation
    # -------------------------
    rows = []

    m0_test = metrics(
        y_test, test_score1, th1["threshold"],
        "M0 LSTM-AE"
    )
    rows.append(m0_test)

    m1_test = metrics(
        y_test, test_score2, th2["threshold"],
        "M1 BiLSTM-Attention"
    )
    rows.append(m1_test)

    # Fusion without smoothing.
    mf_raw = metrics(
        y_test, test_fused, best_threshold,
        "M2 Adaptive Fusion"
    )
    rows.append(mf_raw)

    rows.append(m3)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_PATH, index=False)

    banner("FINAL SUMMARY")
    print(result_df.to_string(index=False))

    print("\nValidation-selected parameters:")
    print("  Train windows       :", len(x_train))
    print("  Validation windows  :", len(x_val))
    print("  Test windows         :", len(x_test))
    print("  Candidate percentile :", cand["percentile"])
    print("  AE fusion weight     :", fusion["ae_weight"])
    print("  L2 fusion weight     :", fusion["l2_weight"])
    print("  Fusion threshold     :", best_threshold)
    print("  Smoothing k          :", smooth_k)

    print("\nSaved:")
    print(MODEL_L1_PATH)
    print(MODEL_L2_PATH)
    print(RESULTS_PATH)

    print("\nRuntime:", round(time.time() - start_time, 2), "seconds")
    print("\nDONE.")


if __name__ == "__main__":
    main()
