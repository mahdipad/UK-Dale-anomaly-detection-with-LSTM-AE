# ============================================================
# LSTM AUTOENCODER BASELINE - UK-DALE
# Energy Consumption Anomaly Detection
# ============================================================

import os

# Reduce TensorFlow unnecessary logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# Data
HOUSE = "house_1"
CHANNEL = "channel_1.dat"

RESAMPLE_RULE = "1min"

# Temporal split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Maximum amount of raw minute-level data used for training
# 350,000 minutes ~= 243 days
MAX_TRAIN_MINUTES = 350_000

# Sequence
SEQUENCE_LENGTH = 60

# Model
LATENT_DIM = 16

# Training
EPOCHS = 15
BATCH_SIZE = 512
LEARNING_RATE = 0.001
PATIENCE = 3

# Threshold
THRESHOLD_METHOD = "percentile_99"

# Reconstruction batch size
PREDICT_BATCH_SIZE = 512

# Output directories
MODELS_DIR_NAME = "models"
RESULTS_DIR_NAME = "results"


# ============================================================
# RANDOM SEED
# ============================================================

def set_random_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Deterministic operations where possible
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print(f"[INFO] Random seed = {seed}")


# ============================================================
# PATH MANAGEMENT
# ============================================================

def get_project_paths():

    """
    Find project root based on the location of this Python file.

    Expected structure:

    ukdale/
    ├── data/
    │   └── house_1/
    │       └── channel_1.dat
    ├── src/
    │   └── baseline.py
    ├── models/
    └── results/
    """

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent

    possible_paths = [

        project_root / "data" / HOUSE / CHANNEL,

        project_root / "data" / CHANNEL,

        project_root.parent / "data" / HOUSE / CHANNEL,

        project_root.parent / "data" / CHANNEL,

    ]

    for path in possible_paths:

        if path.exists():
            return (
                project_root,
                path,
                project_root / MODELS_DIR_NAME,
                project_root / RESULTS_DIR_NAME
            )

    searched = "\n".join(str(p) for p in possible_paths)

    raise FileNotFoundError(
        "\nDataset not found.\n"
        "The following locations were checked:\n\n"
        + searched
        + "\n\n"
        "Please verify the location of channel_1.dat."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    project_root, data_path, models_dir, results_dir = get_project_paths()

    print(f"[INFO] Project root : {project_root}")
    print(f"[INFO] Dataset path : {data_path}")

    # UK-DALE channel files:
    # timestamp power
    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=["timestamp", "power"],
        dtype={
            "timestamp": np.int64,
            "power": np.float32
        }
    )

    print(f"[INFO] Raw shape: {df.shape}")

    # Convert Unix timestamp to datetime
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # Remove invalid power values
    df["power"] = pd.to_numeric(
        df["power"],
        errors="coerce"
    )

    df = df.dropna(subset=["power"])

    # Negative power values are invalid for this baseline
    df.loc[df["power"] < 0, "power"] = np.nan

    df = df.dropna(subset=["power"])

    # Sort chronologically
    df = df.sort_values("timestamp")

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="first"
    )

    df = df.set_index("timestamp")

    print(
        f"[INFO] Time range: "
        f"{df.index.min()} -> {df.index.max()}"
    )

    print(f"[INFO] Clean raw shape: {df.shape}")

    return df, project_root, models_dir, results_dir


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(df):

    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)

    # Resample to one-minute mean
    df = df[["power"]].resample(RESAMPLE_RULE).mean()

    print(f"[INFO] After resampling: {df.shape}")

    missing_before = int(df["power"].isna().sum())

    print(
        f"[INFO] Missing values before interpolation: "
        f"{missing_before}"
    )

    # Time interpolation
    df["power"] = df["power"].interpolate(
        method="time",
        limit_direction="both"
    )

    # Safety fill
    df["power"] = df["power"].ffill().bfill()

    missing_after = int(df["power"].isna().sum())

    print(
        f"[INFO] Missing values after handling: "
        f"{missing_after}"
    )

    df["power"] = df["power"].astype(np.float32)

    print(
        f"[INFO] Final preprocessing shape: "
        f"{df.shape}"
    )

    return df


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(df):

    print("\n" + "=" * 70)
    print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    n = len(df)

    train_end = int(n * TRAIN_RATIO)

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    train_df = df.iloc[:train_end].copy()

    val_df = df.iloc[
        train_end:val_end
    ].copy()

    test_df = df.iloc[
        val_end:
    ].copy()

    print(
        f"[INFO] Train shape      : {train_df.shape}"
    )

    print(
        f"[INFO] Validation shape : {val_df.shape}"
    )

    print(
        f"[INFO] Test shape       : {test_df.shape}"
    )

    print("\nTime periods:")

    print(
        f"Train      : "
        f"{train_df.index.min()} -> "
        f"{train_df.index.max()}"
    )

    print(
        f"Validation : "
        f"{val_df.index.min()} -> "
        f"{val_df.index.max()}"
    )

    print(
        f"Test       : "
        f"{test_df.index.min()} -> "
        f"{test_df.index.max()}"
    )

    return train_df, val_df, test_df


# ============================================================
# SELECT TRAINING SUBSET
# ============================================================

def select_training_subset(train_df):

    print("\n" + "=" * 70)
    print("SELECT TRAINING SUBSET")
    print("=" * 70)

    original_size = len(train_df)

    if original_size <= MAX_TRAIN_MINUTES:

        selected = train_df.copy()

    else:

        # CONTIGUOUS temporal subset.
        # No random sampling.
        selected = train_df.iloc[
            :MAX_TRAIN_MINUTES
        ].copy()

    print(
        f"[INFO] Full Train minutes    : "
        f"{original_size:,}"
    )

    print(
        f"[INFO] Selected Train minutes: "
        f"{len(selected):,}"
    )

    print(
        f"[INFO] Selected period:"
        f"\n       {selected.index.min()}"
        f"\n       -> {selected.index.max()}"
    )

    return selected


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(
    train_df,
    val_df,
    test_df
):

    print("\n" + "=" * 70)
    print("NORMALIZATION")
    print("=" * 70)

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # IMPORTANT:
    # Fit ONLY on training data.
    train_values = train_df[["power"]].values.astype(
        np.float32
    )

    scaler.fit(train_values)

    train_scaled = scaler.transform(
        train_values
    ).astype(np.float32)

    val_scaled = scaler.transform(
        val_df[["power"]].values.astype(np.float32)
    ).astype(np.float32)

    test_scaled = scaler.transform(
        test_df[["power"]].values.astype(np.float32)
    ).astype(np.float32)

    print(
        f"[INFO] Train normalized range: "
        f"{train_scaled.min():.6f} -> "
        f"{train_scaled.max():.6f}"
    )

    print(
        f"[INFO] Validation normalized range: "
        f"{val_scaled.min():.6f} -> "
        f"{val_scaled.max():.6f}"
    )

    print(
        f"[INFO] Test normalized range: "
        f"{test_scaled.min():.6f} -> "
        f"{test_scaled.max():.6f}"
    )

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        scaler
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    data,
    sequence_length=60
):

    """
    Create sliding-window sequences.

    Input:
        (N, 1)

    Output:
        (N-sequence_length+1, sequence_length, 1)
    """

    data = np.asarray(
        data,
        dtype=np.float32
    )

    n = len(data)

    if n < sequence_length:
        raise ValueError(
            f"Data length ({n}) is smaller than "
            f"sequence length ({sequence_length})."
        )

    # Efficient sliding window creation
    windows = np.lib.stride_tricks.sliding_window_view(
        data,
        window_shape=sequence_length,
        axis=0
    )

    # Depending on NumPy version the shape can be:
    # (N-seq+1, 1, seq)
    # Convert to desired shape.
    if windows.ndim == 3:

        if windows.shape[1] == 1:

            windows = np.transpose(
                windows,
                (0, 2, 1)
            )

    windows = np.ascontiguousarray(
        windows,
        dtype=np.float32
    )

    return windows


# ============================================================
# BUILD LSTM AUTOENCODER
# ============================================================

def build_lstm_autoencoder():

    print("\n" + "=" * 70)
    print("BUILDING LSTM AUTOENCODER")
    print("=" * 70)

    inputs = Input(
        shape=(SEQUENCE_LENGTH, 1),
        name="input_sequence"
    )

    # Encoder
    x = LSTM(
        64,
        return_sequences=True,
        name="encoder_lstm_64"
    )(inputs)

    x = LSTM(
        32,
        return_sequences=False,
        name="encoder_lstm_32"
    )(x)

    # Latent representation
    latent = Dense(
        LATENT_DIM,
        activation=None,
        name="latent_vector"
    )(x)

    # Decoder
    x = RepeatVector(
        SEQUENCE_LENGTH,
        name="repeat_latent"
    )(latent)

    x = LSTM(
        32,
        return_sequences=True,
        name="decoder_lstm_32"
    )(x)

    x = LSTM(
        64,
        return_sequences=True,
        name="decoder_lstm_64"
    )(x)

    outputs = TimeDistributed(
        Dense(1),
        name="reconstruction"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="LSTM_Autoencoder"
    )

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    model.summary()

    print(
        f"\n[INFO] Total parameters: "
        f"{model.count_params():,}"
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    X_val
):

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(
        f"[INFO] X_train shape: {X_train.shape}"
    )

    print(
        f"[INFO] X_val shape  : {X_val.shape}"
    )

    print(
        f"[INFO] Epochs        : {EPOCHS}"
    )

    print(
        f"[INFO] Batch size    : {BATCH_SIZE}"
    )

    print(
        f"[INFO] Learning rate : {LEARNING_RATE}"
    )

    print(
        f"[INFO] Early stopping patience: "
        f"{PATIENCE}"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        X_train,

        validation_data=(
            X_val,
            X_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        # Preserve temporal order
        shuffle=False,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )

    training_time = time.time() - start_time

    print(
        f"\n[INFO] Training time: "
        f"{training_time / 60:.2f} minutes"
    )

    actual_epochs = len(
        history.history["loss"]
    )

    print(
        f"[INFO] Actual epochs executed: "
        f"{actual_epochs}"
    )

    return (
        history,
        training_time
    )


# ============================================================
# CALCULATE RECONSTRUCTION ERROR IN BATCHES
# ============================================================

def calculate_reconstruction_error(
    model,
    X,
    batch_size=512,
    name="data"
):

    print("\n" + "-" * 70)
    print(
        f"Calculating reconstruction error: {name}"
    )
    print("-" * 70)

    n = len(X)

    errors = np.empty(
        n,
        dtype=np.float32
    )

    start = time.time()

    for start_idx in range(
        0,
        n,
        batch_size
    ):

        end_idx = min(
            start_idx + batch_size,
            n
        )

        batch = X[
            start_idx:end_idx
        ]

        reconstructed = model.predict(
            batch,
            batch_size=batch_size,
            verbose=0
        )

        # MSE per sequence
        batch_errors = np.mean(
            np.square(
                batch - reconstructed
            ),
            axis=(1, 2)
        )

        errors[
            start_idx:end_idx
        ] = batch_errors.astype(
            np.float32
        )

        # Free temporary reconstruction
        del reconstructed

    elapsed = time.time() - start

    print(
        f"[INFO] {name} error calculation time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"[INFO] Error shape: {errors.shape}"
    )

    print(
        f"[INFO] Mean error: "
        f"{np.mean(errors):.8f}"
    )

    print(
        f"[INFO] Std error: "
        f"{np.std(errors):.8f}"
    )

    print(
        f"[INFO] Min error: "
        f"{np.min(errors):.8f}"
    )

    print(
        f"[INFO] Max error: "
        f"{np.max(errors):.8f}"
    )

    return errors


# ============================================================
# CALCULATE THRESHOLD
# ============================================================

def calculate_threshold(
    validation_errors,
    method="percentile_99"
):

    print("\n" + "=" * 70)
    print("THRESHOLD CALCULATION")
    print("=" * 70)

    if method == "percentile_99":

        threshold = np.percentile(
            validation_errors,
            99
        )

        description = (
            "99th Percentile"
        )

    elif method == "percentile_95":

        threshold = np.percentile(
            validation_errors,
            95
        )

        description = (
            "95th Percentile"
        )

    elif method == "mean_plus_3std":

        mean_error = np.mean(
            validation_errors
        )

        std_error = np.std(
            validation_errors
        )

        threshold = (
            mean_error +
            3 * std_error
        )

        description = (
            "Mean + 3 * Standard Deviation"
        )

    else:

        raise ValueError(
            f"Unknown threshold method: {method}"
        )

    print(
        f"[INFO] Method: {description}"
    )

    print(
        f"[INFO] Threshold: "
        f"{threshold:.8f}"
    )

    return float(threshold)


# ============================================================
# DETECT ANOMALIES
# ============================================================

def detect_anomalies(
    errors,
    threshold
):

    predictions = (
        errors > threshold
    ).astype(np.int8)

    n_anomalies = int(
        np.sum(predictions)
    )

    percentage = (
        n_anomalies /
        len(predictions)
    ) * 100

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION")
    print("=" * 70)

    print(
        f"[INFO] Threshold: "
        f"{threshold:.8f}"
    )

    print(
        f"[INFO] Total sequences: "
        f"{len(predictions):,}"
    )

    print(
        f"[INFO] Detected anomalies: "
        f"{n_anomalies:,}"
    )

    print(
        f"[INFO] Anomaly percentage: "
        f"{percentage:.4f}%"
    )

    return predictions


# ============================================================
# OPTIONAL GROUND TRUTH EVALUATION
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    anomaly_scores
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    if y_true is None:

        print(
            "[INFO] Ground Truth labels are not available."
        )

        print(
            "[INFO] Precision, Recall, F1, Accuracy, "
            "ROC-AUC and PR-AUC cannot be calculated."
        )

        return None

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    results = {}

    results["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results["recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    # ROC-AUC requires both classes
    if len(np.unique(y_true)) > 1:

        results["roc_auc"] = roc_auc_score(
            y_true,
            anomaly_scores
        )

        results["pr_auc"] = average_precision_score(
            y_true,
            anomaly_scores
        )

    else:

        results["roc_auc"] = None
        results["pr_auc"] = None

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    results["confusion_matrix"] = cm.tolist()

    print(
        f"Precision : {results['precision']:.4f}"
    )

    print(
        f"Recall    : {results['recall']:.4f}"
    )

    print(
        f"F1-score  : {results['f1']:.4f}"
    )

    print(
        f"Accuracy  : {results['accuracy']:.4f}"
    )

    print(
        f"ROC-AUC   : {results['roc_auc']}"
    )

    print(
        f"PR-AUC    : {results['pr_auc']}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return results


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    scaler,
    threshold,
    models_dir
):

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        models_dir /
        "lstm_autoencoder_baseline.keras"
    )

    scaler_path = (
        models_dir /
        "minmax_scaler.npy"
    )

    threshold_path = (
        models_dir /
        "threshold.json"
    )

    model.save(
        model_path
    )

    # Save scaler parameters
    scaler_data = {
        "min_": scaler.min_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist()
    }

    np.save(
        scaler_path,
        scaler_data,
        allow_pickle=True
    )

    with open(
        threshold_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "method": THRESHOLD_METHOD,
                "threshold": threshold
            },
            f,
            indent=4
        )

    print("\n[INFO] Model saved:")
    print(model_path)

    print("[INFO] Scaler saved:")
    print(scaler_path)

    print("[INFO] Threshold saved:")
    print(threshold_path)


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

def save_results(
    history,
    train_errors,
    val_errors,
    test_errors,
    test_predictions,
    threshold,
    results_dir
):

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(
        results_dir /
        "training_history.csv",
        index=False
    )

    # --------------------------------------------------------
    # Reconstruction errors
    # --------------------------------------------------------

    np.save(
        results_dir /
        "train_reconstruction_errors.npy",
        train_errors
    )

    np.save(
        results_dir /
        "validation_reconstruction_errors.npy",
        val_errors
    )

    np.save(
        results_dir /
        "test_reconstruction_errors.npy",
        test_errors
    )

    np.save(
        results_dir /
        "test_anomaly_predictions.npy",
        test_predictions
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "threshold_method":
            THRESHOLD_METHOD,

        "threshold":
            float(threshold),

        "train_error_mean":
            float(np.mean(train_errors)),

        "train_error_std":
            float(np.std(train_errors)),

        "validation_error_mean":
            float(np.mean(val_errors)),

        "validation_error_std":
            float(np.std(val_errors)),

        "test_error_mean":
            float(np.mean(test_errors)),

        "test_error_std":
            float(np.std(test_errors)),

        "test_error_min":
            float(np.min(test_errors)),

        "test_error_max":
            float(np.max(test_errors)),

        "number_of_detected_anomalies":
            int(np.sum(test_predictions)),

        "anomaly_percentage":
            float(
                np.mean(test_predictions) * 100
            )
    }

    with open(
        results_dir /
        "baseline_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"\n[INFO] Results saved in:"
        f"\n{results_dir}"
    )


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

def plot_training_history(
    history,
    results_dir
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title(
        "LSTM Autoencoder Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir /
        "01_training_validation_loss.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT ERROR DISTRIBUTION
# ============================================================

def plot_error_distribution(
    validation_errors,
    test_errors,
    threshold,
    results_dir
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        validation_errors,
        bins=100,
        alpha=0.6,
        label="Validation Error"
    )

    plt.hist(
        test_errors,
        bins=100,
        alpha=0.5,
        label="Test Error"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.6f}"
    )

    plt.xlabel(
        "Reconstruction Error"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Reconstruction Error Distribution"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir /
        "02_reconstruction_error_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT ORIGINAL VS RECONSTRUCTED
# ============================================================

def plot_reconstruction(
    model,
    X_test,
    results_dir,
    index=None
):

    if index is None:

        # Pick a middle sample
        index = len(X_test) // 2

    sample = X_test[
        index:index + 1
    ]

    reconstructed = model.predict(
        sample,
        verbose=0
    )

    original = sample[0, :, 0]

    reconstructed = reconstructed[
        0, :, 0
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        original,
        label="Original"
    )

    plt.plot(
        reconstructed,
        label="Reconstructed"
    )

    plt.xlabel(
        "Minute"
    )

    plt.ylabel(
        "Normalized Power"
    )

    plt.title(
        "Original vs Reconstructed Sequence"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir /
        "03_original_vs_reconstructed.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT ANOMALY SCORE
# ============================================================

def plot_anomaly_score(
    errors,
    predictions,
    threshold,
    results_dir
):

    plt.figure(
        figsize=(14, 6)
    )

    x = np.arange(
        len(errors)
    )

    plt.plot(
        x,
        errors,
        linewidth=0.8,
        label="Reconstruction Error"
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label="Threshold"
    )

    anomaly_indices = np.where(
        predictions == 1
    )[0]

    if len(anomaly_indices) > 0:

        plt.scatter(
            anomaly_indices,
            errors[anomaly_indices],
            s=8,
            label="Anomaly"
        )

    plt.xlabel(
        "Sequence Index"
    )

    plt.ylabel(
        "Reconstruction Error"
    )

    plt.title(
        "Anomaly Score Over Test Sequences"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir /
        "04_anomaly_score_test.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT TEST ANOMALIES
# ============================================================

def plot_anomalies_on_signal(
    test_df,
    test_errors,
    predictions,
    threshold,
    results_dir
):

    # Each sequence starts at index i.
    # Map sequence anomaly scores to sequence start timestamps.

    if len(test_df) < SEQUENCE_LENGTH:
        return

    timestamps = test_df.index[
        SEQUENCE_LENGTH - 1:
    ]

    n = min(
        len(timestamps),
        len(test_errors)
    )

    timestamps = timestamps[:n]

    errors = test_errors[:n]
    preds = predictions[:n]

    power = test_df["power"].values

    power = power[
        SEQUENCE_LENGTH - 1:
    ]

    power = power[:n]

    anomaly_mask = preds == 1

    plt.figure(
        figsize=(15, 6)
    )

    plt.plot(
        timestamps,
        power,
        linewidth=0.7,
        label="Aggregate Power"
    )

    if np.any(anomaly_mask):

        plt.scatter(
            timestamps[anomaly_mask],
            power[anomaly_mask],
            s=10,
            label="Detected Anomaly"
        )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Power"
    )

    plt.title(
        "Detected Anomalies on Aggregate Power"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir /
        "05_anomalies_on_signal.png",
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("\n")
    print("=" * 70)
    print("LSTM AUTOENCODER - UK-DALE")
    print("Baseline Anomaly Detection")
    print("=" * 70)

    set_random_seed(SEED)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df, project_root, models_dir, results_dir = load_data()

    # --------------------------------------------------------
    # 2. Preprocessing
    # --------------------------------------------------------

    df = preprocess_data(df)

    print("\n[INFO] Dataset shape after preprocessing:")
    print(df.shape)

    # --------------------------------------------------------
    # 3. Temporal Split
    # --------------------------------------------------------

    train_df, val_df, test_df = temporal_split(df)

    # --------------------------------------------------------
    # 4. Select training subset
    # --------------------------------------------------------

    train_selected_df = select_training_subset(
        train_df
    )

    # --------------------------------------------------------
    # 5. Normalize
    # --------------------------------------------------------

    (
        train_scaled,
        val_scaled,
        test_scaled,
        scaler
    ) = normalize_data(
        train_selected_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # 6. Create sequences
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING SEQUENCES")
    print("=" * 70)

    start_sequence_time = time.time()

    X_train = create_sequences(
        train_scaled,
        SEQUENCE_LENGTH
    )

    X_val = create_sequences(
        val_scaled,
        SEQUENCE_LENGTH
    )

    X_test = create_sequences(
        test_scaled,
        SEQUENCE_LENGTH
    )

    sequence_time = (
        time.time() -
        start_sequence_time
    )

    print(
        f"[INFO] Sequence creation time: "
        f"{sequence_time:.2f} seconds"
    )

    print(
        f"[INFO] X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"[INFO] X_val shape: "
        f"{X_val.shape}"
    )

    print(
        f"[INFO] X_test shape: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # 7. Build model
    # --------------------------------------------------------

    model = build_lstm_autoencoder()

    # --------------------------------------------------------
    # 8. Train
    # --------------------------------------------------------

    (
        history,
        training_time
    ) = train_model(
        model,
        X_train,
        X_val
    )

    # --------------------------------------------------------
    # 9. Reconstruction errors
    # --------------------------------------------------------

    train_errors = calculate_reconstruction_error(
        model,
        X_train,
        PREDICT_BATCH_SIZE,
        "TRAIN"
    )

    val_errors = calculate_reconstruction_error(
        model,
        X_val,
        PREDICT_BATCH_SIZE,
        "VALIDATION"
    )

    test_errors = calculate_reconstruction_error(
        model,
        X_test,
        PREDICT_BATCH_SIZE,
        "TEST"
    )

    # --------------------------------------------------------
    # 10. Threshold
    # --------------------------------------------------------

    threshold = calculate_threshold(
        val_errors,
        THRESHOLD_METHOD
    )

    # --------------------------------------------------------
    # 11. Detect anomalies
    # --------------------------------------------------------

    test_predictions = detect_anomalies(
        test_errors,
        threshold
    )

    # --------------------------------------------------------
    # 12. Ground truth
    # --------------------------------------------------------

    # UK-DALE aggregate data does not automatically provide
    # anomaly labels for this experiment.
    y_true = None

    evaluation_results = evaluate_model(
        y_true,
        test_predictions,
        test_errors
    )

    # --------------------------------------------------------
    # 13. Save model
    # --------------------------------------------------------

    save_model(
        model,
        scaler,
        threshold,
        models_dir
    )

    # --------------------------------------------------------
    # 14. Save results
    # --------------------------------------------------------

    save_results(
        history,
        train_errors,
        val_errors,
        test_errors,
        test_predictions,
        threshold,
        results_dir
    )

    # --------------------------------------------------------
    # 15. Plots
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_training_history(
        history,
        results_dir
    )

    plot_error_distribution(
        val_errors,
        test_errors,
        threshold,
        results_dir
    )

    plot_reconstruction(
        model,
        X_test,
        results_dir
    )

    plot_anomaly_score(
        test_errors,
        test_predictions,
        threshold,
        results_dir
    )

    plot_anomalies_on_signal(
        test_df,
        test_errors,
        test_predictions,
        threshold,
        results_dir
    )

    # --------------------------------------------------------
    # 16. Final report
    # --------------------------------------------------------

    total_time = (
        time.time() -
        total_start
    )

    final_train_loss = history.history[
        "loss"
    ][-1]

    best_val_loss = min(
        history.history[
            "val_loss"
        ]
    )

    final_val_loss = history.history[
        "val_loss"
    ][-1]

    n_anomalies = int(
        np.sum(test_predictions)
    )

    anomaly_percentage = (
        n_anomalies /
        len(test_predictions)
    ) * 100

    print("\n")
    print("=" * 70)
    print("FINAL BASELINE RESULTS")
    print("=" * 70)

    print(
        f"Dataset shape after preprocessing : "
        f"{df.shape}"
    )

    print(
        f"Train shape                       : "
        f"{train_df.shape}"
    )

    print(
        f"Selected training shape           : "
        f"{train_selected_df.shape}"
    )

    print(
        f"Validation shape                  : "
        f"{val_df.shape}"
    )

    print(
        f"Test shape                        : "
        f"{test_df.shape}"
    )

    print(
        f"X_train shape                     : "
        f"{X_train.shape}"
    )

    print(
        f"X_validation shape                : "
        f"{X_val.shape}"
    )

    print(
        f"X_test shape                      : "
        f"{X_test.shape}"
    )

    print(
        f"Sequence length                   : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Model parameters                  : "
        f"{model.count_params():,}"
    )

    print(
        f"Epochs executed                   : "
        f"{len(history.history['loss'])}"
    )

    print(
        f"Training time                     : "
        f"{training_time / 60:.2f} minutes"
    )

    print(
        f"Total execution time              : "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Final training loss               : "
        f"{final_train_loss:.8f}"
    )

    print(
        f"Best validation loss              : "
        f"{best_val_loss:.8f}"
    )

    print(
        f"Final validation loss             : "
        f"{final_val_loss:.8f}"
    )

    print(
        f"Mean Train reconstruction error  : "
        f"{np.mean(train_errors):.8f}"
    )

    print(
        f"Mean Validation reconstruction   : "
        f"{np.mean(val_errors):.8f}"
    )

    print(
        f"Mean Test reconstruction error   : "
        f"{np.mean(test_errors):.8f}"
    )

    print(
        f"Std Validation reconstruction    : "
        f"{np.std(val_errors):.8f}"
    )

    print(
        f"Threshold                         : "
        f"{threshold:.8f}"
    )

    print(
        f"Threshold method                  : "
        f"{THRESHOLD_METHOD}"
    )

    print(
        f"Detected anomalies                : "
        f"{n_anomalies:,}"
    )

    print(
        f"Anomaly percentage                : "
        f"{anomaly_percentage:.4f}%"
    )

    print("\nGround Truth:")
    print(
        "NOT AVAILABLE for this baseline."
    )

    print(
        "\nPrecision / Recall / F1 / Accuracy / "
        "ROC-AUC / PR-AUC:"
    )

    print(
        "NOT CALCULATED because Ground Truth "
        "labels are unavailable."
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPLETED")
    print("=" * 70)

    print(
        f"\nResults directory:\n{results_dir}"
    )

    print(
        f"\nModel directory:\n{models_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()