import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Reconstruction
# ==========================================================

def reconstruction_error_by_feature(
    model,
    X,
    batch_size=512
):
    reconstructed = model.predict(
        X,
        batch_size=batch_size,
        verbose=1
    )

    errors = np.mean(
        np.square(X - reconstructed),
        axis=1
    )

    return errors.astype(np.float32)


def reconstruction_error(
    model,
    X,
    batch_size=512
):
    errors = reconstruction_error_by_feature(
        model,
        X,
        batch_size=batch_size
    )

    return np.mean(
        errors,
        axis=1
    ).astype(np.float32)


# ==========================================================
# Thresholds
# ==========================================================

def compute_threshold(
    errors,
    method="percentile",
    percentile=99.9,
    n_std=3.0
):
    errors = np.asarray(
        errors,
        dtype=np.float64
    )
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        raise ValueError(
            "No finite scores were supplied."
        )

    if method == "percentile":
        return float(
            np.percentile(
                errors,
                percentile
            )
        )

    if method == "std":
        return float(
            errors.mean() +
            n_std * errors.std()
        )

    raise ValueError(
        "Unknown threshold method."
    )


def detect_anomalies(
    errors,
    threshold
):
    return (
        np.asarray(errors) >= threshold
    ).astype(np.int8)


# ==========================================================
# Score calibration and fusion
# ==========================================================

def normalize_score(
    scores,
    reference_scores,
    calibration_percentile=99.9,
    clip_max=3.0
):
    """
    Calibrate using validation only.

    Unlike the previous implementation, values above the
    calibration threshold are NOT collapsed to exactly 1.
    That preserves ranking among strong anomalies.
    """
    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    reference_scores = np.asarray(
        reference_scores,
        dtype=np.float32
    )
    reference_scores = reference_scores[
        np.isfinite(reference_scores)
    ]

    scale = float(
        np.percentile(
            reference_scores,
            calibration_percentile
        )
    )

    scale = max(
        scale,
        1e-8
    )

    normalized = (
        scores / scale
    )

    if clip_max is not None:
        normalized = np.clip(
            normalized,
            0.0,
            clip_max
        )

    return normalized.astype(
        np.float32
    ), scale


def fuse_scores(
    ae_score,
    temporal_score,
    w_ae=0.5,
    w_temporal=0.5
):
    ae_score = np.asarray(
        ae_score,
        dtype=np.float32
    )
    temporal_score = np.asarray(
        temporal_score,
        dtype=np.float32
    )

    if len(ae_score) != len(temporal_score):
        raise ValueError(
            "Score lengths must match."
        )

    total = w_ae + w_temporal

    if total <= 0:
        raise ValueError(
            "Fusion weights must be positive."
        )

    fused = (
        w_ae * ae_score +
        w_temporal * temporal_score
    ) / total

    return fused.astype(
        np.float32
    )


# ==========================================================
# Robust dynamic threshold
# ==========================================================

def _rolling_median_mad(
    values,
    window=1440,
    min_periods=60
):
    """
    Causal rolling median and MAD:
    only current/past values are used.
    """
    series = pd.Series(
        np.asarray(
            values,
            dtype=np.float64
        )
    )

    median = (
        series
        .rolling(
            window=window,
            min_periods=min_periods
        )
        .median()
    )

    mad = (
        series
        .rolling(
            window=window,
            min_periods=min_periods
        )
        .apply(
            lambda x: np.median(
                np.abs(
                    x - np.median(x)
                )
            ),
            raw=True
        )
    )

    return (
        median.to_numpy(),
        mad.to_numpy()
    )


def _filter_consecutive(
    binary_predictions,
    min_duration=2
):
    """
    Keep only runs with at least min_duration consecutive positives.
    """
    pred = np.asarray(
        binary_predictions,
        dtype=np.int8
    )

    if min_duration <= 1:
        return pred.copy()

    result = np.zeros_like(
        pred
    )

    start = None

    for i, value in enumerate(pred):

        if value == 1 and start is None:
            start = i

        if value == 0 and start is not None:
            end = i
            length = end - start

            if length >= min_duration:
                result[start:end] = 1

            start = None

    if start is not None:
        length = len(pred) - start
        if length >= min_duration:
            result[start:] = 1

    return result


def detect_anomalies_robust_dynamic(
    errors,
    reference_errors,
    window=1440,
    k=4.0,
    min_periods=60,
    floor_percentile=99.0,
    static_floor=None,
    cap=None,
    min_duration=2,
    return_components=False
):
    """
    Robust Layer-3 threshold.

    local threshold:
        rolling median + k * 1.4826 * rolling MAD

    floor:
        max(validation percentile floor, static_floor)

    The floor is deliberately taken from validation, never from test.
    """

    errors = np.asarray(
        errors,
        dtype=np.float64
    ).reshape(-1)

    reference_errors = np.asarray(
        reference_errors,
        dtype=np.float64
    ).reshape(-1)

    if len(errors) == 0:
        raise ValueError(
            "errors is empty."
        )

    reference_errors = reference_errors[
        np.isfinite(reference_errors)
    ]

    if len(reference_errors) == 0:
        raise ValueError(
            "reference_errors is empty."
        )

    if min_periods > window:
        min_periods = window

    rolling_median, rolling_mad = (
        _rolling_median_mad(
            errors,
            window=window,
            min_periods=min_periods
        )
    )

    local_threshold = (
        rolling_median +
        k * 1.4826 * rolling_mad
    )

    reference_floor = float(
        np.percentile(
            reference_errors,
            floor_percentile
        )
    )

    if static_floor is not None:
        threshold_floor = max(
            reference_floor,
            float(static_floor)
        )
    else:
        threshold_floor = reference_floor

    # Fill early NaNs with validation floor.
    local_threshold = np.where(
        np.isfinite(local_threshold),
        local_threshold,
        threshold_floor
    )

    final_threshold = np.maximum(
        local_threshold,
        threshold_floor
    )

    if cap is not None:
        final_threshold = np.minimum(
            final_threshold,
            float(cap)
        )

    raw_predictions = (
        errors >= final_threshold
    ).astype(np.int8)

    final_predictions = _filter_consecutive(
        raw_predictions,
        min_duration=min_duration
    )

    if return_components:
        return (
            final_predictions,
            {
                "local_threshold": local_threshold.astype(
                    np.float32
                ),
                "final_threshold": final_threshold.astype(
                    np.float32
                ),
                "raw_predictions": raw_predictions,
                "threshold_floor": threshold_floor
            }
        )

    return (
        final_predictions,
        final_threshold.astype(
            np.float32
        )
    )


# ==========================================================
# Statistics
# ==========================================================

def print_statistics(
    errors,
    predictions=None,
    name="Error"
):
    errors = np.asarray(
        errors
    ).reshape(-1)

    print(
        f"\n===== {name} ====="
    )

    print(
        "Min :",
        errors.min()
    )
    print(
        "Max :",
        errors.max()
    )
    print(
        "Mean:",
        errors.mean()
    )
    print(
        "Std :",
        errors.std()
    )

    print("\nPercentiles:")

    for p in [
        90,
        95,
        97,
        98,
        99,
        99.5,
        99.9
    ]:
        print(
            f"{p}% :",
            np.percentile(
                errors,
                p
            )
        )

    if predictions is not None:
        predictions = np.asarray(
            predictions
        ).reshape(-1)

        print(
            "\nDetected Anomalies:",
            predictions.sum()
        )

        print(
            "Anomaly Percentage:",
            predictions.mean() * 100,
            "%"
        )


# ==========================================================
# Plots
# ==========================================================

def plot_error_distribution(
    validation_errors,
    test_errors,
    threshold
):
    plt.figure(
        figsize=(12, 5)
    )

    plt.hist(
        validation_errors,
        bins=80,
        alpha=0.55,
        density=True,
        label="Validation"
    )

    plt.hist(
        test_errors,
        bins=80,
        alpha=0.55,
        density=True,
        label="Test"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold={threshold:.4f}"
    )

    plt.xlabel(
        "Score"
    )
    plt.ylabel(
        "Density"
    )
    plt.title(
        "Validation vs Test Score Distribution"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dynamic_threshold(
    errors,
    thresholds,
    predictions,
    max_points=15000
):
    errors = np.asarray(
        errors
    ).reshape(-1)

    thresholds = np.asarray(
        thresholds
    ).reshape(-1)

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    n = min(
        len(errors),
        len(thresholds),
        len(predictions),
        max_points
    )

    x = np.arange(n)

    plt.figure(
        figsize=(16, 5)
    )

    plt.plot(
        x,
        errors[:n],
        linewidth=0.8,
        label="Fused Score"
    )

    plt.plot(
        x,
        thresholds[:n],
        linewidth=1.2,
        label="Dynamic Threshold"
    )

    idx = np.where(
        predictions[:n] == 1
    )[0]

    if len(idx) > 0:
        plt.scatter(
            idx,
            errors[:n][idx],
            s=10,
            label="Final Anomaly"
        )

    plt.xlabel(
        "Sequence Index"
    )
    plt.ylabel(
        "Score"
    )
    plt.title(
        "Robust Dynamic Threshold"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anomalies(
    values,
    predictions,
    max_points=15000
):
    values = np.asarray(
        values
    ).reshape(-1)

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    n = min(
        len(values),
        len(predictions),
        max_points
    )

    values = values[:n]
    predictions = predictions[:n]

    idx = np.where(
        predictions == 1
    )[0]

    plt.figure(
        figsize=(16, 5)
    )

    plt.plot(
        values,
        linewidth=0.7,
        label="Aggregate Power"
    )

    if len(idx) > 0:
        plt.scatter(
            idx,
            values[idx],
            s=12,
            label="Detected Anomaly"
        )

    plt.xlabel(
        "Sequence Index"
    )
    plt.ylabel(
        "Power (VA)"
    )
    plt.title(
        "Detected Anomalies on UK-DALE Aggregate"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
