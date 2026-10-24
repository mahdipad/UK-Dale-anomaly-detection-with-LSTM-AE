import numpy as np
import matplotlib.pyplot as plt


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
        np.square(
            X - reconstructed
        ),
        axis=1
    )

    return errors.astype(np.float32)


def compute_threshold(
    errors,
    percentile=99.9
):
    errors = np.asarray(
        errors,
        dtype=np.float64
    )

    return float(
        np.percentile(
            errors,
            percentile
        )
    )


def detect_anomalies(
    errors,
    threshold
):
    return (
        np.asarray(errors) >= threshold
    ).astype(np.int8)


def normalize_score(
    scores,
    reference_scores,
    calibration_percentile=99.0,
    clip_max=5.0
):
    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    reference_scores = np.asarray(
        reference_scores,
        dtype=np.float32
    )

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

    return (
        np.clip(
            normalized,
            0.0,
            clip_max
        ).astype(np.float32),
        scale
    )


def fuse_scores(
    score_1,
    score_2,
    w_ae=0.5,
    w_temporal=0.5
):
    total = (
        w_ae
        + w_temporal
    )

    if total <= 0:
        raise ValueError(
            "Fusion weights must be positive."
        )

    return (
        (
            w_ae * np.asarray(score_1, np.float32)
            +
            w_temporal * np.asarray(score_2, np.float32)
        )
        / total
    ).astype(np.float32)


def adaptive_fusion(
    ae_score,
    temporal_score,
    context_features,
    validation_ae_score,
    validation_temporal_score,
    base_weight=0.5,
    min_weight=0.2,
    max_weight=0.8,
    context_strength=0.35
):
    """
    Context-adaptive fusion.

    Weight is NOT learned from test labels.
    It is determined from local volatility relative to
    validation volatility.

    High local variability -> trust temporal attention more.
    Low local variability  -> trust reconstruction more.
    """
    ae_score = np.asarray(
        ae_score,
        dtype=np.float32
    )

    temporal_score = np.asarray(
        temporal_score,
        dtype=np.float32
    )

    context_features = np.asarray(
        context_features,
        dtype=np.float32
    )

    if context_features.ndim != 2:
        raise ValueError(
            "context_features must be 2D."
        )

    # All inputs to fusion are window-level arrays. A context row
    # must correspond to exactly one AE/temporal score.
    n = len(ae_score)
    if len(temporal_score) != n:
        raise ValueError(
            "adaptive_fusion: ae_score and temporal_score "
            f"must have the same length, got {n} and "
            f"{len(temporal_score)}."
        )
    if len(context_features) != n:
        raise ValueError(
            "adaptive_fusion: context_features must contain "
            f"one row per score. Got {len(context_features)} "
            f"context rows for {n} scores."
        )

    # rolling_std is feature index 3 in the six-feature vector.
    local_volatility = context_features[:, 3]

    val_local_volatility = (
        np.median(
            np.asarray(
                validation_ae_score,
                dtype=np.float32
            )
        )
        +
        np.median(
            np.abs(
                np.asarray(
                    validation_temporal_score,
                    dtype=np.float32
                )
                -
                np.median(
                    validation_temporal_score
                )
            )
        )
    )

    scale = max(
        float(val_local_volatility),
        1e-6
    )

    normalized_volatility = (
        local_volatility / scale
    )

    # More volatile regimes -> lower AE weight.
    ae_weight = (
        base_weight
        -
        context_strength
        * np.tanh(
            normalized_volatility - 1.0
        )
    )

    ae_weight = np.clip(
        ae_weight,
        min_weight,
        max_weight
    )

    temporal_weight = (
        1.0 - ae_weight
    )

    fused = (
        ae_weight * ae_score
        +
        temporal_weight * temporal_score
    )

    return (
        fused.astype(np.float32),
        ae_weight.astype(np.float32),
        temporal_weight.astype(np.float32)
    )


def print_statistics(
    scores,
    predictions=None,
    name="Scores"
):
    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    print(
        f"\n===== {name} ====="
    )

    print("Min:", scores.min())
    print("Max:", scores.max())
    print("Mean:", scores.mean())
    print("Std:", scores.std())

    print("\nPercentiles:")

    for p in (
        90,
        95,
        97,
        98,
        99,
        99.5,
        99.9
    ):
        print(
            f"{p}%:",
            np.percentile(
                scores,
                p
            )
        )

    if predictions is not None:
        predictions = np.asarray(
            predictions
        )
        print(
            "Anomalies:",
            predictions.sum()
        )
        print(
            "Anomaly %:",
            predictions.mean() * 100
        )


def plot_error_distribution(
    validation_scores,
    test_scores,
    threshold
):
    plt.figure(
        figsize=(12, 5)
    )

    plt.hist(
        validation_scores,
        bins=80,
        alpha=0.55,
        density=True,
        label="Validation"
    )

    plt.hist(
        test_scores,
        bins=80,
        alpha=0.55,
        density=True,
        label="Test"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold={threshold:.6f}"
    )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title(
        "Validation vs Test Score Distribution"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anomalies(
    power,
    predictions,
    max_points=15000
):
    n = min(
        len(power),
        len(predictions),
        max_points
    )

    power = power[:n]
    predictions = predictions[:n]

    idx = np.where(
        predictions == 1
    )[0]

    plt.figure(
        figsize=(16, 5)
    )

    plt.plot(
        power,
        linewidth=0.7,
        label="Aggregate Power"
    )

    if len(idx) > 0:
        plt.scatter(
            idx,
            power[idx],
            s=12,
            label="Anomaly"
        )

    plt.xlabel("Sequence")
    plt.ylabel("Power")
    plt.title(
        "Detected Anomalies"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
