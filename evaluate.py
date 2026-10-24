import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def inject_synthetic_anomalies(
    df,
    n_events=16,
    seed=42
):
    rng = np.random.default_rng(seed)

    result = df.copy()
    result["synthetic_anomaly"] = 0

    n = len(result)

    starts = np.linspace(
        1000,
        n - 1000,
        n_events,
        dtype=int
    )

    anomaly_types = [
        "spike",
        "drop",
        "high_level",
        "low_level",
        "abrupt_change",
        "contextual_shift"
    ]

    events = []

    for event_id, start in enumerate(starts):
        kind = anomaly_types[
            event_id % len(anomaly_types)
        ]

        duration = int(
            rng.integers(5, 11)
        )

        end = min(
            start + duration - 1,
            n - 1
        )

        original = (
            result["aggregate"]
            .iloc[start:end + 1]
            .to_numpy(np.float32)
        )

        mean = float(original.mean())
        std = float(original.std())

        std = max(
            std,
            max(mean * 0.05, 1.0)
        )

        if kind == "spike":
            modified = original + 5.0 * std
        elif kind == "drop":
            modified = original * 0.10
        elif kind == "high_level":
            modified = original * 2.5
        elif kind == "low_level":
            modified = original * 0.30
        elif kind == "abrupt_change":
            modified = original + 4.0 * std
        else:
            modified = np.maximum(
                original,
                mean + 5.0 * std
            )

        idx = result.index[
            start:end + 1
        ]

        result.loc[
            idx,
            "aggregate"
        ] = modified.astype(np.float32)

        result.loc[
            idx,
            "synthetic_anomaly"
        ] = 1

        events.append(
            {
                "event_id": event_id,
                "type": kind,
                "start": str(result.index[start]),
                "end": str(result.index[end]),
                "duration_minutes": end - start + 1
            }
        )

    return (
        result,
        result["synthetic_anomaly"].to_numpy(np.int8),
        events
    )


def endpoint_window_labels(
    point_labels,
    starts,
    sequence_length
):
    ends = (
        np.asarray(starts)
        + sequence_length
        - 1
    )

    return np.asarray(
        point_labels,
        dtype=np.int8
    )[ends]


def binary_metrics(
    y_true,
    y_pred
):
    y_true = np.asarray(y_true, np.int8)
    y_pred = np.asarray(y_pred, np.int8)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    accuracy = (
        (tp + tn) / max(len(y_true), 1)
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0 else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if fn + tp > 0 else 0.0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "FPR": fpr,
        "FNR": fnr
    }


def ranking_metrics(
    y_true,
    scores
):
    y_true = np.asarray(y_true, np.int8)
    scores = np.asarray(scores, np.float64)

    result = {
        "PR_AUC": np.nan,
        "ROC_AUC": np.nan
    }

    if len(np.unique(y_true)) < 2:
        return result

    try:
        result["PR_AUC"] = float(
            average_precision_score(
                y_true,
                scores
            )
        )
    except Exception:
        pass

    try:
        result["ROC_AUC"] = float(
            roc_auc_score(
                y_true,
                scores
            )
        )
    except Exception:
        pass

    return result


def print_metrics(
    name,
    metrics
):
    print(
        f"\n===== {name} ====="
    )

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )


def evaluation_table(
    results
):
    rows = []

    for name, metrics in results.items():
        rows.append(
            {
                "Model": name,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "Accuracy": metrics["Accuracy"],
                "FPR": metrics["FPR"],
                "FNR": metrics["FNR"],
                "PR_AUC": metrics.get("PR_AUC", np.nan),
                "ROC_AUC": metrics.get("ROC_AUC", np.nan)
            }
        )

    return pd.DataFrame(rows)


def tune_threshold(
    y_true,
    scores,
    min_precision=0.05
):
    """
    Calibrate a threshold on a separate labelled validation set.

    Selection criterion:
        maximize F1 subject to a minimum precision.
    """
    y_true = np.asarray(y_true, np.int8)
    scores = np.asarray(scores, np.float64)

    percentiles = np.linspace(
        70.0,
        99.9,
        120
    )

    thresholds = np.unique(
        np.percentile(
            scores,
            percentiles
        )
    )

    best = None
    best_f1 = -1.0

    for threshold in thresholds:
        pred = (
            scores >= threshold
        ).astype(np.int8)

        metrics = binary_metrics(
            y_true,
            pred
        )

        if (
            metrics["Precision"] >= min_precision
            and metrics["F1"] > best_f1
        ):
            best_f1 = metrics["F1"]
            best = (
                float(threshold),
                metrics
            )

    if best is None:
        best_threshold = float(
            np.percentile(
                scores,
                99.0
            )
        )
        best_metrics = binary_metrics(
            y_true,
            (
                scores >= best_threshold
            ).astype(np.int8)
        )
        return best_threshold, best_metrics

    return best


def tune_candidate_percentile(
    y_true,
    scores,
    percentiles=(
        85,
        87,
        90,
        92,
        94,
        95,
        96,
        97,
        98
    ),
    min_recall=0.90
):
    """
    Choose the strictest candidate threshold whose validation
    recall remains high enough to protect the hierarchical stage.
    """
    best = None

    for percentile in percentiles:
        threshold = np.percentile(
            scores,
            percentile
        )

        pred = (
            scores >= threshold
        ).astype(np.int8)

        metrics = binary_metrics(
            y_true,
            pred
        )

        candidate_rate = pred.mean()

        if (
            metrics["Recall"] >= min_recall
            and (
                best is None
                or candidate_rate < best["candidate_rate"]
            )
        ):
            best = {
                "percentile": percentile,
                "threshold": float(threshold),
                "recall": metrics["Recall"],
                "candidate_rate": candidate_rate
            }

    if best is None:
        percentile = 85.0
        threshold = np.percentile(
            scores,
            percentile
        )
        pred = (
            scores >= threshold
        ).astype(np.int8)

        metrics = binary_metrics(
            y_true,
            pred
        )

        best = {
            "percentile": percentile,
            "threshold": float(threshold),
            "recall": metrics["Recall"],
            "candidate_rate": pred.mean()
        }

    return best
