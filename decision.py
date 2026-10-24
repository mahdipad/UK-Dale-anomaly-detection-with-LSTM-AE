import numpy as np


def dynamic_threshold_mad(
        errors,
        window=1440,
        k=3.5,
        min_periods=60
):

    errors = np.asarray(
        errors,
        dtype=np.float64
    )

    thresholds = np.zeros(
        len(errors)
    )

    for i in range(len(errors)):

        start = max(
            0,
            i - window + 1
        )

        local = errors[
            start:i + 1
        ]

        if len(local) < min_periods:

            thresholds[i] = np.percentile(
                errors,
                99.9
            )

            continue

        median = np.median(local)

        mad = np.median(
            np.abs(local - median)
        )

        # Robust standard deviation
        robust_std = 1.4826 * mad

        thresholds[i] = (
            median +
            k * robust_std
        )

    return thresholds