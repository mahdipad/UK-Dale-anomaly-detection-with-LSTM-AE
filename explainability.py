import numpy as np


def feature_importance_from_error(
        original,
        reconstructed
):

    error = np.abs(
        original - reconstructed
    )

    feature_importance = np.mean(
        error,
        axis=1
    )

    return feature_importance