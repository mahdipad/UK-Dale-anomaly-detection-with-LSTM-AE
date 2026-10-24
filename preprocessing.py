from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = [
    "aggregate",
    "delta",
    "rolling_mean",
    "rolling_std",
    "hour",
    "dayofweek"
]


class UKDALELoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def load_aggregate(self, house=1):
        path = (
            self.base_path
            / f"house_{house}"
            / "channel_1.dat"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["timestamp", "aggregate"],
            dtype={"aggregate": "float32"}
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="s"
        )

        return (
            df.sort_values("timestamp")
            .set_index("timestamp")
        )

    def summarize(self, df):
        print("\n========== DATA SUMMARY ==========")
        print(df.info())
        print("\n========== DESCRIBE ==========")
        print(df.describe())
        print("\nDuplicate timestamps:")
        print(df.index.duplicated().sum())
        print("\nMissing values:")
        print(df.isnull().sum())

    def resample_data(self, df, freq="1min", method="mean"):
        if method == "mean":
            return df.resample(freq).mean()
        if method == "median":
            return df.resample(freq).median()
        if method == "max":
            return df.resample(freq).max()
        raise ValueError(
            "method must be mean, median, or max."
        )

    def handle_missing(self, df, method="ffill"):
        if method == "ffill":
            return df.ffill()
        if method == "bfill":
            return df.bfill()
        if method == "interpolate":
            return df.interpolate()
        if method == "drop":
            return df.dropna()
        raise ValueError(
            "Unsupported missing-value method."
        )

    def train_test_split(self, df, train_ratio=0.8):
        split = int(len(df) * train_ratio)
        return (
            df.iloc[:split].copy(),
            df.iloc[split:].copy()
        )

    def add_features(self, df, rolling_window=5):
        result = df.copy()

        result["delta"] = result["aggregate"].diff()

        result["rolling_mean"] = (
            result["aggregate"]
            .rolling(
                window=rolling_window,
                min_periods=1
            )
            .mean()
        )

        result["rolling_std"] = (
            result["aggregate"]
            .rolling(
                window=rolling_window,
                min_periods=1
            )
            .std()
            .fillna(0.0)
        )

        result["hour"] = result.index.hour
        result["dayofweek"] = result.index.dayofweek

        return result.ffill().bfill()

    def normalize(self, train_df, other_df):
        scaler = MinMaxScaler()

        train_values = (
            train_df[FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        other_values = (
            other_df[FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        train_scaled = scaler.fit_transform(
            train_values
        )

        other_scaled = scaler.transform(
            other_values
        )

        return (
            train_scaled.astype(np.float32),
            other_scaled.astype(np.float32),
            scaler
        )

    @staticmethod
    def scale_with_existing_scaler(
        scaler,
        df
    ):
        values = (
            df[FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )
        return scaler.transform(
            values
        ).astype(np.float32)

    @staticmethod
    def create_sequences(
        data,
        sequence_length
    ):
        data = np.asarray(
            data,
            dtype=np.float32
        )

        if data.ndim != 2:
            raise ValueError(
                "Expected shape (samples, features)."
            )

        if len(data) < sequence_length:
            raise ValueError(
                "Not enough rows."
            )

        n = (
            len(data)
            - sequence_length
            + 1
        )

        shape = (
            n,
            sequence_length,
            data.shape[1]
        )

        strides = (
            data.strides[0],
            data.strides[0],
            data.strides[1]
        )

        windows = np.lib.stride_tricks.as_strided(
            data,
            shape=shape,
            strides=strides,
            writeable=False
        )

        return np.ascontiguousarray(
            windows,
            dtype=np.float32
        )

    @staticmethod
    def create_sequences_at_indices(
        data,
        start_indices,
        sequence_length
    ):
        data = np.asarray(
            data,
            dtype=np.float32
        )

        starts = np.asarray(
            start_indices,
            dtype=np.int64
        )

        return np.stack(
            [
                data[i:i + sequence_length]
                for i in starts
            ],
            axis=0
        ).astype(np.float32, copy=False)
