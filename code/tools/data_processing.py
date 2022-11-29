from copy import deepcopy
from itertools import compress

import colorednoise as cn
from mne.time_frequency import tfr_array_morlet
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import torch
from torch.utils import data

from .feature_extraction import FeatureExtractor


def wavelet_tf(timeseries: np.ndarray, fs: float) -> np.ndarray:
    """Wavelet-based time-frequency decomposition of epoched waveforms

    Args:
        timeseries (np.ndarray): Shape (n_epochs, n_channels, n_times)
        fs (float): Sampling frequency

    Returns:
        np.ndarray: Shape (n_epochs, n_channels, n_freqs, n_times_decimated)
    """

    frequencies = np.arange(8, 201, 3)
    n_cycles = frequencies / 6

    power = tfr_array_morlet(
        timeseries,
        n_cycles=n_cycles,
        freqs=frequencies,
        decim=4,
        output="power",
        sfreq=fs,
    )

    return power


class Dataset(pl.LightningDataModule):
    def __init__(
        self,
        timeseries: np.ndarray,
        label: np.ndarray,
        Fs: float,
        window_length: float = 0.250,
        hop_size: float = 0.250,
        batch_size: int = 96,
        tf_transform: bool = False,
        quantile_transform: bool = False,
        ar_len: int | None = 10,
        feat_ar: bool = True,
        n_folds: int = 5,
        fold_id: int = 4,
        data_aug_intensity: float = 0.2,
    ):
        self.data_aug_intensity = data_aug_intensity

        ar_len = None if ar_len == 0 else ar_len

        self.bs = batch_size
        N = len(label)

        timeseries = timeseries.astype(np.float32)
        if timeseries.ndim < 2:
            timeseries = timeseries[np.newaxis, ...]

        self.timeseries = timeseries
        self.n_channels = timeseries.shape[0]

        # Epoch data:
        idx_start = np.arange(0, N - 1, int(Fs * hop_size))
        idx_end = idx_start + int(Fs * window_length)
        while idx_end[-1] >= N:
            idx_end = idx_end[:-1]
            idx_start = idx_start[:-1]

        self.X = [
            timeseries[:, id_start:id_end]
            for id_start, id_end in zip(idx_start, idx_end)
        ]
        self.labels = [
            np.mean(label[id_start:id_end]) > 0.25
            for id_start, id_end in zip(idx_start, idx_end)
        ]

        # Train / Validation split:
        assert fold_id < n_folds, "fold_id is greater than number of folds"
        idx = np.ones(len(self.X))
        idx[
            int(fold_id * len(self.X) // n_folds) : int(
                (fold_id + 1) * len(self.X) // n_folds
            )
        ] = 0
        is_train_idx = idx.astype(bool)

        # Prepare epoched timeseries:
        self.X_train = list(compress(self.X, is_train_idx))
        self.X_valid = list(compress(self.X, np.logical_not(is_train_idx)))

        assert all(
            el.shape == self.X_train[0].shape for el in self.X_train
        ), "Train data with unexpected shape encountered"
        assert all(
            el.shape == self.X_valid[0].shape for el in self.X_valid
        ), "Valid data with unexpected shape encountered"

        self.y_train = list(compress(self.labels, is_train_idx))
        self.y_valid = list(compress(self.labels, np.logical_not(is_train_idx)))

        if quantile_transform:
            x_train = np.concatenate(self.X_train, axis=-1)
            qt = QuantileTransformer(n_quantiles=4096).fit(x_train.T)
            self.X_train = [qt.transform(x_.T).T for x_ in self.X_train]
            self.X_valid = [qt.transform(x_.T).T for x_ in self.X_valid]

        # z-score timeseries:
        mean = np.mean([np.mean(x_, axis=-1) for x_ in self.X_train], axis=0)[..., None]
        std = (
            np.mean([np.std(x_, axis=-1) for x_ in self.X_train], axis=0)[..., None]
            + 1e-8
        )

        self.X_train = [(x_ - mean) / std for x_ in self.X_train]
        self.X_valid = [(x_ - mean) / std for x_ in self.X_valid]

        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.valid_data = [(x, y) for x, y in zip(self.X_valid, self.y_valid)]

        # Extract and prepare features:
        self.X_features = FeatureExtractor(Fs).extract_features(np.stack(self.X))

        self.X_features_train = self.X_features[is_train_idx, :]
        self.X_features_valid = self.X_features[np.logical_not(is_train_idx), :]

        self.scaler = StandardScaler().fit(self.X_features_train)
        self.X_features_train_scaled = self.scaler.transform(self.X_features_train)
        self.X_features_valid_scaled = self.scaler.transform(self.X_features_valid)

        if (
            ar_len is not None
        ):  # TODO: For folds that are not first or last avoid breaks in continuity in sequence data
            train_with_feats = [
                (x, feats, y)
                for (x, y), feats in zip(self.train_data, self.X_features_train_scaled)
            ]
            valid_with_feats = [
                (x, feats, y)
                for (x, y), feats in zip(self.valid_data, self.X_features_valid_scaled)
            ]
            self.train_ar, self.valid_ar = [], []
            for jj in range(len(train_with_feats) - ar_len):
                self.train_ar.append(train_with_feats[jj : jj + ar_len])

            for jj in range(len(valid_with_feats) - ar_len):
                self.valid_ar.append(valid_with_feats[jj : jj + ar_len])

        if feat_ar:
            if ar_len is None:
                raise ValueError("To use sequence of features, `ar_len` cannot be None")

            self.X_features_ar_train = np.stack(
                [[x_[1] for x_ in el] for el in self.train_ar]
            ).reshape(-1, ar_len * self.X_features.shape[1])
            self.y_ar_train = [el[-1][2] for el in self.train_ar]

            self.X_features_ar_valid = np.stack(
                [[x_[1] for x_ in el] for el in self.valid_ar]
            ).reshape(-1, ar_len * self.X_features.shape[1])
            self.y_ar_valid = [el[-1][2] for el in self.valid_ar]

        if tf_transform:
            power = wavelet_tf(np.stack(self.X), Fs).astype(
                np.float32
            )  # (n_samples, n_chans, n_freqs, n_times)

            self.X_tf_train = power[is_train_idx, ...]
            self.X_tf_valid = power[np.logical_not(is_train_idx), ...]

            # Z-score each frequency (extract stats from training data only):
            mean = np.expand_dims(
                np.expand_dims(
                    np.nanmean(self.X_tf_train, axis=(0, -1), keepdims=False), 0
                ),
                -1,
            )
            std = (
                np.expand_dims(
                    np.expand_dims(
                        np.nanstd(self.X_tf_train, axis=(0, -1), keepdims=False), 0
                    ),
                    -1,
                )
                + 1e-8
            )

            self.X_tf_train = (self.X_tf_train - mean) / std
            self.X_tf_valid = (self.X_tf_valid - mean) / std

            self.tf_train_data = [(x, y) for x, y in zip(self.X_tf_train, self.y_train)]
            self.tf_valid_data = [(x, y) for x, y in zip(self.X_tf_valid, self.y_valid)]

    def get_n_channels(self):
        return self.n_channels

    def train_ar_dataloader(self):
        dataset = DatasetAR(self.train_ar, self.data_aug_intensity)
        return data.DataLoader(dataset, self.bs, shuffle=True, drop_last=True)

    def valid_ar_dataloader(self):
        return data.DataLoader(self.valid_ar, self.bs, shuffle=False, drop_last=True)

    def train_tf_dataloader(self):
        return data.DataLoader(
            self.tf_train_data, self.bs, shuffle=True, drop_last=True
        )

    def valid_tf_dataloader(self):
        return data.DataLoader(
            self.tf_valid_data, self.bs, shuffle=False, drop_last=True
        )

    def train_dataloader(self):
        dataset = Dataset1d(self.train_data, self.data_aug_intensity)
        return data.DataLoader(dataset, self.bs, shuffle=True, drop_last=True)

    def valid_dataloader(self):
        return data.DataLoader(self.valid_data, self.bs, shuffle=False, drop_last=True)


class ColoredNoiseAdder:
    """Adds random 1-f noise to a single sample"""

    def __init__(self, intensity: float = 0.1):
        self.intensity = intensity

    def __call__(self, signal: np.ndarray):
        """
        Args
        ----
            signal (np.ndarray): Single signal epoch of shape (n_chan, n_times)
        """

        if self.intensity == 0.0 or (randomizer := np.random.rand(1))[0] < 0.1:
            return torch.tensor(signal).float()

        rms = np.sqrt(np.mean(signal**2, axis=-1))
        noise = cn.powerlaw_psd_gaussian(1, signal.shape)

        return torch.tensor(
            signal + self.intensity * rms[..., None] * noise * randomizer
        ).float()


class DatasetAR(data.Dataset):
    def __init__(self, train_data: list, intensity: float = 0.1):
        self.train_data = train_data
        self.transform = ColoredNoiseAdder(intensity)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return [(self.transform(el[0]), el[1], el[2]) for el in self.train_data[idx]]


class Dataset1d(data.Dataset):
    def __init__(self, train_data: list, intensity: float = 0.1):

        self.train_data = train_data
        self.transform = ColoredNoiseAdder(intensity)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        x, y = self.train_data[idx]
        return self.transform(x), y
