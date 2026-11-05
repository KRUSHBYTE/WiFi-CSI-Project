"""Task 1 recorded-environment fingerprint (Room A vs Room B) used for routing.

Input is only the 150x256 amplitude window (no filename, extension, length or
packet-count information). The window is standardized with the POOLED
development statistics (fitted on both rooms without room labels), so the
preprocessing itself does not reveal the room.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.features import handcrafted_features


def centre_crop(windows: np.ndarray, length: int) -> np.ndarray:
    total = windows.shape[1]
    start = (total - length) // 2
    return windows[:, start:start + length]


def router_features(windows: np.ndarray, pooled, gain_invariant: bool = False) -> np.ndarray:
    """(N, T, 256) raw amplitude -> handcrafted features."""
    windows = np.asarray(windows, dtype=np.float64)
    if gain_invariant:
        # Remove the per-window gain so only the spectral *shape* remains.
        logs = np.log(np.maximum(windows, 1e-6))
        x = logs - logs.mean(axis=(1, 2), keepdims=True)
    else:
        x = (windows - pooled.mean) / pooled.std
    return handcrafted_features(x)


class RoomRouter:
    def __init__(self, pooled, gain_invariant: bool = False, C: float = 1.0, length: int = 150):
        self.pooled = pooled
        self.gain_invariant = gain_invariant
        self.length = length
        self.model = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=42))

    def _features(self, windows):
        if windows.shape[1] != self.length:
            windows = centre_crop(windows, self.length)
        return router_features(windows, self.pooled, self.gain_invariant)

    def fit(self, windows, rooms):
        self.model.fit(self._features(windows), np.asarray(rooms))
        return self

    def predict_proba(self, windows):
        return self.model.predict_proba(self._features(windows))

    def predict(self, windows):
        proba = self.predict_proba(windows)
        classes = self.model.classes_
        return classes[proba.argmax(axis=1)], proba.max(axis=1)

    @property
    def classes_(self):
        return self.model.classes_
