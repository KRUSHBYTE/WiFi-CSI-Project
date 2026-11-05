"""Load the benchmark windows produced by src/preprocessing/build_dataset.py."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowSet:
    """Raw-amplitude windows with their manifest rows (same order)."""

    meta: pd.DataFrame          # window_id, capture_id, room, activity, split, ...
    windows: np.ndarray         # (N, 150, 256) float32 raw amplitude

    def subset(self, mask) -> "WindowSet":
        mask = np.asarray(mask)
        return WindowSet(self.meta[mask].reset_index(drop=True), self.windows[mask])

    def __len__(self):
        return len(self.meta)


def load_windows(cfg) -> WindowSet:
    root = cfg.path("windows")
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found - run src/preprocessing/build_dataset.py first")
    manifest = pd.read_csv(manifest_path)
    ok = manifest[manifest.status == "ok"].reset_index(drop=True)
    stale = set(ok.preprocessing_hash) - {cfg.preprocessing_hash}
    if stale:
        raise RuntimeError(
            f"windows were built with preprocessing config {stale}, current is "
            f"{cfg.preprocessing_hash}; rebuild with build_dataset.py")
    windows = np.stack([np.load(root / rel) for rel in ok.window_path]).astype(np.float32)
    return WindowSet(ok, windows)
