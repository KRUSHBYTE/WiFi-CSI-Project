"""Capture ingestion -> common 256-column amplitude form -> QC -> 150x256 window.

Used by the batch dataset builder AND by run_episode.py, so development,
validation and unseen episode data all pass through identical code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.pcap_reader import PcapStats, iter_udp_payloads

N_SUBCARRIERS = 256
COLUMNS = [f"subcarrier_{i:03d}" for i in range(1, N_SUBCARRIERS + 1)]


# ---------------------------------------------------------------------------
# Experiment-2 (Room A): Nexmon BCM4366c0 packed CSI
# ---------------------------------------------------------------------------

def decode_csi_batch(words: np.ndarray) -> np.ndarray:
    """Vectorized form of ``preprocessing.csi_decode.decode_csi``.

    ``words``: (N, 256) uint32 packed words, one row per packet.
    Bit layout per word (BCM4366c0, 80 MHz):
        bit 29       real sign      bits 18-28  real mantissa (11 bits)
        bit 17       imag sign      bits  6-16  imag mantissa (11 bits)
        bits 0-5     shared exponent (6-bit two's complement)
    Each packet is rescaled so that its largest component occupies 11 bits
    (packet-wise scaling). Verified bit-exact against the scalar reference in
    scripts/verify_decoder.py.
    """
    words = np.asarray(words, dtype=np.uint32)
    if words.ndim != 2 or words.shape[1] != N_SUBCARRIERS:
        raise ValueError(f"expected (N, 256) words, got {words.shape}")

    real_mag = ((words >> 18) & 0x7FF).astype(np.int64)
    imag_mag = ((words >> 6) & 0x7FF).astype(np.int64)
    real_neg = ((words >> 29) & 1).astype(bool)
    imag_neg = ((words >> 17) & 1).astype(bool)
    exponent = (words & 0x3F).astype(np.int64)
    exponent[exponent >= 32] -= 64

    magnitude = real_mag | imag_mag
    nonzero = magnitude > 0
    bit_length = np.zeros_like(magnitude)
    bit_length[nonzero] = np.floor(np.log2(magnitude[nonzero])).astype(np.int64) + 1
    effective = np.where(nonzero, exponent + bit_length - 1, -32)
    maxbit = np.maximum(effective.max(axis=1, keepdims=True), -32)
    shift = 10 - maxbit
    e = exponent + shift                                  # per-subcarrier scale

    def scale(mag, negative):
        out = np.zeros(mag.shape, dtype=np.float64)
        left = e >= 0
        right = (e < 0) & (e >= -12)
        # Left shifts only matter where mag > 0, and then e <= 10, so no overflow.
        out[left] = (mag[left] << np.minimum(e[left], 62)).astype(np.float64)
        out[right] = (mag[right] >> (-e[right])).astype(np.float64)
        out[negative] *= -1
        return out

    return scale(real_mag, real_neg) + 1j * scale(imag_mag, imag_neg)


def read_room_a_capture(path: str | Path, cfg) -> tuple[np.ndarray, dict]:
    """Return (amplitude[N,256] float32, report) for one Experiment-2 PCAP."""
    pcfg = cfg["preprocessing"]["pcap"]
    payload_length = int(pcfg["payload_length"])
    magic = bytes.fromhex(str(pcfg["magic"]))

    stats = PcapStats()
    words, rejected_shape = [], 0
    for packet in iter_udp_payloads(path, stats):
        payload = packet.payload
        if len(payload) != payload_length or payload[:2] != magic:
            rejected_shape += 1
            continue
        words.append(np.frombuffer(payload, dtype="<u4", offset=18, count=N_SUBCARRIERS))

    report = {
        "records": stats.records,
        "truncated_records": stats.truncated,
        "non_udp_records": stats.non_udp,
        "rejected_payloads": rejected_shape,
        "csi_packets": len(words),
    }
    if not words:
        return np.zeros((0, N_SUBCARRIERS), dtype=np.float32), report
    csi = decode_csi_batch(np.stack(words))
    return np.abs(csi).astype(np.float32), report


# ---------------------------------------------------------------------------
# Experiment-3 (Room B): amplitude CSV
# ---------------------------------------------------------------------------

def read_room_b_capture(path: str | Path) -> tuple[np.ndarray, dict]:
    """Experiment-3 CSV: first row is a header (integer or Subcarrier_* names)."""
    frame = pd.read_csv(path, header=0)
    # Drop an accidental index column if present (e.g. 'Unnamed: 0').
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    if frame.shape[1] != N_SUBCARRIERS:
        raise ValueError(f"{path}: expected 256 amplitude columns, found {frame.shape[1]}")
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    report = {"records": int(values.shape[0]), "csi_packets": int(values.shape[0])}
    return values.astype(np.float32), report


def read_common_csv(path: str | Path) -> tuple[np.ndarray, dict]:
    frame = pd.read_csv(path)
    if list(frame.columns) != COLUMNS:
        return read_room_b_capture(path)
    values = frame.to_numpy(dtype=np.float32)
    return values, {"records": len(values), "csi_packets": len(values)}


def read_capture(path: str | Path, cfg) -> tuple[np.ndarray, dict]:
    """Dispatch on file *format* (ingestion only - never used as a model feature)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".pcap", ".cap"):
        return read_room_a_capture(path, cfg)
    if suffix == ".csv":
        return read_common_csv(path)
    if suffix == ".npy":
        window = np.load(path, allow_pickle=False).astype(np.float32)
        if window.ndim != 2 or window.shape[1] != N_SUBCARRIERS:
            raise ValueError(f"{path}: expected (T, 256) array, got {window.shape}")
        return window, {"records": len(window), "csi_packets": len(window)}
    raise ValueError(f"Unsupported capture format: {path}")


# ---------------------------------------------------------------------------
# Quality control and windowing
# ---------------------------------------------------------------------------

def quality_control(amplitude: np.ndarray, cfg) -> tuple[np.ndarray, dict]:
    """Remove observations that carry no valid CSI. Temporal order is preserved."""
    qc = cfg["preprocessing"]["qc"]
    keep = np.ones(len(amplitude), dtype=bool)
    report = {"qc_input_rows": int(len(amplitude))}

    finite = np.isfinite(amplitude).all(axis=1)
    report["qc_nonfinite_rows"] = int((~finite).sum())
    if qc.get("drop_nonfinite_rows", True):
        keep &= finite

    safe = np.where(np.isfinite(amplitude), amplitude, 0.0)
    zero = ~(np.abs(safe) > 0).any(axis=1)
    report["qc_zero_rows"] = int((zero & finite).sum())
    if qc.get("drop_zero_rows", True):
        keep &= ~zero

    duplicate = np.zeros(len(amplitude), dtype=bool)
    if len(amplitude) > 1:
        duplicate[1:] = (safe[1:] == safe[:-1]).all(axis=1) & ~zero[1:]
    report["qc_duplicate_rows"] = int(duplicate.sum())
    if qc.get("drop_consecutive_duplicates", True):
        keep &= ~duplicate

    negative = (safe < 0).any(axis=1)
    report["qc_negative_rows"] = int(negative.sum())
    if qc.get("drop_negative_rows", False):  # magnitudes cannot be negative
        keep &= ~negative

    cleaned = amplitude[keep]
    report["qc_valid_rows"] = int(len(cleaned))
    fraction = len(cleaned) / max(len(amplitude), 1)
    report["qc_rejected_capture"] = bool(
        len(amplitude) == 0 or fraction < float(qc.get("min_valid_fraction", 0.0))
    )
    return cleaned, report


def select_window(amplitude: np.ndarray, cfg, allow_short: bool = False):
    """Deterministic centre window of 150 consecutive valid observations.

    Returns (window or None, start index, padded flag). ``allow_short`` is used
    only at inference time (run_episode.py) so a short episode capture still
    gets a prediction; benchmark windows never use it.
    """
    size = int(cfg["preprocessing"]["window_size"])
    n = len(amplitude)
    if n >= size:
        start = (n - size) // 2
        return amplitude[start:start + size].astype(np.float32), start, False
    if not allow_short or n == 0:
        return None, -1, False
    pad_before = (size - n) // 2
    window = np.pad(amplitude, ((pad_before, size - n - pad_before), (0, 0)), mode="edge")
    return window.astype(np.float32), -pad_before, True


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """Per-subcarrier z-score with statistics fitted on development windows only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, name: str = ""):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.where(np.asarray(std, dtype=np.float64) < 1e-8, 1.0, std)
        self.name = name

    @classmethod
    def fit(cls, windows, name: str = "") -> "Normalizer":
        total = np.zeros(N_SUBCARRIERS)
        total_sq = np.zeros(N_SUBCARRIERS)
        count = 0
        for window in windows:
            w = np.asarray(window, dtype=np.float64)
            total += w.sum(axis=0)
            total_sq += np.square(w).sum(axis=0)
            count += len(w)
        if count == 0:
            raise ValueError("cannot fit normalizer on zero windows")
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
        return cls(mean, std, name)

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        data = np.load(path)
        return cls(data["mean"], data["std"], Path(path).stem)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        out = ((np.asarray(window, dtype=np.float64) - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("non-finite value after normalization")
        return out


def load_normalizers(cfg) -> dict:
    """{'A': room-A stats, 'B': room-B stats, 'pooled': both rooms} from configs/normalization."""
    folder = cfg.path("normalization_dir")
    return {name: Normalizer.load(folder / f"{name}.npz") for name in ("A", "B", "pooled")}
