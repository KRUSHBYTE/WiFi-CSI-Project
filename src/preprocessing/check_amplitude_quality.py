
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Load the experimental amplitude matrix
file = ROOT / "results" / "fall10_candidate.npy"

amplitude = np.load(file)

print("Matrix shape:", amplitude.shape)
print("Data type:", amplitude.dtype)

# Basic numerical quality
print("\nNumerical checks:")
print("All finite:", np.isfinite(amplitude).all())
print("Negative amplitudes:", np.count_nonzero(amplitude < 0))
print("Zero amplitudes:", np.count_nonzero(amplitude == 0))

# Distribution
print("\nAmplitude distribution:")
for p in [0, 1, 25, 50, 75, 99, 100]:
    print(f"Percentile {p}: {np.percentile(amplitude, p):.6f}")

# Temporal variation across observations
changes = np.abs(np.diff(amplitude, axis=0))
motion_score = changes.mean(axis=1)

print("\nTemporal variation:")
print("Mean change:", motion_score.mean())
print("Maximum change:", motion_score.max())

# Per-subcarrier variability
subcarrier_std = amplitude.std(axis=0)

print("\nSubcarrier variability:")
print("Minimum std:", subcarrier_std.min())
print("Maximum std:", subcarrier_std.max())
print("Zero-variance subcarriers:",
      np.count_nonzero(subcarrier_std == 0))
