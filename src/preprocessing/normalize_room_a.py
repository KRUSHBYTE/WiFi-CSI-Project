
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/strict_candidate_windows"
OUTPUT = ROOT / "data/strict_normalized"
CONFIG = ROOT / "configs"

development = sorted(
    (INPUT / "development/room_A").rglob("*.npy")
)

if not development:
    raise ValueError("No Room A development windows found")

# Fit normalization using development data only
total = np.zeros(256, dtype=np.float64)
total_sq = np.zeros(256, dtype=np.float64)
count = 0

for file in development:
    data = np.load(file).astype(np.float64)

    if data.shape != (150, 256):
        raise ValueError(f"Invalid shape: {file}")

    total += data.sum(axis=0)
    total_sq += np.square(data).sum(axis=0)
    count += len(data)

mean = total / count
variance = np.maximum(total_sq / count - mean**2, 0)
std = np.sqrt(variance)
std = np.where(std < 1e-8, 1.0, std)

CONFIG.mkdir(exist_ok=True)
np.savez(CONFIG / "room_a_normalization.npz",
         mean=mean, std=std)

# Apply to both splits
for split in ["development", "validation"]:
    source = INPUT / split / "room_A"
    files = sorted(source.rglob("*.npy"))

    for file in files:
        data = np.load(file).astype(np.float64)
        normalized = ((data - mean) / std).astype(np.float32)

        if not np.isfinite(normalized).all():
            raise ValueError(f"Invalid values: {file}")

        destination = (
            OUTPUT / split / "room_A" / file.relative_to(source)
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, normalized)

    print(f"{split}: {len(files)} windows")

print("Candidate normalization completed")
