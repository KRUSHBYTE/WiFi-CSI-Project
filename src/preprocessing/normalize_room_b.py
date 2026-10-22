
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data/windows"
OUTPUT_DIR = ROOT / "data/normalized"

development_dir = INPUT_DIR / "development/room_B"

files = sorted(development_dir.glob("*.npy"))

if not files:
    raise ValueError("No development windows found")

# Calculate per-subcarrier statistics
total = np.zeros(256, dtype=np.float64)
total_sq = np.zeros(256, dtype=np.float64)
count = 0

for file in files:
    data = np.load(file)

    total += data.sum(axis=0)
    total_sq += np.square(data.astype(np.float64)).sum(axis=0)
    count += data.shape[0]

mean = total / count
variance = np.maximum(total_sq / count - mean ** 2, 0)
std = np.sqrt(variance)

# Avoid division by zero
std = np.where(std < 1e-8, 1.0, std)

# Save normalization parameters
CONFIG_DIR = ROOT / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

np.savez(
    CONFIG_DIR / "room_b_normalization.npz",
    mean=mean,
    std=std
)

# Apply the same parameters to both splits
for split in ["development", "validation"]:

    source = INPUT_DIR / split / "room_B"
    destination = OUTPUT_DIR / split / "room_B"

    destination.mkdir(parents=True, exist_ok=True)

    processed = 0

    for file in sorted(source.glob("*.npy")):

        data = np.load(file).astype(np.float64)

        normalized = ((data - mean) / std).astype(np.float32)

        if not np.isfinite(normalized).all():
            raise ValueError(f"Invalid output: {file.name}")

        np.save(destination / file.name, normalized)
        processed += 1

    print(f"{split}: {processed} windows")

print("Normalization completed")
