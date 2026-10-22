
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data/candidate_csv/room_A"
OUTPUT_DIR = ROOT / "data/strict_candidate_windows"
SPLIT_FILE = ROOT / "configs/room_a_split.csv"

WINDOW_SIZE = 150
metadata = pd.read_csv(SPLIT_FILE)

processed = 0
padded = 0

for _, record in metadata.iterrows():
    relative = Path(record["capture_id"]).with_suffix(".csv")
    source = INPUT_DIR / relative

    data = pd.read_csv(source).to_numpy(dtype=np.float32)

    if data.shape[1] != 256 or len(data) == 0:
        raise ValueError(f"Invalid capture: {source}")

    if len(data) >= WINDOW_SIZE:
        start = (len(data) - WINDOW_SIZE) // 2
        window = data[start:start + WINDOW_SIZE]
    else:
        print(f"Short capture: {source.name} ({len(data)} observations)")
        continue

    if window.shape != (150, 256):
        raise ValueError(f"Incorrect window shape: {source}")

    destination = (
        OUTPUT_DIR / record["split"] / "room_A" /
        relative.with_suffix(".npy")
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, window)

    processed += 1

print("Total windows:", processed)
print("Padded captures:", padded)
print("Window shape:", window.shape)
