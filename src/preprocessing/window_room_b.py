
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data/intermediate_csv/room_B"
OUTPUT_DIR = ROOT / "data/windows"

split_file = ROOT / "configs/room_b_split.csv"

metadata = pd.read_csv(split_file)

WINDOW_SIZE = 150
processed = 0

for _, record in metadata.iterrows():

    file = INPUT_DIR / record["capture_id"]

    data = pd.read_csv(file).to_numpy(dtype=np.float32)

    if data.shape[1] != 256:
        raise ValueError(f"Invalid columns: {file.name}")

    if not np.isfinite(data).all():
        raise ValueError(f"Invalid CSI data: {file.name}")

    if len(data) < WINDOW_SIZE:
        raise ValueError(f"Capture too short: {file.name}")

    start = (len(data) - WINDOW_SIZE) // 2
    window = data[start:start + WINDOW_SIZE]

    output_folder = OUTPUT_DIR / record["split"] / "room_B"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / (file.stem + ".npy")

    np.save(output_file, window)
    processed += 1

print("Windows created:", processed)
print("Window shape:", window.shape)
print("Data type:", window.dtype)
