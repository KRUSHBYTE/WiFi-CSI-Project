

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data/raw/Experiment-3/Data"
OUTPUT_DIR = ROOT / "data/intermediate_csv/room_B"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.csv"))
converted = 0
failed = []

columns = [f"subcarrier_{i:03d}" for i in range(1, 257)]

for file in files:
    try:
        data = pd.read_csv(file, header=0)

        if data.shape[1] != 256:
            raise ValueError("Expected 256 columns")

        values = data.to_numpy(dtype=np.float32)

        if values.shape[0] == 0:
            raise ValueError("Empty capture")

        if not np.isfinite(values).all():
            raise ValueError("Invalid amplitude values")

        output = pd.DataFrame(values, columns=columns)

        output.to_csv(
            OUTPUT_DIR / file.name,
            index=False
        )

        converted += 1

    except Exception as error:
        failed.append((file.name, str(error)))

print("Total files:", len(files))
print("Converted:", converted)
print("Failed:", len(failed))
print("Example failures:", failed[:5])
