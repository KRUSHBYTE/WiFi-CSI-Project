
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Experiment-3" / "Data"

csv_files = sorted(DATA_DIR.glob("*.csv"))

valid_files = 0
invalid_files = []
row_counts = []

print("Total CSV files:", len(csv_files))

for file in csv_files:
    try:
        # Read first row as column headings
        data = pd.read_csv(file, header=0)

        # Each observation must have 256 subcarriers
        if data.shape[1] != 256:
            invalid_files.append((file.name, "Incorrect column count"))
            continue

        # Convert the amplitude values to numbers
        values = data.apply(pd.to_numeric, errors="coerce").to_numpy()

        # Detect missing or non-finite measurements
        if not np.isfinite(values).all():
            invalid_files.append((file.name, "Invalid amplitude values"))
            continue

        if values.shape[0] == 0:
            invalid_files.append((file.name, "Empty recording"))
            continue

        valid_files += 1
        row_counts.append(values.shape[0])

    except Exception as error:
        invalid_files.append((file.name, str(error)))

print("\n--- Validation Results ---")
print("Valid files:", valid_files)
print("Invalid files:", len(invalid_files))

if row_counts:
    print("Minimum rows:", min(row_counts))
    print("Maximum rows:", max(row_counts))
    print("Average rows:", round(np.mean(row_counts), 2))

print("\nFirst 10 invalid files:")
for item in invalid_files[:10]:
    print(item)
