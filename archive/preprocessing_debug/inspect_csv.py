
from pathlib import Path
import pandas as pd

# Locate the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate Experiment-3 CSV files
DATA_DIR = PROJECT_ROOT / "data/raw/Experiment-3/Data"

# Find all CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

print("Total CSV files:", len(csv_files))

if not csv_files:
    raise FileNotFoundError("No CSV files found")

# Select the first CSV file
file = csv_files[0]

print("\nSelected file:", file.name)

# Read the CSV without assuming a header
data = pd.read_csv(file, header=0)

print("\nShape:", data.shape)

print("\nFirst five rows:")
print(data.head())

print("\nData types:")
print(data.dtypes.value_counts())

print("\nMissing values:", data.isna().sum().sum())
