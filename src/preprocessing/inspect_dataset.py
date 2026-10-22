
from pathlib import Path
from collections import Counter

# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of the datasets
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# Examine both experiments
for experiment in ["Experiment-2", "Experiment-3"]:

    folder = DATA_ROOT / experiment

    print(f"\nDataset: {experiment}")

    if not folder.exists():
        print("ERROR: Folder not found")
        continue

    # Find all files inside the experiment
    files = [p for p in folder.rglob("*") if p.is_file()]

    # Count files by extension
    extensions = Counter(p.suffix.lower() for p in files)

    print("Total files:", len(files))
    print("File types:", dict(extensions))

    # Display first five filenames
    print("Example files:")

    for file in sorted(files)[:5]:
        print(file.relative_to(folder))
