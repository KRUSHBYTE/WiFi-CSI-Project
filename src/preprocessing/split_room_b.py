
from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/Experiment-3/Data"
OUTPUT = ROOT / "configs/room_b_split.csv"

activities = {
    "EMPTY": "Empty",
    "LYING": "Lying",
    "SIT": "Sitting",
    "STAND": "Standing",
    "WALK": "Walking",
}

rng = random.Random(42)
records = []

for label, prefix in activities.items():

    files = sorted(
        p for p in DATA_DIR.glob("*.csv")
        if p.name.startswith(prefix + "_")
    )

    if len(files) < 25:
        raise ValueError(f"Insufficient captures for {label}")

    selected = set(rng.sample(files, 25))

    for file in files:
        records.append({
            "capture_id": file.name,
            "room": "B",
            "activity": label,
            "split": (
                "validation" if file in selected
                else "development"
            ),
        })

if len(records) != 1201:
    raise ValueError("Unexpected total capture count")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["capture_id", "room", "activity", "split"]
    )
    writer.writeheader()
    writer.writerows(records)

print("Total:", len(records))
print("Development:", sum(
    r["split"] == "development" for r in records
))
print("Validation:", sum(
    r["split"] == "validation" for r in records
))
print("Saved:", OUTPUT)
