
from pathlib import Path
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/Experiment-2"
OUTPUT = ROOT / "configs/room_a_split.csv"

activities = {
    "fall": "FALL",
    "sit": "SIT",
    "sitdown": "SIT-DOWN",
    "stand": "STAND",
    "standup": "STAND-UP",
    "walk": "WALK",
}

records = []

for file in sorted(DATA_DIR.rglob("*.pcap")):
    relative = file.relative_to(DATA_DIR)
    parts = [part.lower() for part in relative.parts]

    # Preserve the supplied validation split
    split = (
        "validation"
        if "validation" in parts
        else "development"
    )

    activity = file.parent.name.lower()

    if activity not in activities:
        raise ValueError(
            f"Unknown activity folder: {file}"
        )

    records.append({
        "capture_id": relative.as_posix(),
        "room": "A",
        "activity": activities[activity],
        "split": split,
    })

counts = Counter(r["split"] for r in records)

print("Total:", len(records))
print("Development:", counts["development"])
print("Validation:", counts["validation"])

if (
    len(records) != 1084
    or counts["development"] != 934
    or counts["validation"] != 150
):
    raise ValueError("Room A split counts do not match PDF")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "capture_id",
            "room",
            "activity",
            "split"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

print("Saved:", OUTPUT)
