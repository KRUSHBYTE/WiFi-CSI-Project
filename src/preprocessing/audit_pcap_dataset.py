
from pathlib import Path
import csv

from scapy.all import PcapReader, UDP

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/Experiment-2"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

pcap_files = sorted(DATA_DIR.rglob("*.pcap"))

records = []

for index, file in enumerate(pcap_files, start=1):

    total = 0
    valid = 0

    with PcapReader(str(file)) as reader:
        for packet in reader:
            total += 1

            if UDP not in packet:
                continue

            payload = bytes(packet[UDP].payload)

            if (
                len(payload) == 1042
                and payload[:2] == b"\x11\x11"
            ):
                valid += 1

    rejected = total - valid

    activity = file.parent.name.lower()
    split = (
        "validation"
        if "validation" in file.relative_to(DATA_DIR).parts
        else "development"
    )

    records.append({
        "capture_id": str(file.relative_to(DATA_DIR)),
        "activity": activity,
        "split": split,
        "total_packets": total,
        "valid_packets": valid,
        "rejected_packets": rejected,
        "supports_150_window": valid >= 150,
    })

    if index % 100 == 0:
        print(f"Inspected {index}/{len(pcap_files)} files")

output_file = OUTPUT_DIR / "pcap_audit.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print("\n--- Dataset Audit ---")
print("Total captures:", len(records))
print("Total packets:", sum(r["total_packets"] for r in records))
print("Valid-format packets:", sum(r["valid_packets"] for r in records))
print("Rejected packets:", sum(r["rejected_packets"] for r in records))

print(
    "Captures supporting a 150-row window:",
    sum(r["supports_150_window"] for r in records)
)

print(
    "Captures shorter than 150 observations:",
    sum(not r["supports_150_window"] for r in records)
)

print("Audit saved:", output_file)

