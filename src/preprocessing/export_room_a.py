
from pathlib import Path
import numpy as np
import pandas as pd
from scapy.all import PcapReader, UDP

from csi_decode import decode_csi, calculate_amplitude

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data/raw/Experiment-2"
OUTPUT_DIR = ROOT / "data/candidate_csv/room_A"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pcap_files = sorted(INPUT_DIR.rglob("*.pcap"))

columns = [
    f"subcarrier_{i:03d}"
    for i in range(1, 257)
]

converted = 0
failed = []

for file in pcap_files:
    try:
        amplitude_rows = []

        with PcapReader(str(file)) as reader:
            for packet in reader:
                if UDP not in packet:
                    continue

                payload = bytes(packet[UDP].payload)

                if len(payload) != 1042:
                    continue

                if payload[:2] != b"\x11\x11":
                    continue

                words = np.frombuffer(
                    payload[18:], dtype="<u4"
                )

                csi = decode_csi(words)
                amplitude = calculate_amplitude(csi)

                amplitude_rows.append(amplitude)

        if not amplitude_rows:
            raise ValueError("No valid CSI observations")

        matrix = np.stack(amplitude_rows)

        # Preserve source structure and capture identity
        relative = file.relative_to(INPUT_DIR)
        destination = (
            OUTPUT_DIR / relative
        ).with_suffix(".csv")

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        pd.DataFrame(
            matrix, columns=columns
        ).to_csv(destination, index=False)

        converted += 1

    except Exception as error:
        failed.append((str(file), str(error)))

print("Total PCAP files:", len(pcap_files))
print("Converted:", converted)
print("Failed:", len(failed))
print("First failures:", failed[:5])
print("Candidate output:", OUTPUT_DIR)
