
from pathlib import Path
from scapy.all import PcapReader, UDP
import numpy as np

from csi_decode import decode_csi_candidate, calculate_amplitude

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/Experiment-2"

activities = [
    "fall",
    "sit",
    "sitdown",
    "stand",
    "standup",
    "walk"
]

for activity in activities:

    files = sorted(
        p for p in DATA_DIR.rglob("*.pcap")
        if p.parent.name.lower() == activity
        and "validation" not in p.parts
    )

    if not files:
        print(f"{activity}: No files found")
        continue

    file = files[0]
    amplitudes = []
    rejected = 0

    with PcapReader(str(file)) as reader:
        for packet in reader:

            if UDP not in packet:
                rejected += 1
                continue

            payload = bytes(packet[UDP].payload)

            if len(payload) != 1042 or payload[:2] != b"\x11\x11":
                rejected += 1
                continue

            words = np.frombuffer(payload[18:], dtype="<u4")

            csi = decode_csi_candidate(words)
            amplitude = calculate_amplitude(csi)

            amplitudes.append(amplitude)

    if not amplitudes:
        print(f"{activity}: No valid CSI observations")
        continue

    matrix = np.stack(amplitudes)

    print(f"\nActivity: {activity}")
    print("File:", file.name)
    print("Shape:", matrix.shape)
    print("Rejected packets:", rejected)
    print("Minimum:", matrix.min())
    print("Maximum:", matrix.max())
    print("Mean:", matrix.mean())
    print("Finite values:", np.isfinite(matrix).all())
