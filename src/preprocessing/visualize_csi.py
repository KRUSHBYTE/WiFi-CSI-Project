
from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt
from scapy.all import PcapReader, UDP

from csi_decode import decode_csi_provisional

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/Experiment-2"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

pcap_file = next(DATA_DIR.rglob("fall10.pcap"))

amplitude_rows = []

with PcapReader(str(pcap_file)) as reader:
    for packet in reader:
        if UDP not in packet:
            continue

        payload = bytes(packet[UDP].payload)

        if len(payload) != 1042:
            continue

        if payload[:2] != b"\x11\x11":
            continue

        words = struct.unpack("<256I", payload[18:])
        csi = decode_csi_provisional(words)
        amplitude = np.abs(csi)

        if np.isfinite(amplitude).all():
            amplitude_rows.append(amplitude)

if not amplitude_rows:
    raise ValueError("No suitable CSI packets found")

matrix = np.stack(amplitude_rows)

print("Amplitude matrix shape:", matrix.shape)
print("Minimum:", matrix.min())
print("Maximum:", matrix.max())
print("Mean:", matrix.mean())

# Plot the amplitude matrix
plt.figure(figsize=(12, 5))
plt.imshow(matrix, aspect="auto", origin="lower")
plt.colorbar(label="Provisional amplitude")
plt.xlabel("Subcarrier index")
plt.ylabel("CSI observation index")
plt.title("Fall10 - Provisional CSI Amplitude")
plt.tight_layout()

output_file = OUTPUT_DIR / "fall10_provisional.png"
plt.savefig(output_file, dpi=150)
plt.show()

print("Plot saved:", output_file)
