
from pathlib import Path

import numpy as np
from scapy.all import PcapReader, UDP

from csi_decode import decode_csi, calculate_amplitude

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "Experiment-2"

files = list(DATA_DIR.rglob("fall10.pcap"))

if not files:
    raise FileNotFoundError("fall10.pcap not found")

pcap_file = files[0]

amplitude_rows = []
rejected_packets = 0

with PcapReader(str(pcap_file)) as reader:
    for packet in reader:

        if UDP not in packet:
            rejected_packets += 1
            continue

        payload = bytes(packet[UDP].payload)

        if len(payload) != 1042:
            rejected_packets += 1
            continue

        if payload[:2] != b"\x11\x11":
            rejected_packets += 1
            continue

        # 18-byte header followed by 256 little-endian words
        words = np.frombuffer(
            payload[18:], dtype="<u4"
        )

        csi = decode_csi(words)
        amplitudes = calculate_amplitude(csi)

        amplitude_rows.append(amplitudes)

if not amplitude_rows:
    raise ValueError("No structurally valid CSI observations")

matrix = np.stack(amplitude_rows)

print("File:", pcap_file.name)
print("Amplitude matrix:", matrix.shape)
print("Data type:", matrix.dtype)
print("Rejected packets:", rejected_packets)
print("Minimum amplitude:", matrix.min())
print("Maximum amplitude:", matrix.max())
print("Mean amplitude:", matrix.mean())
print("Finite values:", np.isfinite(matrix).all())

# Save only as an experimental diagnostic
output_dir = ROOT / "results"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "fall10_candidate.npy"

np.save(output_path, matrix)

print("Candidate saved:", output_path)
