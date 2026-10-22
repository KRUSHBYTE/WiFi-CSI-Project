
from pathlib import Path
import numpy as np
import csiread

from scapy.all import PcapReader, UDP
from csi_decode import decode_csi

ROOT = Path(__file__).resolve().parents[2]
file = next((ROOT / "data/raw/Experiment-2").rglob("fall10.pcap"))

# Independent reference decoder
reference = csiread.Nexmon(str(file), chip="4366c0", bw=80)
reference.read()

print("Reference CSI shape:", reference.csi.shape)


our_packets = []

with PcapReader(str(file)) as reader:
    for packet in reader:
        if UDP not in packet:
            continue

        payload = bytes(packet[UDP].payload)

        if len(payload) != 1042 or payload[:2] != b"\x11\x11":
            continue

        words = np.frombuffer(payload[18:], dtype="<u4")
        our_packets.append(decode_csi(words))

ours = np.stack(our_packets)
refs = reference.csi

if ours.shape != refs.shape:
    raise ValueError(
        f"Shape mismatch: {ours.shape} versus {refs.shape}"
    )

for i in [0, 1, 2, 10, 50, 100, 200, 239]:
    a = np.abs(ours[i]).astype(np.float64)
    b = np.abs(refs[i]).astype(np.float64)

    valid = np.isfinite(a) & np.isfinite(b) & (b > 0)
    a, b = a[valid], b[valid]

    scale = np.dot(a, b) / np.dot(b, b)
    error = np.linalg.norm(a - scale * b) / np.linalg.norm(a)
    correlation = np.corrcoef(a, b)[0, 1]

    print(
        f"Packet {i}: "
        f"scale={scale:.8f}, "
        f"error={error:.6f}, "
        f"correlation={correlation:.6f}"
    )