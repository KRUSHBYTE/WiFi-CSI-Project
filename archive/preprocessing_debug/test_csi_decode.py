
from pathlib import Path
import struct
import numpy as np
from scapy.all import PcapReader, UDP

from csi_decode import decode_csi_provisional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/raw/Experiment-2"

# Use the recording inspected previously
pcap_files = list(DATA_DIR.rglob("fall10.pcap"))

if not pcap_files:
    raise FileNotFoundError("fall10.pcap not found")

pcap_file = pcap_files[0]

# Read the first packet
with PcapReader(str(pcap_file)) as reader:
    packet = next(iter(reader))

if UDP not in packet:
    raise ValueError("Packet does not contain UDP")

payload = bytes(packet[UDP].payload)

if len(payload) != 1042:
    raise ValueError("Unexpected payload length")

# Extract 256 packed CSI words
words = struct.unpack("<256I", payload[18:])

# Apply our provisional reconstruction
csi = decode_csi_provisional(words)

# Calculate amplitude
amplitude = np.abs(csi)

print("File:", pcap_file.name)
print("CSI shape:", csi.shape)
print("Amplitude shape:", amplitude.shape)

print("\nFirst five provisional complex values:")
print(csi[:5])

print("\nFirst five provisional amplitudes:")
print(amplitude[:5])

print("\nAmplitude statistics:")
print("Minimum:", amplitude.min())
print("Maximum:", amplitude.max())
print("Mean:", amplitude.mean())
print("Finite values:", np.isfinite(amplitude).all())
print("Zero amplitudes:", np.count_nonzero(amplitude == 0))
