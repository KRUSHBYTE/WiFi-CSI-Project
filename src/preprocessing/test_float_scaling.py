
from pathlib import Path
import struct
import numpy as np
from scapy.all import PcapReader, UDP

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/Experiment-2"

pcap_file = next(DATA_DIR.rglob("fall10.pcap"))

with PcapReader(str(pcap_file)) as reader:
    packet = next(iter(reader))

payload = bytes(packet[UDP].payload)

if len(payload) != 1042:
    raise ValueError("Unexpected payload length")

words = struct.unpack("<256I", payload[18:])

real_values = []
imag_values = []

for word in words:
    real_mag = (word >> 18) & 0x7FF
    imag_mag = (word >> 6) & 0x7FF

    real_sign = (word >> 29) & 1
    imag_sign = (word >> 17) & 1

    exp = word & 0x3F
    if exp >= 32:
        exp -= 64

    real = -real_mag if real_sign else real_mag
    imag = -imag_mag if imag_sign else imag_mag

    # Diagnostic floating-point interpretation
    real_values.append(np.ldexp(float(real), exp))
    imag_values.append(np.ldexp(float(imag), exp))

csi = np.array(real_values) + 1j * np.array(imag_values)
amplitude = np.abs(csi)

print("CSI shape:", csi.shape)

print("\nFirst five diagnostic CSI values:")
print(csi[:5])

print("\nAmplitude statistics:")
print("Minimum:", amplitude.min())
print("Maximum:", amplitude.max())
print("Mean:", amplitude.mean())
print("Finite:", np.isfinite(amplitude).all())
