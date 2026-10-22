
from pathlib import Path
from collections import Counter
import struct

from scapy.all import PcapReader, UDP

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/Experiment-2"

pcap_file = sorted(DATA_DIR.rglob("*.pcap"))[0]

with PcapReader(str(pcap_file)) as reader:
    packet = next(iter(reader))

payload = bytes(packet[UDP].payload)

if len(payload) != 1042:
    raise ValueError("Unexpected payload length")

words = struct.unpack("<256I", payload[18:])

real_signs = Counter()
imag_signs = Counter()
exponents = Counter()

for word in words:
    real_sign = (word >> 31) & 1
    imag_sign = (word >> 18) & 1
    exponent = word & 0x3F

    real_signs[real_sign] += 1
    imag_signs[imag_sign] += 1
    exponents[exponent] += 1

print("File:", pcap_file.name)
print("Total CSI words:", len(words))

print("\nReal sign distribution:", dict(real_signs))
print("Imaginary sign distribution:", dict(imag_signs))

print("\nExponent field:")
print("Minimum:", min(exponents))
print("Maximum:", max(exponents))
print("Unique exponent values:", len(exponents))

print("\nMost common exponents:")
for exponent, count in exponents.most_common(10):
    print(exponent, ":", count)
