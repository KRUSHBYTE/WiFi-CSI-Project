
from pathlib import Path
from collections import Counter
import struct
from scapy.all import PcapReader, UDP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/raw/Experiment-2"

pcap_file = sorted(DATA_DIR.rglob("*.pcap"))[0]

with PcapReader(str(pcap_file)) as reader:
    packet = next(iter(reader))

payload = bytes(packet[UDP].payload)

header_size = 18
csi_bytes = payload[header_size:]

if len(csi_bytes) != 1024:
    raise ValueError("Unexpected CSI data length")

# Interpret each group of 4 bytes as one unsigned word
words = struct.unpack("<256I", csi_bytes)

print("Selected PCAP:", pcap_file.name)
print("Number of CSI words:", len(words))
print("Bits per word:", 32)

print("\nFirst 10 packed CSI words:")
for i, word in enumerate(words[:10]):
    print(f"{i}: {word:032b}")

print("\nCSI word statistics:")
print("Zero words:", sum(word == 0 for word in words))
print("Unique words:", len(set(words)))
print("Minimum raw word:", hex(min(words)))
print("Maximum raw word:", hex(max(words)))

# Inspect the six most significant bits
upper_bits = Counter(word >> 26 for word in words)

print("\nMost common upper-six-bit patterns:")
for value, count in upper_bits.most_common(10):
    print(f"{value:06b}: {count}")
