
from pathlib import Path
from collections import Counter
from scapy.all import PcapReader, UDP

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Experiment-2"

pcap_files = sorted(DATA_DIR.rglob("*.pcap"))

if not pcap_files:
    raise FileNotFoundError("No PCAP recordings found")

file = pcap_files[0]

print("Selected file:", file.name)

packet_count = 0
udp_count = 0
valid_structure = 0

payload_lengths = Counter()
magic_values = Counter()

with PcapReader(str(file)) as reader:

    for packet in reader:
        packet_count += 1

        if UDP not in packet:
            continue

        udp_count += 1

        payload = bytes(packet[UDP].payload)

        payload_lengths[len(payload)] += 1

        if len(payload) >= 4:
            magic = payload[:4].hex()
            magic_values[magic] += 1

        if (
            len(payload) == 1042
            and payload[:4] == bytes.fromhex("11111111")
        ):
            valid_structure += 1

print("\n--- PCAP Structure Report ---")
print("Total packets:", packet_count)
print("UDP packets:", udp_count)
print("Expected-format packets:", valid_structure)

print("\nPayload lengths:")
for length, count in payload_lengths.items():
    print(length, "bytes:", count, "packets")

print("\nFirst-four-byte patterns:")
for magic, count in magic_values.items():
    print(magic, ":", count)
