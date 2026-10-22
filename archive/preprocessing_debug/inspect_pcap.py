
from pathlib import Path
from scapy.all import PcapReader, UDP

# Locate project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate Experiment-2
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Experiment-2"

# Find all PCAP files
pcap_files = sorted(DATA_DIR.rglob("*.pcap"))

print("Total PCAP files:", len(pcap_files))

if not pcap_files:
    raise FileNotFoundError("No PCAP files found")

# Select first recording
file = pcap_files[0]
print("Selected file:", file.relative_to(DATA_DIR))

packet_count = 0

# Read packets one by one
with PcapReader(str(file)) as reader:
    for packet in reader:
        packet_count += 1

        # Inspect only first three packets
        if packet_count <= 3:
            print(f"\n--- Packet {packet_count} ---")
            print("Timestamp:", packet.time)
            print("Packet length:", len(packet))
            print("Layers:", packet.summary())

            if UDP in packet:
                payload = bytes(packet[UDP].payload)
                print("UDP payload length:", len(payload))
                print("First 32 payload bytes:", payload[:32].hex())

print("\nTotal packets:", packet_count)
