
from pathlib import Path
from scapy.all import PcapReader, UDP
import struct

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/raw/Experiment-2"

pcap_file = sorted(DATA_DIR.rglob("*.pcap"))[0]

with PcapReader(str(pcap_file)) as reader:
    packet = next(iter(reader))

if UDP not in packet:
    raise ValueError("First packet is not UDP")

payload = bytes(packet[UDP].payload)

if len(payload) != 1042:
    raise ValueError("Unexpected payload length")

header = payload[:18]
csi_data = payload[18:]

print("File:", pcap_file.name)
print("Header length:", len(header))
print("CSI data length:", len(csi_data))

print("\nHeader fields:")
print("Magic:", header[0:2].hex())
print("RSSI byte:", header[2])
print("Frame control byte:", header[3])
print("Source MAC:", header[4:10].hex(":"))

seq = struct.unpack_from("<H", header, 10)[0]
core_stream = struct.unpack_from("<H", header, 12)[0]
chanspec = struct.unpack_from("<H", header, 14)[0]
chip = struct.unpack_from("<H", header, 16)[0]

print("Sequence:", seq)
print("Core/stream:", core_stream)
print("Chanspec:", hex(chanspec))
print("Chip version:", hex(chip))

print("\nFirst 8 packed CSI words:")
words = struct.unpack_from("<8I", csi_data)

for index, word in enumerate(words):
    print(f"Subcarrier {index}: 0x{word:08x}")
