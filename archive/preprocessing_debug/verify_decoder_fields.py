
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

assert len(payload) == 1042
assert payload[:2] == b"\x11\x11"

# Read 256 unsigned 32-bit packed CSI values
words = np.array(
    struct.unpack("<256I", payload[18:]),
    dtype=np.uint32
)

# Extract encoded fields
real_mag = (words >> 18) & 0x7FF
imag_mag = (words >> 6) & 0x7FF

real_sign = (words >> 29) & 1
imag_sign = (words >> 17) & 1

exp_raw = words & 0x3F

# Convert 6-bit exponent to signed integer
exponent = exp_raw.astype(np.int32)
exponent[exponent >= 32] -= 64

# Apply component signs
real = real_mag.astype(np.int32)
imag = imag_mag.astype(np.int32)

real[real_sign == 1] *= -1
imag[imag_sign == 1] *= -1

print("CSI words:", len(words))
print("Real magnitude range:", real_mag.min(), real_mag.max())
print("Imag magnitude range:", imag_mag.min(), imag_mag.max())
print("Signed exponent range:", exponent.min(), exponent.max())

print("\nFirst five extracted values:")

for i in range(5):
    print(
        f"{i}: real={real[i]}, "
        f"imag={imag[i]}, "
        f"exponent={exponent[i]}"
    )
