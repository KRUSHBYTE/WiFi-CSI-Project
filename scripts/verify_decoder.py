"""Check that the vectorized decoder equals the scalar reference decoder bit-for-bit.

Uses random packed words (including zero, negative-exponent and saturated cases)
and, if available, real packets from data/raw/Experiment-2.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from csi_decode import decode_csi  # noqa: E402  (Member 1 scalar reference)
from pipeline.ingest import decode_csi_batch  # noqa: E402


def random_words(rng, n):
    words = rng.integers(0, 2 ** 32, size=(n, 256), dtype=np.uint64).astype(np.uint32)
    words[0] = 0                                    # empty packet
    words[1, :128] = 0                              # half-empty packet
    words[2] = (words[2] & ~np.uint32(0x3F)) | np.uint32(0x20)   # exponent -32
    words[3] = (words[3] & ~np.uint32(0x3F)) | np.uint32(0x1F)   # exponent +31
    return words


def main():
    rng = np.random.default_rng(0)
    words = random_words(rng, 400)
    reference = np.stack([decode_csi(w) for w in words])
    fast = decode_csi_batch(words)
    assert np.array_equal(reference, fast), "vectorized decoder differs from reference"
    print(f"random words: {len(words)} packets identical")

    raw = ROOT / "data/raw/Experiment-2"
    pcaps = sorted(raw.rglob("*.pcap"))[:3]
    if pcaps:
        from preprocessing.pcap_reader import iter_udp_payloads
        rows = []
        for pcap in pcaps:
            for packet in iter_udp_payloads(pcap):
                if len(packet.payload) == 1042 and packet.payload[:2] == b"\x11\x11":
                    rows.append(np.frombuffer(packet.payload, "<u4", offset=18, count=256))
        rows = np.stack(rows)
        assert np.array_equal(np.stack([decode_csi(w) for w in rows]), decode_csi_batch(rows))
        print(f"real packets from {len(pcaps)} captures: {len(rows)} packets identical")
    print("OK")


if __name__ == "__main__":
    main()
