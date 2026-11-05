"""Generate a small SYNTHETIC dataset in the exact on-disk formats of the challenge.

Purpose: an end-to-end smoke test of the whole pipeline (PCAP parsing, packed-CSI
decoding, CSV ingestion, QC, splits, windows, Tasks 1-3, run_episode.py) when the
real IEEE DataPort archive is not available. Results on this data mean nothing
scientifically; they only prove the code runs.

    python scripts/make_synthetic_dataset.py --out /tmp/synth_ws

writes <out>/data/raw/Experiment-2/..., <out>/data/raw/Experiment-3/Data/...,
<out>/episode/ (a mixed unseen episode) and <out>/experiment_synthetic.yaml.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.pcap_reader import build_udp_frame, write_pcap  # noqa: E402

A_ACTS = ["fall", "sit", "sitdown", "stand", "standup", "walk"]
B_ACTS = {"Empty": "EMPTY", "Lying": "LYING", "Sitting": "SIT", "Standing": "STAND", "Walking": "WALK"}
CANON = {"fall": "FALL", "sit": "SIT", "sitdown": "SIT-DOWN", "stand": "STAND",
         "standup": "STAND-UP", "walk": "WALK"}
RATE = 33.0  # observations per second (illustrative)


def smooth(rng, scale=1.0, terms=4):
    s = np.arange(256) / 256.0
    out = np.zeros(256)
    for _ in range(terms):
        out += rng.normal() * np.cos(2 * np.pi * (rng.uniform(0.5, 4) * s + rng.uniform()))
    return scale * out / terms


class World:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.posture = {a: smooth(rng, 0.25) for a in
                        ["EMPTY", "SIT", "STAND", "LYING", "WALK", "FALL", "SIT-DOWN", "STAND-UP"]}
        self.profile = {"A": 1.0 + 0.5 * np.abs(smooth(rng, 2.0)) + 0.2,
                        "B": 1.0 + 0.5 * np.abs(smooth(rng, 2.0)) + 0.2}
        self.room_mix = {r: {a: smooth(rng, 0.25) for a in self.posture} for r in "AB"}
        self.walk_phase = {r: rng.uniform(0, 2 * np.pi, 256) for r in "AB"}

    def capture(self, rng, room, act, n):
        t = np.arange(n) / RATE
        posture = 0.75 * self.posture[act] + 0.25 * self.room_mix[room][act] + smooth(rng, 0.03)
        m = np.tile(posture, (n, 1))
        noise = 0.02 if act != "EMPTY" else 0.008
        if act in ("SIT", "STAND", "LYING"):
            m += 0.02 * np.sin(2 * np.pi * rng.uniform(0.2, 0.4) * t)[:, None]
        if act == "WALK":
            f = rng.uniform(1.0, 2.0)
            m += 0.35 * np.sin(2 * np.pi * f * t[:, None] + self.walk_phase[room][None, :])
        if act in ("FALL", "SIT-DOWN", "STAND-UP"):
            centre = rng.uniform(0.35, 0.65) * n
            width = {"FALL": 6.0, "SIT-DOWN": 14.0, "STAND-UP": 14.0}[act]
            burst = np.exp(-0.5 * ((np.arange(n) - centre) / width) ** 2)
            amp = {"FALL": 0.8, "SIT-DOWN": 0.35, "STAND-UP": 0.35}[act]
            m += amp * burst[:, None] * np.sin(2 * np.pi * 3 * t[:, None] + self.walk_phase[room][None, :])
            if act == "STAND-UP":
                m += 0.1 * (np.arange(n) > centre)[:, None]
        m += noise * rng.normal(size=m.shape)
        gain = rng.uniform(0.9, 1.1)
        return np.maximum(gain * self.profile[room][None, :] * (1.0 + m), 1e-3)


def encode_words(amplitude, rng, scale=400.0):
    """Pack amplitudes into BCM4366c0-style words (inverse of the decoder)."""
    phase = rng.uniform(0, 2 * np.pi, amplitude.shape)
    re = amplitude * scale * np.cos(phase)
    im = amplitude * scale * np.sin(phase)
    m = np.maximum(np.abs(re), np.abs(im))
    exp = np.where(m > 0, np.ceil(np.log2(np.maximum(m, 1e-12) / 2047.0)), 0).astype(np.int64)
    exp = np.clip(exp, -32, 31)
    rm = np.minimum(np.round(np.abs(re) / 2.0 ** exp), 2047).astype(np.uint32)
    imm = np.minimum(np.round(np.abs(im) / 2.0 ** exp), 2047).astype(np.uint32)
    words = ((re < 0).astype(np.uint32) << 29) | (rm << 18) | ((im < 0).astype(np.uint32) << 17) \
        | (imm << 6) | (exp.astype(np.int64) & 0x3F).astype(np.uint32)
    return words.astype("<u4")


def nexmon_payload(words, seq):
    header = b"\x11\x11" + bytes([200, 0x08]) + b"\x00\x11\x22\x33\x44\x55" + \
        struct.pack("<HHHH", seq & 0xFFFF, 0, 0xE02A, 0x4366)
    return header + words.tobytes()


def write_room_a_capture(world, rng, path, act, n_packets):
    amplitude = world.capture(rng, "A", CANON[act], n_packets)
    frames = []
    for i, row in enumerate(amplitude):
        frames.append(build_udp_frame(nexmon_payload(encode_words(row[None, :], rng)[0], i)))
        if rng.uniform() < 0.01:                        # non-CSI UDP packet -> rejected
            frames.append(build_udp_frame(b"\x00" * 100))
        if rng.uniform() < 0.005:                       # zero CSI -> QC zero row
            frames.append(build_udp_frame(nexmon_payload(np.zeros(256, "<u4"), i)))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pcap(path, frames)


def write_room_b_capture(world, rng, path, act, rows, header_style):
    amplitude = world.capture(rng, "B", act, rows) * 25.0
    header = [str(i) for i in range(256)] if header_style == 0 else [f"Subcarrier_{i}" for i in range(1, 257)]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, amplitude, delimiter=",", fmt="%.3f", header=",".join(header), comments="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--a-dev", type=int, default=24, help="Room A development captures per class")
    parser.add_argument("--a-val", type=int, default=25)
    parser.add_argument("--b-per-class", type=int, default=45)
    parser.add_argument("--b-rows", type=int, default=300)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    rng = np.random.default_rng(123)
    world = World()

    exp2 = out / "data/raw/Experiment-2/realdata/input_data/30ms/csidata"
    shorts = {("sit", "development", 0), ("sit", "development", 1), ("walk", "validation", 0)}
    for act in A_ACTS:
        for split, count, folder in (("development", args.a_dev, exp2 / act),
                                     ("validation", args.a_val, exp2 / "validation" / act)):
            for i in range(count):
                n = int(rng.integers(120, 140)) if (act, split, i) in shorts else int(rng.integers(170, 260))
                number = i + 1 if split == "development" else 500 + i
                write_room_a_capture(world, rng, folder / f"{act}{number}.pcap", act, n)
    print("Room A captures written:", sum(1 for _ in (out / "data/raw/Experiment-2").rglob("*.pcap")))

    exp3 = out / "data/raw/Experiment-3/Data"
    for prefix, act in B_ACTS.items():
        for i in range(1, args.b_per_class + 1):
            name = f"{prefix}_{i}.csv" if (prefix == "Walking" or i % 9 == 0) else f"{prefix}_Amp_{i}.csv"
            write_room_b_capture(world, rng, exp3 / name, act, args.b_rows, i % 2)
    print("Room B captures written:", sum(1 for _ in exp3.glob("*.csv")))

    # A small "unseen" episode mixing both formats, as the teaching team might supply.
    episode = out / "episode"
    erng = np.random.default_rng(999)
    for j, act in enumerate(A_ACTS):
        write_room_a_capture(world, erng, episode / f"capture_{j:02d}.pcap", act, 200)
    for j, (prefix, act) in enumerate(B_ACTS.items()):
        write_room_b_capture(world, erng, episode / f"capture_{10 + j:02d}.csv", act, 300, j % 2)
    truth = [(f"capture_{j:02d}.pcap", "A", CANON[a]) for j, a in enumerate(A_ACTS)] + \
            [(f"capture_{10 + j:02d}.csv", "B", a) for j, a in enumerate(B_ACTS.values())]
    with open(out / "episode_truth.csv", "w") as handle:
        handle.write("window_id,room,label\n")
        for row in truth:
            handle.write(",".join(row) + "\n")

    total_a = len(A_ACTS) * (args.a_dev + args.a_val)
    config = f"""# Synthetic smoke-test configuration (generated)
inherit: {ROOT / 'configs' / 'experiment.yaml'}
paths:
  raw_experiment2: {out}/data/raw/Experiment-2
  raw_experiment3: {out}/data/raw/Experiment-3/Data
  intermediate_csv: {out}/data/intermediate_csv
  windows: {out}/data/windows
  normalization_dir: {out}/configs/normalization
  split_a: {out}/configs/room_a_split.csv
  split_b: {out}/configs/room_b_split.csv
  task2_room_a_root: {out}/data/strict_normalized
  task2_room_b_root: {out}/data/normalized
  results: {out}/results
  models: {out}/models
expected_counts:
  enforce: true
  room_a_total: {total_a}
  room_a_validation: {len(A_ACTS) * args.a_val}
  room_b_total: {len(B_ACTS) * args.b_per_class}
  room_b_validation_per_class: 25
task3:
  seeds: [0, 1, 2]
"""
    (out / "experiment_synthetic.yaml").write_text(config)
    print("Config:", out / "experiment_synthetic.yaml")


if __name__ == "__main__":
    main()
