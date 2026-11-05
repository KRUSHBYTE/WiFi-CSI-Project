"""One-command preprocessing: raw source captures -> provenance-tracked 150x256 windows.

    python src/preprocessing/build_dataset.py [--config configs/experiment.yaml]

Stages (all driven by configs/experiment.yaml):
  1. capture-level split manifests (Room A: supplied validation/ folder;
     Room B: sorted filenames, 25 per class, seed 42 - created once, then reused)
  2. ingestion to the common 256-column amplitude form (PCAP decode / CSV read)
  3. quality control (non-finite, zero-information and duplicate observations)
  4. deterministic centre window of 150 valid observations (short captures excluded)
  5. normalization statistics fitted on DEVELOPMENT windows only:
       A      - Room A development windows
       B      - Room B development windows
       pooled - development windows of both rooms (room-agnostic)
  6. room-normalized copies in the layout Task 2 (src/task2) expects

Outputs: data/windows/manifest.csv (one row per capture, including excluded ones,
with the preprocessing-config hash) and results/preprocessing/summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import ROOM_A_FOLDERS, ROOM_B_PREFIXES, load_config  # noqa: E402
from pipeline.ingest import (  # noqa: E402
    COLUMNS, Normalizer, quality_control, read_room_a_capture, read_room_b_capture, select_window,
)


# ---------------------------------------------------------------------------
# 1. Split manifests
# ---------------------------------------------------------------------------

def make_split_a(raw_dir: Path) -> pd.DataFrame:
    records = []
    for file in sorted(raw_dir.rglob("*.pcap")):
        relative = file.relative_to(raw_dir)
        folder = file.parent.name.lower()
        if folder not in ROOM_A_FOLDERS:
            raise ValueError(f"Unknown activity folder: {file}")
        split = "validation" if "validation" in [p.lower() for p in relative.parts] else "development"
        records.append({"capture_id": relative.as_posix(), "room": "A",
                        "activity": ROOM_A_FOLDERS[folder], "split": split})
    return pd.DataFrame(records)


def make_split_b(raw_dir: Path, per_class: int = 25, seed: int = 42) -> pd.DataFrame:
    """Sorted filenames, one shared random.Random(42), 25 per class (same as split_room_b.py)."""
    rng = random.Random(seed)
    records = []
    for label, prefix in ROOM_B_PREFIXES.items():
        files = sorted(p for p in raw_dir.glob("*.csv") if p.name.startswith(prefix + "_"))
        if len(files) < per_class:
            raise ValueError(f"Insufficient captures for {label}")
        selected = set(rng.sample(files, per_class))
        for file in files:
            records.append({"capture_id": file.name, "room": "B", "activity": label,
                            "split": "validation" if file in selected else "development"})
    return pd.DataFrame(records)


def load_or_create_split(cfg, room: str) -> pd.DataFrame:
    path = cfg.path("split_a" if room == "A" else "split_b")
    raw = cfg.path("raw_experiment2" if room == "A" else "raw_experiment3")
    if path.exists():
        split = pd.read_csv(path)
        print(f"Room {room}: reusing split manifest {path}")
    else:
        split = make_split_a(raw) if room == "A" else make_split_b(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        print(f"Room {room}: created split manifest {path}")

    expected = cfg["expected_counts"]
    counts = Counter(split["split"])
    if expected.get("enforce", True):
        if room == "A":
            ok = len(split) == expected["room_a_total"] and counts["validation"] == expected["room_a_validation"]
        else:
            per_class = split[split.split == "validation"].groupby("activity").size()
            ok = len(split) == expected["room_b_total"] and (per_class == expected["room_b_validation_per_class"]).all()
        if not ok:
            raise ValueError(f"Room {room} split counts {dict(counts)} do not match expected_counts")
    # Every manifest entry must exist on disk.
    missing = [c for c in split.capture_id if not (raw / c).exists()]
    if missing:
        raise FileNotFoundError(f"Room {room}: {len(missing)} manifest captures missing, e.g. {missing[:3]}")
    return split


# ---------------------------------------------------------------------------
# 2-4. Per-capture processing (runs in worker processes)
# ---------------------------------------------------------------------------

def _process(job):
    cfg, record = job
    room, capture_id, split = record["room"], record["capture_id"], record["split"]
    raw = cfg.path("raw_experiment2" if room == "A" else "raw_experiment3")
    source = raw / capture_id
    row = dict(record)
    try:
        if room == "A":
            amplitude, report = read_room_a_capture(source, cfg)
        else:
            amplitude, report = read_room_b_capture(source)
        cleaned, qc = quality_control(amplitude, cfg)
        row.update(report)
        row.update(qc)

        relative = Path(capture_id).with_suffix("")
        if cfg["preprocessing"].get("write_intermediate_csv", True) and len(cleaned):
            csv_path = cfg.path("intermediate_csv") / f"room_{room}" / relative.with_suffix(".csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cleaned, columns=COLUMNS).to_csv(csv_path, index=False, float_format="%.6g")

        if qc["qc_rejected_capture"]:
            row["status"] = "rejected_qc"
            return row, None
        window, start, _ = select_window(cleaned, cfg, allow_short=False)
        if window is None:
            row["status"] = "short"
            return row, None

        out = cfg.path("windows") / split / f"room_{room}" / relative.with_suffix(".npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, window)
        row.update({"status": "ok", "window_start": start,
                    "window_path": out.relative_to(cfg.path("windows")).as_posix()})
        return row, out
    except Exception as error:  # keep going; the failure is recorded in the manifest
        row["status"] = f"error: {error}"
        return row, None


# ---------------------------------------------------------------------------
# 5-6. Normalization
# ---------------------------------------------------------------------------

def task2_path(cfg, row) -> Path:
    """Exact layout Krishna's src/task2/dataset.py reads."""
    if row["room"] == "A":
        root = cfg.path("task2_room_a_root")
        return root / row["split"] / "room_A" / Path(row["capture_id"]).with_suffix(".npy")
    root = cfg.path("task2_room_b_root")
    return root / row["split"] / "room_B" / (Path(row["capture_id"]).stem + ".npy")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    split = pd.concat([load_or_create_split(cfg, "A"), load_or_create_split(cfg, "B")], ignore_index=True)
    jobs = [(cfg, r) for r in split.to_dict("records")]
    print(f"Processing {len(jobs)} captures with {args.workers} workers ...")

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (row, _) in enumerate(pool.map(_process, jobs, chunksize=8), start=1):
            rows.append(row)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}")
    manifest = pd.DataFrame(rows)
    manifest["preprocessing_hash"] = cfg.preprocessing_hash
    ok = manifest[manifest.status == "ok"].copy()

    def windows_of(frame):
        for rel in frame.window_path:
            yield np.load(cfg.path("windows") / rel)

    dev = ok[ok.split == "development"]
    norm_dir = cfg.path("normalization_dir")
    normalizers = {
        "A": Normalizer.fit(windows_of(dev[dev.room == "A"]), "A"),
        "B": Normalizer.fit(windows_of(dev[dev.room == "B"]), "B"),
        "pooled": Normalizer.fit(windows_of(dev), "pooled"),
    }
    for name, norm in normalizers.items():
        norm.save(norm_dir / f"{name}.npz")
    # Legacy file names used by Member 1's handover (configs/room_*_normalization.npz).
    normalizers["A"].save(norm_dir.parent / "room_a_normalization.npz")
    normalizers["B"].save(norm_dir.parent / "room_b_normalization.npz")

    task2_paths = []
    for row in ok.to_dict("records"):
        window = np.load(cfg.path("windows") / row["window_path"])
        out = task2_path(cfg, row)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, normalizers[row["room"]](window))
        task2_paths.append(str(out))
    manifest.loc[ok.index, "task2_path"] = task2_paths
    manifest["window_id"] = manifest["room"] + ":" + manifest["capture_id"]

    front = ["window_id", "capture_id", "room", "activity", "split", "status", "window_start",
             "window_path", "csi_packets", "qc_valid_rows", "preprocessing_hash"]
    manifest = manifest[front + [c for c in manifest.columns if c not in front]]
    out_manifest = cfg.path("windows") / "manifest.csv"
    manifest.to_csv(out_manifest, index=False)

    summary = {"preprocessing_hash": cfg.preprocessing_hash, "captures": int(len(manifest))}
    for (room, sp), group in manifest.groupby(["room", "split"]):
        summary[f"room_{room}_{sp}"] = {k: int(v) for k, v in Counter(group.status).items()}
    for column in ("truncated_records", "non_udp_records", "rejected_payloads", "qc_zero_rows",
                   "qc_duplicate_rows", "qc_nonfinite_rows", "qc_negative_rows"):
        if column in manifest:
            summary[column] = int(manifest[column].fillna(0).sum())
    results = cfg.path("results") / "preprocessing"
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    manifest[manifest.status != "ok"].to_csv(results / "excluded_captures.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("Manifest:", out_manifest)


if __name__ == "__main__":
    main()
