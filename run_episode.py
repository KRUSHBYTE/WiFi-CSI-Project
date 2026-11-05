"""End-to-end inference on a challenge episode.

    python run_episode.py --episode path/to/episode_dir --output predictions.csv
    python run_episode.py --episode DIR --support support.csv --output predictions.csv

The episode directory may contain, at any depth, raw Experiment-2 style ``.pcap``
captures, Experiment-3 style amplitude ``.csv`` captures, and/or ``.npy`` amplitude
windows of shape (T, 256). Each capture goes through exactly the same Python
pipeline as the benchmark data:

    ingest -> common 256-column amplitude -> QC -> centre 150 x 256 window
    -> Task 1 room routing -> room-specific normalization -> features
    -> per-room feature alignment -> adapted few-shot activity classifier

Optional ``--support`` CSV (columns ``path,label``; paths relative to the CSV) gives
episode-specific labeled captures of target-exclusive activities. They are routed
like any other capture and replace the stored development support for that room,
so the model adapts to the K shots supplied with the episode.

Output CSV columns: window_id, room, label, confidence
(a diagnostics file <output>.details.csv adds QC/window notes per capture).

Requires models/deploy.joblib, produced by src/task3/few_shot_adaptation.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.adaptation import align, make_classifier  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.features import FeatureExtractor  # noqa: E402
from pipeline.ingest import Normalizer, quality_control, read_capture, select_window  # noqa: E402

CAPTURE_SUFFIXES = {".pcap", ".cap", ".csv", ".npy"}


def collect(episode: Path, exclude: set[Path]) -> list[Path]:
    files = [p for p in sorted(episode.rglob("*"))
             if p.is_file() and p.suffix.lower() in CAPTURE_SUFFIXES and p.resolve() not in exclude
             and not p.name.startswith("predictions")]
    if not files:
        raise SystemExit(f"No .pcap/.csv/.npy captures found under {episode}")
    return files


def to_windows(paths, cfg):
    """Return (windows (N,150,256) | None per path, details rows)."""
    windows, details = [], []
    for path in paths:
        note = {"source": str(path)}
        try:
            amplitude, report = read_capture(path, cfg)
            cleaned, qc = quality_control(amplitude, cfg)
            window, start, padded = select_window(cleaned, cfg, allow_short=True)
            note.update(report)
            note.update({"valid_observations": qc["qc_valid_rows"], "window_start": start,
                         "padded_short_capture": padded})
            if window is None:
                raise ValueError("no valid CSI observations")
            note["status"] = "ok_padded" if padded else "ok"
        except Exception as error:
            window = None
            note["status"] = f"error: {error}"
        windows.append(window)
        details.append(note)
    return windows, details


class EpisodeModel:
    def __init__(self, bundle, cfg):
        self.bundle = bundle
        self.router = bundle["router"]
        self.norm = {d: Normalizer(*bundle["normalizers"][d], d) for d in ("A", "B", "pooled")}
        self.extractor = FeatureExtractor(bundle["extractor"], cfg)
        self.models = {r: bundle["models"][r]["classifier"] for r in "AB"}

    def features(self, windows, room):
        x = self.extractor(np.stack([self.norm[room](w) for w in windows]))
        return align(x, self.bundle["feature_stats"][room]) if self.bundle["feature_alignment"] else x

    def adapt(self, windows, labels):
        """Replace the stored support with episode-provided labeled captures."""
        rooms, _ = self.router.predict(np.stack(windows))
        labels = np.asarray(labels)
        for room in sorted(set(rooms)):
            rows = np.flatnonzero(rooms == room)
            allowed = set(self.bundle["target_exclusive"][room])
            bad = sorted(set(labels[rows]) - allowed)
            if bad:
                print(f"warning: support labels {bad} routed to room {room} are not target-exclusive "
                      f"activities of that room {sorted(allowed)}; they are still used as support")
            support_x = self.features([windows[i] for i in rows], room)
            train = self.bundle["training"][room]
            x = np.concatenate([train["source_x"], support_x])
            y = np.concatenate([train["source_y"], labels[rows]])
            self.models[room] = make_classifier(self.bundle["classifier"]).fit(x, y)
            print(f"room {room}: adapted with {len(rows)} episode support captures "
                  f"({pd.Series(labels[rows]).value_counts().to_dict()})")

    def predict(self, windows):
        stack = np.stack(windows)
        rooms, _ = self.router.predict(stack)
        labels = np.empty(len(windows), dtype=object)
        confidence = np.zeros(len(windows))
        for room in sorted(set(rooms)):
            rows = np.flatnonzero(rooms == room)
            model = self.models[room]
            proba = model.predict_proba(self.features([windows[i] for i in rows], room))
            labels[rows] = model.classes_[proba.argmax(axis=1)]
            confidence[rows] = proba.max(axis=1)
        return rooms, labels, confidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", required=True, help="directory with the episode's captures")
    parser.add_argument("--output", default=None, help="predictions CSV (default <episode>/predictions.csv)")
    parser.add_argument("--support", default=None, help="optional CSV with columns path,label")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    bundle_path = cfg.path("models") / "deploy.joblib"
    if not bundle_path.exists():
        raise SystemExit(f"{bundle_path} missing: run scripts/run_all.py (or task1 + task3) first")
    bundle = joblib.load(bundle_path)
    if bundle["preprocessing_hash"] != cfg.preprocessing_hash:
        raise SystemExit("deploy.joblib was trained with a different preprocessing config; retrain it")
    model = EpisodeModel(bundle, cfg)

    episode = Path(args.episode).resolve()
    exclude = set()
    if args.support:
        support_csv = Path(args.support).resolve()
        exclude.add(support_csv)
        table = pd.read_csv(support_csv)
        paths = [(support_csv.parent / p).resolve() for p in table["path"]]
        exclude.update(paths)
        windows, details = to_windows(paths, cfg)
        failed = [d["source"] for w, d in zip(windows, details) if w is None]
        if failed:
            raise SystemExit(f"support captures could not be processed: {failed}")
        model.adapt(windows, table["label"].astype(str).str.upper().tolist())

    paths = collect(episode, exclude)
    windows, details = to_windows(paths, cfg)
    ok = [i for i, w in enumerate(windows) if w is not None]
    rooms, labels, confidence = model.predict([windows[i] for i in ok]) if ok else ([], [], [])

    result = pd.DataFrame({
        "window_id": [paths[i].relative_to(episode).as_posix() for i in range(len(paths))],
        "room": "", "label": "UNPROCESSABLE", "confidence": 0.0})
    result.loc[ok, "room"] = list(rooms)
    result.loc[ok, "label"] = list(labels)
    result.loc[ok, "confidence"] = np.round(confidence, 6) if len(ok) else []

    output = Path(args.output) if args.output else episode / "predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    pd.DataFrame(details).assign(window_id=result.window_id).to_csv(
        output.with_suffix(".details.csv"), index=False)
    print(result.to_string(index=False))
    print(f"\n{len(ok)}/{len(paths)} captures predicted -> {output}")


if __name__ == "__main__":
    main()
