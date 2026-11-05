"""Task 1 - Recorded-environment identification (Room A vs Room B).

    python src/task1/room_identification.py [--config configs/experiment.yaml]

* Trains on DEVELOPMENT windows of both rooms, scores the fixed VALIDATION windows.
* Headline: official 150-observation windows.
* Environment-fingerprint figure: accuracy vs. centre-cropped window length,
  for the default (pooled-standardized) features and a gain-invariant variant.
* Saves the 150-observation router used by Task 3 and run_episode.py.

Outputs (results/task1/): metrics.csv, room_accuracy_vs_window.csv/.png,
predictions.csv, confusion_matrix.csv; models/room_router.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.config import load_config  # noqa: E402
from pipeline.data import load_windows  # noqa: E402
from pipeline.ingest import load_normalizers  # noqa: E402
from pipeline.router import RoomRouter  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    tcfg = cfg["task1"]

    data = load_windows(cfg)
    pooled = load_normalizers(cfg)["pooled"]
    dev = data.subset(data.meta.split == "development")
    val = data.subset(data.meta.split == "validation")
    print(f"development windows: {len(dev)}  validation windows: {len(val)}")

    out = cfg.path("results") / "task1"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for gain_invariant in (False, True):
        variant = "gain_invariant" if gain_invariant else "pooled_standardized"
        for length in tcfg["window_lengths"]:
            router = RoomRouter(pooled, gain_invariant, tcfg["C"], length)
            router.fit(dev.windows, dev.meta.room)
            pred, _ = router.predict(val.windows)
            acc = accuracy_score(val.meta.room, pred)
            per_room = {f"acc_room_{r}": accuracy_score(val.meta.room[val.meta.room == r],
                                                        pred[val.meta.room == r]) for r in "AB"}
            rows.append({"variant": variant, "window_length": length, "accuracy": acc, **per_room})
            print(f"{variant:20s} L={length:3d}  accuracy={acc:.4f}")
    curve = pd.DataFrame(rows)
    curve.to_csv(out / "room_accuracy_vs_window.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for variant, group in curve.groupby("variant"):
        ax.plot(group.window_length, group.accuracy, marker="o", label=variant.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xticks(tcfg["window_lengths"])
    ax.set_xticklabels(tcfg["window_lengths"])
    ax.set_xlabel("window length (consecutive observations, centre crop)")
    ax.set_ylabel("room accuracy (validation)")
    ax.set_ylim(min(0.45, curve.accuracy.min() - 0.02), 1.01)
    ax.set_title("Environment fingerprint: room accuracy vs window length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "room_accuracy_vs_window.png", dpi=200)
    plt.close(fig)

    # Headline router: official 150-observation windows, configured variant.
    router = RoomRouter(pooled, bool(tcfg["gain_invariant"]), tcfg["C"], 150)
    router.fit(dev.windows, dev.meta.room)
    pred, conf = router.predict(val.windows)
    accuracy = accuracy_score(val.meta.room, pred)
    cm = confusion_matrix(val.meta.room, pred, labels=["A", "B"])
    pd.DataFrame(cm, index=["true_A", "true_B"], columns=["pred_A", "pred_B"]).to_csv(
        out / "confusion_matrix.csv")
    pd.DataFrame({"window_id": val.meta.window_id, "room": pred, "confidence": conf,
                  "true_room": val.meta.room}).to_csv(out / "predictions.csv", index=False)
    variant = "gain_invariant" if tcfg["gain_invariant"] else "pooled_standardized"
    pd.DataFrame([
        {"experiment": "task1_room_id", "target_room": "A+B", "K": "", "metric": "accuracy",
         "value": accuracy, "variant": variant, "n_validation": len(val)},
        {"experiment": "task1_room_id", "target_room": "A", "K": "", "metric": "accuracy",
         "value": cm[0, 0] / max(cm[0].sum(), 1), "variant": variant, "n_validation": int(cm[0].sum())},
        {"experiment": "task1_room_id", "target_room": "B", "K": "", "metric": "accuracy",
         "value": cm[1, 1] / max(cm[1].sum(), 1), "variant": variant, "n_validation": int(cm[1].sum())},
    ]).to_csv(out / "metrics.csv", index=False)

    models = cfg.path("models")
    models.mkdir(parents=True, exist_ok=True)
    joblib.dump({"router": router, "preprocessing_hash": cfg.preprocessing_hash},
                models / "room_router.joblib")
    print(f"\nHeadline ({variant}, 150 obs): validation room accuracy = {accuracy:.4f}")
    print("confusion (rows true A/B):\n", cm)
    print("Saved:", out, "and", models / "room_router.joblib")


if __name__ == "__main__":
    main()
