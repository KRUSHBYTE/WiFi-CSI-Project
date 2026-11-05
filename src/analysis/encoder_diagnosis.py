"""Encoder diagnosis on the regenerated windows (no PyTorch needed).

    python src/analysis/encoder_diagnosis.py [--config configs/experiment.yaml]

1. Re-scores the Task 2 frozen linear probe on the CURRENT windows for the SSL
   encoder, the same architecture with random weights, and handcrafted features
   (room-normalized inputs, StandardScaler + LogisticRegression(balanced), dev->val).
2. 2-D t-SNE of validation+development embeddings coloured once by room and once
   by activity (qualitative diagnostic only).

Outputs: results/encoder_diagnosis/linear_probe.csv, tsne_{extractor}_by_{room,activity}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.config import ALL_LABELS, load_config  # noqa: E402
from pipeline.data import load_windows  # noqa: E402
from pipeline.features import FeatureExtractor  # noqa: E402
from pipeline.ingest import load_normalizers  # noqa: E402


def scatter(points, labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for value in sorted(set(labels)):
        mask = labels == value
        ax.scatter(points[mask, 0], points[mask, 1], s=8, alpha=0.6, label=value)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tsne-max", type=int, default=2500)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    data = load_windows(cfg)
    norms = load_normalizers(cfg)
    normalized = np.stack([norms[r](w) for r, w in zip(data.meta.room, data.windows)])
    dev = (data.meta.split == "development").to_numpy()
    y = data.meta.activity.to_numpy()
    rooms = ("room " + data.meta.room).to_numpy()

    out = cfg.path("results") / "encoder_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ("ssl", "random", "handcrafted"):
        x = FeatureExtractor(kind, cfg)(normalized)
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=42))
        clf.fit(x[dev], y[dev])
        pred = clf.predict(x[~dev])
        rows.append({"extractor": kind,
                     "macro_f1": f1_score(y[~dev], pred, labels=ALL_LABELS, average="macro", zero_division=0),
                     "balanced_accuracy": balanced_accuracy_score(y[~dev], pred),
                     "n_train": int(dev.sum()), "n_validation": int((~dev).sum())})
        print(rows[-1])
        if kind in ("ssl", "handcrafted"):
            rng = np.random.default_rng(42)
            pick = np.sort(rng.choice(len(x), size=min(args.tsne_max, len(x)), replace=False))
            z = StandardScaler().fit_transform(x[pick])
            points = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                          random_state=42).fit_transform(z)
            scatter(points, rooms[pick], f"{kind} features - t-SNE by room", out / f"tsne_{kind}_by_room.png")
            scatter(points, y[pick], f"{kind} features - t-SNE by activity", out / f"tsne_{kind}_by_activity.png")
    pd.DataFrame(rows).to_csv(out / "linear_probe.csv", index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
