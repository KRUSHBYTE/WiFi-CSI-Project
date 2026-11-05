"""How transferable are the SHARED activities (SIT / STAND / WALK) between rooms?

    python src/analysis/shared_transfer_probe.py [--config configs/experiment.yaml]

For each feature extractor, a 3-class logistic regression is trained on the shared
activities of one room's DEVELOPMENT captures and scored on the shared-activity
VALIDATION captures of (a) the same room and (b) the other room. The gap between
the two is an upper-bound-style diagnostic of what Task 3 can achieve on the
shared activities without target labels (chance macro-F1 = 0.33).

Output: results/analysis/shared_transfer_probe.csv (+ confusion matrices)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.adaptation import FeatureBank  # noqa: E402
from pipeline.config import load_config, other_room  # noqa: E402
from pipeline.data import load_windows  # noqa: E402
from pipeline.features import FeatureExtractor  # noqa: E402
from pipeline.ingest import load_normalizers  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    shared = cfg["task3"]["shared_activities"]

    data = load_windows(cfg)
    meta = data.meta
    norms = {r: n for r, n in load_normalizers(cfg).items() if r in ("A", "B")}
    dev = (meta.split == "development").to_numpy()
    is_shared = meta.activity.isin(shared).to_numpy()
    y = meta.activity.to_numpy()

    rows, confusions = [], []
    for kind in cfg["task3"]["feature_extractors"]:
        bank = FeatureBank(data.windows, meta, norms, FeatureExtractor(kind, cfg), alignment=True)
        for source in "AB":
            target = other_room(source)
            train = np.flatnonzero(dev & is_shared & (meta.room == source).to_numpy())
            clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=42)
            clf.fit(bank.get(train, source), y[train])
            for evaluated, label in ((source, "within_room"), (target, "cross_room")):
                test = np.flatnonzero(~dev & is_shared & (meta.room == evaluated).to_numpy())
                pred = clf.predict(bank.get(test, evaluated))
                rows.append({"extractor": kind, "train_room": source, "test_room": evaluated,
                             "setting": label, "macro_f1": f1_score(y[test], pred, labels=shared, average="macro"),
                             "n_test": len(test)})
                if label == "cross_room":
                    cm = confusion_matrix(y[test], pred, labels=shared)
                    for i, t in enumerate(shared):
                        confusions.append({"extractor": kind, "direction": f"{source}->{target}", "true": t,
                                           **{f"pred_{p}": int(cm[i, j]) for j, p in enumerate(shared)}})
    out = cfg.path("results") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "shared_transfer_probe.csv", index=False)
    pd.DataFrame(confusions).to_csv(out / "shared_transfer_confusion.csv", index=False)
    print(table.pivot_table(index="extractor", columns=["train_room", "setting"], values="macro_f1").round(3))
    print("Saved:", out)


if __name__ == "__main__":
    main()
