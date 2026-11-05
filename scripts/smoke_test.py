"""End-to-end smoke test on a generated synthetic dataset (no real data needed).

    python scripts/smoke_test.py [--workspace /tmp/csi_smoke]

Generates synthetic captures in the real formats, runs scripts/run_all.py with the
generated config, runs run_episode.py on a mixed PCAP/CSV episode, and checks that
every expected output exists and is well-formed. Numbers are NOT meaningful.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()
    ws = Path(args.workspace or tempfile.mkdtemp(prefix="csi_smoke_")).resolve()
    py = sys.executable

    run([py, ROOT / "scripts/make_synthetic_dataset.py", "--out", ws])
    config = ws / "experiment_synthetic.yaml"
    run([py, ROOT / "scripts/run_all.py", "--config", config])
    run([py, ROOT / "run_episode.py", "--config", config, "--episode", ws / "episode",
         "--output", ws / "episode_predictions.csv"])

    expected = [
        "data/windows/manifest.csv", "configs/normalization/A.npz", "configs/normalization/pooled.npz",
        "results/preprocessing/summary.json", "results/task1/metrics.csv",
        "results/task1/room_accuracy_vs_window.png", "results/encoder_diagnosis/linear_probe.csv",
        "results/task3/summary.csv", "results/task3/label_efficiency.png",
        "results/task3/routing_contribution.png", "results/task3/confusion_K5_target_A.png",
        "results/task3/confusion_K5_target_B.png", "models/deploy.joblib", "episode_predictions.csv",
    ]
    missing = [p for p in expected if not (ws / p).exists()]
    assert not missing, f"missing outputs: {missing}"

    predictions = pd.read_csv(ws / "episode_predictions.csv")
    assert list(predictions.columns) == ["window_id", "room", "label", "confidence"], predictions.columns
    truth = pd.read_csv(ws / "episode_truth.csv")
    merged = predictions.merge(truth, on="window_id", suffixes=("", "_true"))
    assert len(merged) == len(truth), "episode predictions missing captures"
    summary = pd.read_csv(ws / "results/task3/summary.csv")
    for target in "AB":
        for k in (1, 5, 10, 20):
            assert ((summary.target_room == target) & (summary.K == k)).any(), (target, k)
    print(f"\nSMOKE TEST PASSED  (workspace {ws})")
    print("episode room accuracy :", round((merged.room == merged.room_true).mean(), 3))
    print("episode label accuracy:", round((merged.label == merged.label_true).mean(), 3))


if __name__ == "__main__":
    main()
