"""Reproduce every result from the raw source data with one command.

    python scripts/run_all.py                      # real data, configs/experiment.yaml
    python scripts/run_all.py --config other.yaml  # any config
    python scripts/run_all.py --skip-build         # reuse existing windows

Order: decoder check -> build_dataset -> Task 1 -> encoder diagnosis -> Task 3.
Task 2 SSL (re)training needs PyTorch and is run separately (see README).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("decoder check", ["scripts/verify_decoder.py"], False),
    ("preprocessing", ["src/preprocessing/build_dataset.py"], True),
    ("task 1", ["src/task1/room_identification.py"], True),
    ("encoder diagnosis", ["src/analysis/encoder_diagnosis.py"], True),
    ("shared-activity transfer probe", ["src/analysis/shared_transfer_probe.py"], True),
    ("task 3", ["src/task3/few_shot_adaptation.py"], True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    for name, command, takes_config in STEPS:
        if args.skip_build and name == "preprocessing":
            continue
        cmd = [sys.executable, str(ROOT / command[0]), *command[1:]]
        if takes_config and args.config:
            cmd += ["--config", args.config]
        print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
        started = time.time()
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"=== {name} finished in {time.time() - started:.0f}s", flush=True)
    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
