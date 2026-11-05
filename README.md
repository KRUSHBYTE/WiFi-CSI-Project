# WiFi CSI — Cross-Room Few-Shot Human Activity Recognition

**CS60055 Ubiquitous Computing | Hackathon Challenge 2**

A Python pipeline that goes from raw source captures (Experiment-2 Nexmon `.pcap`, Experiment-3 amplitude `.csv`) to (1) recorded-environment identification, (2) a self-supervised CSI encoder, and (3) cross-room few-label activity recognition, plus an end-to-end `run_episode.py` entry point.

All numbers below are **validation results** on the fixed challenge validation sets. They are not unseen-test accuracy.

---

## 1. Status

| Component | What it does | Status |
|---|---|---|
| Preprocessing | PCAP/CSV ingestion → 256-column amplitude → QC → 150×256 windows → dev-only normalization, with a provenance manifest | Done, run on the real data |
| Task 1 | Room A vs Room B from the CSI window only | Done: **100.0 % validation accuracy** (272 windows) |
| Task 2 | Masked-reconstruction SSL encoder + frozen linear probe + baselines | Done (Member 2). Re-scored on the regenerated windows, which reproduce the original numbers exactly |
| Task 3 | Few-label adaptation, K ∈ {1, 5, 10, 20}, A→B and B→A, routing study | Done: see section 8 |
| `run_episode.py` | Raw episode folder → `window_id, room, label, confidence` | Done, tested on real captures |

## 2. Data

| Room | Source | Classes | Captures | Development / validation | Windows used (dev / val) |
|---|---|---|---:|---|---:|
| A | Experiment-2 `.pcap` (BCM4366c0, 80 MHz) | FALL, SIT, SIT-DOWN, STAND, STAND-UP, WALK | 1,084 | 934 / 150, supplied `validation/` folder | 925 / 147 |
| B | Experiment-3 amplitude `.csv` | EMPTY, LYING, SIT, STAND, WALK | 1,201 | 1,076 / 125, 25 per class, sorted filenames, `random.Random(42)` | 1,076 / 125 |

Shared activities: SIT, STAND, WALK.

Twelve Room A captures (9 development, 3 validation) have fewer than 150 valid observations. They are excluded from all benchmark windows (`results/preprocessing/excluded_captures.csv`), so Room A validation results use 147 of the 150 official captures. `run_episode.py` still predicts such captures at inference time by edge-padding, and flags them in its `.details.csv`.

**Dataset placement.** Never commit the data. Put it here:

```text
data/raw/Experiment-2/realdata/input_data/30ms/csidata/{fall,sit,...,validation/...}/*.pcap
data/raw/Experiment-3/Data/*.csv
```

Extract only these folders from `Experiment-2.zip` and `Experiment-3.zip`. Do not use the archives' `matlab/` folder, notebook or `Results/`, and skip `__MACOSX/` and `.DS_Store`.

The Google-Drive *folder* download contains one extra file, `stand/stand65(1).pcap`. It is a byte-identical duplicate of `stand65.pcap` and must not be copied. The split manifest has exactly 1,084 Room A captures, and `build_dataset.py` refuses to run if counts or files don't match.

## 3. Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The core pipeline needs only NumPy, pandas, SciPy, scikit-learn, matplotlib, joblib and PyYAML.

**PyTorch is needed only to retrain the Task 2 encoder** (`src/task2/`). Inference with the pretrained encoder runs in NumPy: `src/pipeline/features.py` reads `checkpoints/task2/csi_encoder.pt` directly and re-implements the forward pass. It reproduces Member 2's PyTorch linear-probe result exactly (macro-F1 0.8546, balanced accuracy 0.8753).

## 4. Reproduce everything

```bash
python scripts/run_all.py
```

This runs, in order:

1. `scripts/verify_decoder.py`: the vectorized decoder is bit-identical to the reference `csi_decode.py` on random words and real packets.
2. `src/preprocessing/build_dataset.py`: raw data to windows, manifest and normalization.
3. `src/task1/room_identification.py`: Task 1 and the router.
4. `src/analysis/encoder_diagnosis.py`: Task 2 linear probe re-scored and t-SNE plots.
5. `src/analysis/shared_transfer_probe.py`: how transferable SIT/STAND/WALK are between rooms.
6. `src/task3/few_shot_adaptation.py`: Task 3, its figures, and `models/deploy.joblib`.

The full run takes about 15 min on a 2-core CPU. Use `--skip-build` to reuse the existing windows.

**Smoke test without the real data.** This generates a synthetic dataset in the exact source formats, runs everything plus `run_episode.py`, and checks every output:

```bash
python scripts/smoke_test.py
```

**Task 2 retraining** (PyTorch, from the project root): see `src/task2/README.md`.

```bash
python src/task2/train_ssl.py && python src/task2/linear_probe.py
```

## 5. Run an episode

```bash
python run_episode.py --episode path/to/episode --output predictions.csv
python run_episode.py --episode path/to/episode --support support.csv --output predictions.csv
```

* The episode folder may contain `.pcap`, `.csv` and/or `.npy` (T×256 amplitude) files at any depth.
* `predictions.csv` columns: `window_id, room, label, confidence`. `predictions.details.csv` adds packet, QC and window notes per capture.
* `--support` (optional): a CSV with columns `path,label`, giving the K labeled target-exclusive captures supplied with the episode. They replace the stored development support for the room they are routed to.
* Every capture goes through the same code as the benchmark:

  ingest → QC → centre window → Task 1 routing → that room's normalization → features → that room's feature alignment → adapted classifier

## 6. Preprocessing design (`src/pipeline/ingest.py`, `src/preprocessing/build_dataset.py`)

* **PCAP parsing.** `src/preprocessing/pcap_reader.py` is a small dependency-free libpcap reader (Ethernet/VLAN/SLL/raw-IP → IPv4 → UDP). It counts truncated records explicitly.
  * It replaces Scapy in the active pipeline; Scapy remains only in `archive/` and the superseded per-room scripts.
  * The 2 records rejected in the Member 1 audit are exactly the 2 truncated records that Scapy had warned about.
* **CSI reconstruction.** A valid observation is a 1,042-byte UDP payload with magic `0x1111`: an 18-byte Nexmon header followed by 256 little-endian packed words.
  * The BCM4366c0 packed fields are 11-bit real/imag mantissas with sign bits and a shared 6-bit exponent.
  * They are decoded with packet-wise scaling by `decode_csi_batch`, a vectorized, bit-exact version of Member 1's decoder. Amplitude = |CSI|, and 256 subcarriers are kept.
* **Room B.** The first row is a header (integer or `Subcarrier_*` names). The 256 numeric columns are kept in the original row order.
* **Common form.** One CSV per capture with columns `subcarrier_001 … subcarrier_256` in `data/intermediate_csv/room_{A,B}/`.
* **Quality control.** Drops non-finite rows, all-zero rows and exact consecutive duplicates, keeping temporal order. It counts negative values without dropping them by default. It rejects a capture if fewer than 50 % of its observations survive.
  * On the real data, QC removes 0 rows.
  * Every count is recorded per capture in `data/windows/manifest.csv`.
* **Windowing.** A deterministic centre window: start = (N − 150) // 2 over valid observations. Windows never mix captures, and captures shorter than 150 are excluded.
* **Normalization.** Per-subcarrier z-score, with statistics fitted **only on development windows** (`configs/normalization/`):
  * `A`, `B`: per room. Used after routing, and for Task 2 inputs.
  * `pooled`: both rooms, no room labels. Used by the Task 1 router and by the no-routing ablation.
  * Regenerated `A`/`B` statistics equal Member 1's committed files exactly (max difference 0.0), which settles the open "recalculate Room A normalization" item.
* **Provenance.** `data/windows/manifest.csv` has one row per original capture, including excluded ones: `window_id`, capture path, room, label, split, status, window start, packet and QC counts, and the SHA-256 hash of the `preprocessing` block of `configs/experiment.yaml`. Every downstream script refuses windows built with a different hash.

## 7. Task 1 — recorded-environment identification

* **Input.** The 150×256 amplitude window only, standardized with the **pooled** statistics so preprocessing does not reveal the room. No filename, extension, length or packet-count features.
* **Features.** Per-subcarrier mean, standard deviation and mean |first difference| (768 values).
* **Classifier.** StandardScaler + logistic regression, trained on development windows of both rooms.

| Variant | Validation room accuracy (150 obs) |
|---|---:|
| Pooled-standardized amplitude (headline) | **1.000** (147/147 A, 125/125 B) |
| Gain-invariant (log amplitude minus window mean; spectral shape only) | 0.996 |

Accuracy vs window length (`results/task1/room_accuracy_vs_window.png`):

* The pooled-standardized variant reaches 100 % even from **1** observation, because the static multipath profile of each room/session is very distinctive.
* The gain-invariant variant reaches 87.5 % at 1 observation and 99.3 % at 2.

Because the two rooms were recorded in different sessions, this fingerprint may capture session effects as well as geometry (challenge limitation 1).

## 8. Task 3 — cross-room few-label adaptation

**Protocol** (`src/pipeline/adaptation.py`). For target room T and source S:

* **Shared activities** (SIT, STAND, WALK) are learned **only from S's labeled development captures**. No target labels are used for them.
* **Target-exclusive activities** (A: SIT-DOWN, STAND-UP, FALL; B: EMPTY, LYING) get **K distinct target development captures each**, with one window per capture, so K captures = K shots.
* **Support draws.** Five independent draws (seeds 0–4) per K. Validation captures are never support.
* **Scoring.** On all eligible validation windows of T, over T's full label set.

**Environment-aware routing.** Each window is assigned a domain, which selects:

1. its input normalization (room statistics), and
2. its feature alignment: per-domain standardization of features using the domain's **unlabeled** development windows (unsupervised feature alignment, as listed in challenge section 15).

The domain comes from the Task 1 router (`predicted`), the true room (`oracle`), or one pooled domain for everything (`none`, the same model without routing).

**Headline system:** "dynamics" features, meaning per-subcarrier temporal std and mean |first difference| with the static mean removed, a class-balanced logistic regression, and predicted routing. Macro-F1 is mean ± std over 5 support draws.

| Direction | K | Macro-F1 (all target activities) | Target-exclusive macro-F1 | Shared macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| A→B | 1 | 0.260 ± 0.082 | 0.260 | 0.260 | 0.274 |
| A→B | 5 | 0.429 ± 0.041 | 0.630 | 0.295 | 0.413 |
| A→B | 10 | 0.468 ± 0.027 | 0.723 | 0.297 | 0.453 |
| A→B | 20 | **0.517 ± 0.022** | 0.802 | 0.327 | 0.507 |
| B→A | 1 | 0.154 ± 0.058 | 0.284 | 0.024 | 0.147 |
| B→A | 5 | 0.299 ± 0.026 | 0.538 | 0.059 | 0.343 |
| B→A | 10 | 0.355 ± 0.033 | 0.647 | 0.063 | 0.425 |
| B→A | 20 | **0.339 ± 0.016** | 0.633 | 0.045 | 0.429 |

**Feature extractors** (macro-F1, logistic regression, predicted routing):

| Extractor | A→B K=1 | 5 | 10 | 20 | B→A K=1 | 5 | 10 | 20 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SSL encoder (Task 2) | 0.159 | 0.256 | 0.331 | 0.280 | 0.207 | 0.367 | 0.361 | 0.374 | 0.292 |
| Same CNN, random weights | 0.256 | 0.403 | 0.432 | 0.428 | 0.192 | 0.351 | 0.385 | 0.411 | 0.357 |
| Handcrafted (mean, std, diff) | 0.215 | 0.356 | 0.403 | 0.421 | 0.183 | 0.329 | 0.365 | 0.356 | 0.329 |
| **Dynamics (std, diff)** | 0.260 | 0.429 | 0.468 | 0.517 | 0.154 | 0.299 | 0.355 | 0.339 | 0.353 |

**Routing contribution** (dynamics, logistic regression, macro-F1):

| Routing | A→B K=1 | 5 | 10 | 20 | B→A K=1 | 5 | 10 | 20 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predicted room (Task 1) | 0.260 | 0.429 | 0.468 | 0.517 | 0.154 | 0.299 | 0.355 | 0.339 | 0.353 |
| Oracle room | 0.260 | 0.429 | 0.468 | 0.517 | 0.154 | 0.299 | 0.355 | 0.339 | 0.353 |
| No routing (pooled) | 0.341 | 0.353 | 0.360 | 0.363 | 0.177 | 0.263 | 0.316 | 0.317 | 0.311 |

### Interpretation

* **Label efficiency works for the target-labeled activities.** Target-exclusive macro-F1 rises from 0.26–0.28 at K=1 to 0.63–0.80 at K=20.
* **Transferring the shared activities across rooms largely fails.** `src/analysis/shared_transfer_probe.py` trains a plain 3-class SIT/STAND/WALK classifier in one room:

  | Setting | Macro-F1 |
  |---|---:|
  | Same room | 0.97–0.99 |
  | Other room, A→B | 0.35 (chance 0.33) |
  | Other room, B→A | 0.03 |

  * B→A is *systematically* inverted: Room-A SIT and WALK are mostly predicted as STAND, and STAND as SIT.
  * This matches challenge limitations 1 and 2 (room/session confound; activity–location confound in Experiment 2). The static posture/location signature of an activity does not carry over between the two recordings.
* **Routing.** It helps once K ≥ 5: mean +0.07 macro-F1 over K ∈ {5, 10, 20}, +0.04 averaged over all K, and up to +0.15 for A→B at K=20. Predicted routing equals oracle routing because the router is 100 % accurate. At K=1 the model without routing is better in both directions: a single target shot gives too little to anchor the room-specific feature space.
* **SSL contribution.** In Task 3, the masked-reconstruction encoder is *below* the same architecture with random weights (mean 0.292 vs 0.357). This is consistent with the Task 2 probe, where SSL scored 0.855, random 0.838–0.861 and handcrafted 0.955.
* **How the default was chosen.** On this validation comparison, which the challenge allows for design decisions, "dynamics" and the random-weight encoder are effectively tied: random is ahead by 0.002–0.008 on every seed. "dynamics" is the default because it is parameter-free and excludes the room fingerprint by construction.
* **Confusion matrices at K=5** (`results/task3/confusion_K5_target_{A,B}.png`, `per_class_K5_target_*.csv`): FALL, STAND-UP, SIT-DOWN, EMPTY and LYING are recognized from target shots, while the transferred SIT/STAND (and WALK for target A) are the main errors.

**Other variants tried and not adopted.** These came from an exploratory notebook-style run that is not part of the repository. None of them improved shared-activity transfer on validation:

* subcarrier-order-invariant motion/spectral features,
* CORAL covariance alignment,
* confidence-based self-training on unlabeled target development windows.

## 9. Outputs

| Path | Content |
|---|---|
| `results/preprocessing/summary.json`, `excluded_captures.csv` | Capture counts, QC counts, excluded captures |
| `results/task1/` | `metrics.csv`, `predictions.csv` (window_id, room, confidence, true_room), confusion matrix, accuracy-vs-length CSV/PNG |
| `results/task2/` | Member 2's original Task 2 outputs (PyTorch) |
| `results/encoder_diagnosis/` | Task 2 linear probe re-scored on the regenerated windows; t-SNE coloured by room and by activity |
| `results/analysis/` | Shared-activity cross-room transfer probe and its confusion counts |
| `results/task3/summary.csv` | `experiment, direction, target_room, K, extractor, classifier, routing, metric, value, std, n_seeds` |
| `results/task3/all_runs.csv` | Every individual run |
| `results/task3/*.png` | Label-efficiency curve, extractor / routing / classifier comparisons, K=5 confusion matrices |
| `results/task3/predictions_K5_seed0.csv` | `window_id, room, label, confidence` (+ true labels) for the headline system |
| `models/room_router.joblib`, `models/deploy.joblib` | Task 1 router; deployment models used by `run_episode.py` |

## 10. Leakage controls

* The original capture is the split unit. Room A uses the supplied validation folder. Room B's seed-42 split was created once, and `build_dataset.py` re-derives it and confirms it equals the committed manifest.
* Every learned parameter — normalization, feature alignment, scalers, classifiers and the router — is fitted on development windows only.
* Task 3 support comes only from target development captures, and shared-activity labels of the target room are never used.
* Model inputs are signal-only. Room is predicted from the window; file format determines only *how* a file is parsed.
* `src/task2/check_leakage.py` (Member 2's audit) checks for duplicate capture paths and identical windows across splits; it was re-run on the regenerated windows and again found none.
* One known deviation: 3 of the 150 official Room A validation captures cannot form a strict 150-observation window. They are excluded and reported as such.

## 11. Project structure

```text
configs/            experiment.yaml (single source of truth), split manifests, normalization stats
src/pipeline/       shared library: config, ingest (decode/QC/window/normalize), features,
                    router (Task 1), adaptation (Task 3), data loading
src/preprocessing/  build_dataset.py (one-command preprocessing), pcap_reader.py, csi_decode.py,
                    Member 1's original per-room scripts (superseded by build_dataset.py, kept for history)
src/task1/          room_identification.py
src/task2/          Member 2's SSL encoder, training and evaluation (PyTorch)
src/task3/          few_shot_adaptation.py
src/analysis/       encoder_diagnosis.py, shared_transfer_probe.py
scripts/            run_all.py, smoke_test.py, verify_decoder.py, make_synthetic_dataset.py
run_episode.py      end-to-end inference entry point
checkpoints/task2/  pretrained encoder
archive/            exploratory preprocessing scripts (not part of the pipeline)
```

## 12. Team contributions

* **Member 1 — Preprocessing:** CSI ingestion, Room A decoding, amplitude processing, dataset validation, windowing, normalization, capture-level split manifests.
* **Member 2 — Self-supervised learning:** CNN encoder, masked-reconstruction training, frozen linear probe, random-encoder and handcrafted baselines, leakage checks, t-SNE.
* **Member 3 — Integration, Task 1, Task 3, episode runner:** unified config-driven pipeline (`src/pipeline`, `build_dataset.py`), dependency-free PCAP reader and vectorized decoder, QC and provenance manifest, room router and fingerprint study, NumPy encoder inference, few-shot adaptation with routing study, shared-transfer probe, `run_episode.py`, smoke test, documentation.

## 13. Academic integrity and AI disclosure

The project follows the challenge restriction against using, translating or porting the source authors' MATLAB code, processed `.mat` files or LSTM notebook. None of those files are used or included.

AI assistance (Claude) was used for code drafting, debugging and documentation. Every reported result comes from the scripts in this repository, run locally on the challenge data. Team members are responsible for understanding and explaining every stage.

## 14. References

* CS60055 Ubiquitous Computing — Hackathon Challenge 2, *Cross-Room Few-Shot Adaptation*.
* Schäfer et al., *Human Activity Recognition Using CSI Information with Nexmon*, Applied Sciences 11(19):8860, 2021. https://doi.org/10.3390/app11198860
* CSI Human Activity dataset: https://ieee-dataport.org/open-access/csi-human-activity
* Nexmon CSI: https://github.com/seemoo-lab/nexmon_csi
* libpcap file format: https://wiki.wireshark.org/Development/LibpcapFileFormat
* Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS 2017.
* Sun & Saenko, *Deep CORAL*, ECCV-W 2016 (variant tried).
* scikit-learn, NumPy, pandas, matplotlib; PyTorch (Task 2).
