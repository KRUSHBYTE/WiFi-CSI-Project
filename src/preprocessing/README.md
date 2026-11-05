# WiFi CSI Preprocessing — Cross-Room Human Activity Recognition

> **Update (integration).** The per-room scripts below have been consolidated into one
> config-driven command, `python src/preprocessing/build_dataset.py`. It uses the same
> decoding, splits, centre windows and normalization, but replaces Scapy with
> `pcap_reader.py` and adds QC counts and a provenance manifest (`data/windows/manifest.csv`).
> Results on the full dataset:
>
> * Identical counts: 925/147 Room A windows, 1,076/125 Room B windows, 12 short captures.
> * The 2 rejected packets are the 2 truncated pcap records.
> * The regenerated Room A and Room B normalization statistics are identical to the committed
>   ones, so the "recalculate Room A normalization" item below is resolved.
> * Short captures stay excluded from benchmark windows.
>
> The scripts below are kept for history and still need Scapy.

## 1. Overview

This repository contains the Python preprocessing work for a cross-room, few-shot human activity recognition project using WiFi Channel State Information (CSI). It prepares Experiment-2 (Room A) and Experiment-3 (Room B) recordings for the machine learning pipeline developed by other team members.

**Preprocessing deliverable:** one amplitude window per eligible capture, stored as a NumPy array with shape `(150, 256)` and dtype `float32`, together with capture metadata, split manifests, and normalization parameters.

This README documents the preprocessing contribution only; it does not claim completion of the room classifier, self-supervised encoder, or few-shot adaptation experiments.

## 2. Input datasets

| Source | Room | Original format | Captures | Classes |
|---|---|---|---:|---|
| Experiment-2 | A | Nexmon CSI `.pcap` | 1,084 | FALL, SIT, SIT-DOWN, STAND, STAND-UP, WALK |
| Experiment-3 | B | CSI amplitude `.csv` | 1,201 | EMPTY, LYING, SIT, STAND, WALK |

Place the original permitted recordings under:

```text
data/raw/
├── Experiment-2/
│   ├── ReadMe.pdf
│   └── realdata/
└── Experiment-3/
    ├── ReadMe.pdf
    └── Data/
```

The source archives may also include authors' MATLAB code, processed results, and a classification notebook. These are **not used** in the preprocessing implementation. Keep raw data unchanged and out of version control.

## 3. Environment setup

The development environment used macOS, Python 3.11, VS Code, Git, and a Python virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project uses NumPy, pandas, Scapy, and other scientific Python packages. `csiread` was installed for *independent decoder comparison*, not as the submitted PCAP decoding implementation.

## 4. Preprocessing workflow

```text
Room A: PCAP → UDP packet validation → packed CSI reconstruction → amplitude
                                                                  |
Room B: CSV → header handling → numerical validation → amplitude --+
                                                                  |
                           capture-level split → windowing → normalization
                                                                  |
                        (150, 256) float32 NumPy arrays + metadata
```

### 4.1 Room A: Experiment-2

1. Enumerate `.pcap` files while retaining original relative paths and capture identities.
2. Read PCAP packets using Scapy.
3. Inspect UDP payload size and Nexmon magic bytes.
4. Extract the 18-byte metadata header and the following 1,024-byte CSI payload from structurally compatible packets.
5. Interpret the CSI region as 256 little-endian packed 32-bit words.
6. Independently reconstruct real and imaginary CSI components using the BCM4366c0 packed-field layout and packet-wise scaling.
7. Calculate amplitude using `abs(complex_csi)` and export capture-level amplitude CSV files.
8. Select 150 consecutive observations using deterministic central windowing where sufficient observations exist.
9. Normalize using statistics fitted on Room A development windows only.

The earlier `mantissa × 2^exponent` decoder was exploratory and is **not** the current reconstruction method. The revised packet-wise scaled implementation was checked against `csiread` on eight sampled packets from `fall10.pcap` and matched to six displayed decimal places for correlation, fitted scale, and relative error. This sample-level agreement does not constitute exhaustive validation of every PCAP capture.

### 4.2 Room B: Experiment-3

1. Read each CSV with its first row treated as column headings.
2. Accept different observed header-name styles (e.g., integer indices and `Subcarrier_*` names).
3. Require exactly 256 numeric columns and finite measurements.
4. Export consistently named amplitude columns in the original row order.
5. Select the central 150 observations from each 500-row capture.
6. Normalize using Room B development-set statistics, then apply those same statistics to its validation windows.

The observed Room B inspection found 1,201 CSV files with 500 data rows and 256 numeric columns after header handling, with no missing or non-finite measurements in the initial validation.

## 5. Data-quality audit

### Experiment-2 PCAP audit

| Measure | Observed |
|---|---:|
| Original captures | 1,084 |
| Packets read | 234,021 |
| Structurally accepted packets | 234,019 |
| Rejected packets | 2 |
| Captures with at least 150 accepted observations | 1,072 |
| Captures shorter than 150 observations | 12 |

Scapy reported two truncated-packet warnings during the audit. Their exact relationship to the two rejected packets has not been established. The audit is saved in `results/pcap_audit.csv` (locally; the `results/` directory is ignored by Git).

### Short Room A captures

The 12 short captures are distributed as follows:

| Split | Short captures |
|---|---:|
| Development | 9 |
| Validation | 3 |

An early exploratory windowing version used edge padding for these captures. **The strict-window version skips them**, because the project specifies 150 consecutive *valid* CSI observations. The original files and padded experimental outputs have not been deleted.

**Unresolved benchmark issue:** The official Room A validation split specifies 150 captures, but strict 150-observation windows currently exist for only 147. The three short validation captures require a decision from the teaching team (for example, a formally approved special handling rule). Do not silently treat the 147-capture subset as the complete official benchmark.

## 6. Development and validation splits

| Room | Original development | Original validation | Strict available development windows | Strict available validation windows |
|---|---:|---:|---:|---:|
| A | 934 | 150 | 925 | 147 |
| B | 1,076 | 125 | 1,076 | 125 |

- **Room A:** Preserve the supplied original validation split. Manifest: `configs/room_a_split.csv`.
- **Room B:** Sort candidate filenames and sample 25 captures per activity for validation with Python random seed `42`. Manifest: `configs/room_b_split.csv`.
- Split at the **original capture** level; never distribute windows from a single capture across development and validation.
- Never fit development-data normalization parameters using validation captures.
- Avoid using source filenames or directory metadata as classifier input features. Keep them only for provenance and permitted labels/splits.

## 7. Windowing and normalization

**Window size:** 150 consecutive CSI observations by 256 subcarriers.

**Baseline selection:** deterministic center window. For a capture of length `N ≥ 150`, the start index is `(N - 150) // 2`.

**Short captures:** excluded from strict Room A output pending an approved handling decision. The output array shape alone is not evidence of 150 real observations.

**Standard normalization:** per-subcarrier mean and standard deviation are computed from the applicable room's development windows. The same statistics are used when transforming that room's validation windows. Near-zero standard deviations are replaced by 1 to avoid division by zero.

```text
normalized = (amplitude - development_mean) / development_std
```

Normalization parameter files:

```text
configs/room_a_normalization.npz
configs/room_b_normalization.npz
```

**Note:** `room_a_normalization.npz` must be recalculated from the strict Room A development windows after the corrected decoder has been used to regenerate them. The corresponding normalizer was fitted independently for each room as a preprocessing baseline; the ML team should assess whether this choice is appropriate for the cross-room transfer protocol, especially if target-room normalization would use statistics not permitted by a particular training scenario.

## 8. Project scripts

| Script | Purpose |
|---|---|
| `inspect_dataset.py` | Enumerate dataset files and extensions |
| `inspect_csv.py` | Inspect example Room B recording |
| `validate_csv.py` | Check Room B column count and numerical validity |
| `inspect_pcap.py` | Inspect PCAP packet structure |
| `check_pcap_structure.py` | Check payload lengths and header markers |
| `audit_pcap_dataset.py` | Generate full Room A packet/capture audit |
| `csi_decode.py` | Independent packed CSI reconstruction and amplitude calculation |
| `compare_decoder.py` | Reference comparison using `csiread` |
| `export_room_a.py` | Export Room A capture-level amplitude CSVs |
| `export_room_b.py` | Export Room B capture-level amplitude CSVs |
| `split_room_a.py` | Preserve Room A capture-level split |
| `split_room_b.py` | Create deterministic Room B capture-level split |
| `window_room_a.py` | Create strict Room A center windows; skip short captures |
| `window_room_b.py` | Create Room B center windows |
| `normalize_room_a.py` | Fit/apply Room A development-data normalization |
| `normalize_room_b.py` | Fit/apply Room B development-data normalization |

Other experimental inspection or plotting scripts may remain in the directory; they are not required for downstream model input.

## 9. Reproduce the preprocessing

Run commands from the **project root** after activating `.venv` and arranging the input dataset as above.

```bash
# Validate and audit source data
python src/preprocessing/validate_csv.py
python src/preprocessing/audit_pcap_dataset.py

# Generate capture-level split manifests
python src/preprocessing/split_room_a.py
python src/preprocessing/split_room_b.py

# Export common amplitude representations
python src/preprocessing/export_room_a.py
python src/preprocessing/export_room_b.py

# Generate windows
python src/preprocessing/window_room_a.py
python src/preprocessing/window_room_b.py

# Normalize using development data only
python src/preprocessing/normalize_room_a.py
python src/preprocessing/normalize_room_b.py
```

Before executing the commands, ensure `window_room_a.py` is configured with `data/strict_candidate_windows` and skips short captures, and `normalize_room_a.py` reads from that directory and writes to `data/strict_normalized`. The ordered commands regenerate Room A with the updated decoder. Do not use old exploratory candidate/padded outputs as final training inputs.

## 10. Output structure and loading

The current processed outputs are organized as:

```text
data/
├── strict_normalized/
│   ├── development/room_A/.../*.npy
│   └── validation/room_A/.../*.npy
└── normalized/
    ├── development/room_B/*.npy
    └── validation/room_B/*.npy
```

All emitted windows should be `float32`, have shape `(150, 256)`, and contain finite values. The files use original capture-derived paths/names. Room A preserves nested source paths; Room B uses source filename stems.

Example loading:

```python
from pathlib import Path
import numpy as np

path = next(Path("data/strict_normalized/development/room_A").rglob("*.npy"))
window = np.load(path)
assert window.shape == (150, 256)
assert window.dtype == np.float32
```

Read class labels and split assignments from the CSV manifests rather than attempting to infer them from normalized signal values.

The earlier structural validation reported the expected counts and numerical properties of the available output windows. The strict Room A output does **not** yet represent the complete official validation set, for the reason described above.

## 11. Handover to the ML team

Provide:

1. This preprocessing source code and the two split manifests.
2. Processed `.npy` windows from both rooms, transferred through an approved shared storage location (not GitHub).
3. Room-specific normalization parameter `.npz` files.
4. `results/pcap_audit.csv` and the list of short captures.
5. A note that the strict Room A validation subset currently has 147 instead of 150 windows and must not be reported as the complete official evaluation without an approved policy.
6. The decoder verification limitation and the need to preserve the same preprocessing configuration when evaluating different model variants.

The ML team handles environment classification, self-supervised learning, few-shot adaptation, and reporting model accuracy/macro-F1/balanced accuracy. Preprocessing variants (such as robust normalization or motion-aware windows) are **proposed future experiments**, not implemented or accuracy-validated results in this baseline.

## 12. Version control and restrictions

The `.gitignore` excludes datasets, virtual environments, generated models, and large local results. Do not upload raw or processed CSI captures to the repository. Keep original data read-only. Use independent Python preprocessing rather than the dataset authors' MATLAB scripts or original processed training outputs. Cite research or external tools used for implementation or verification as required by the assignment.

## 13. References

- Project challenge document: *Cross-Room Few-Shot Adaptation* (CS60055 Ubiquitous Computing Hackathon Challenge 2).
- Nexmon CSI format and tooling: https://github.com/seemoo-lab/nexmon_csi
- CSI Human Activity dataset: https://ieee-dataport.org/open-access/csi-human-activity
- `csiread` (decoder verification): https://github.com/citysu/csiread

## 14. Current status

**Implemented/observed:** Both data formats inspected, CSV validation passed, Room A packet audit performed, candidate decoder corrected and spot-checked against `csiread`, both capture-level manifests created, and baseline windowing/normalization scripts run.

**Outstanding:** Obtain an approved treatment for the 12 short Room A captures (including three official validation captures), and ensure any final reported experiment uses a documented, permitted normalization protocol. This README does not claim that model-training results or research-based preprocessing improvements have been measured.
