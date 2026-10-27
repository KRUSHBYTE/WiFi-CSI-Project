# WiFi CSI — Cross-Room Few-Shot Human Activity Recognition

**CS60055 Ubiquitous Computing | Hackathon Challenge 2**

## 1. Project Overview

This project develops a Python-based WiFi Channel State Information (CSI) pipeline for environment identification, self-supervised representation learning, and cross-room few-shot human activity recognition.

The project uses the CSI Human Activity dataset, specifically:

- **Room A (Experiment 2):** FALL, SIT, SIT-DOWN, STAND, STAND-UP, WALK
- **Room B (Experiment 3):** EMPTY, LYING, SIT, STAND, WALK

The shared activities are SIT, STAND, and WALK.

The main objective is to learn useful activity representations from CSI data and transfer activity knowledge between environments with limited labeled examples.

## 2. Project Tasks

| Component | Description | Current status |
|---|---|---|
| Preprocessing | CSI reconstruction, cleaning, windowing, normalization | Implemented |
| Task 1 | Room A/B identification | Not yet verified in this repository |
| Task 2 | Self-supervised CSI representation learning | Implemented and evaluated |
| Task 3 | Cross-room few-shot adaptation | Pending integration |

## 3. Project Structure

```text
WiFi_CSI_Task2/
├── archive/
│   └── preprocessing_debug/
├── checkpoints/
│   └── task2/
│       ├── csi_encoder.pt
│       └── ssl_model.pt
├── configs/
│   ├── room_a_split.csv
│   ├── room_b_split.csv
│   ├── room_a_normalization.npz
│   ├── room_b_normalization.npz
│   └── short_captures.csv
├── data/                       # Excluded from Git
│   ├── strict_normalized/
│   └── normalized/
├── results/
│   ├── task1/
│   ├── task2/
│   └── task3/
├── scripts/
├── src/
│   ├── preprocessing/
│   ├── task1/
│   ├── task2/
│   └── task3/
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. Dataset and Preprocessing

Each eligible capture is converted into a NumPy CSI amplitude window.

**Model input specification:**

- Shape: `(150, 256)`
- Data type: `float32`
- 150 consecutive CSI observations
- 256 subcarriers

Room A is reconstructed from raw Nexmon PCAP recordings. Room B begins with amplitude CSV recordings.

The preprocessing implementation uses deterministic central windowing and room-specific normalization statistics derived from development data.

### Processed Dataset Counts

| Room | Development | Validation |
|---|---:|---:|
| Room A | 925 | 147 |
| Room B | 1,076 | 125 |
| **Total** | **2,001** | **272** |

**Important limitation:** Twelve Room A captures contain fewer than 150 valid observations. Nine belong to development and three to validation. The strict preprocessing pipeline currently excludes them pending an approved handling policy.

Therefore, the reported validation results use 272 windows rather than the complete official validation pool.

The preprocessing handover also flags the need to confirm that Room A normalization parameters were recalculated after the corrected CSI decoder was applied.

For additional details, see `src/preprocessing/README.md`.

## 5. Environment Setup

Development has been performed with Python 3.11.

The Task 2 training environment uses PyTorch with CUDA acceleration on an NVIDIA GeForce RTX 3050 Laptop GPU.

Install the preprocessing dependencies:

```powershell
py -3.11 -m pip install -r requirements.txt
```

Install or verify the ML dependencies:

```powershell
py -3.11 -m pip install scikit-learn joblib matplotlib
```

For CUDA-enabled PyTorch, follow the official installation instructions at https://pytorch.org/get-started/locally/ and select a build compatible with your GPU and NVIDIA driver.

Verify GPU support:

```powershell
py -3.11 -c "import torch; print(torch.cuda.is_available())"
```

The original preprocessing requirements file may not contain every machine-learning dependency. The final team environment specification should include the tested PyTorch/CUDA build.

## 6. Task 2: Self-Supervised Learning

A lightweight 1D convolutional neural network learns CSI representations using masked reconstruction.

During pretraining:

1. Load development windows from both rooms.
2. Randomly mask 20% of temporal observations.
3. Reconstruct the masked CSI values.
4. Calculate mean squared reconstruction error over masked positions.
5. Update the encoder and decoder without using room or activity labels.

After pretraining, the encoder is frozen and a linear activity classifier is trained on labeled development samples.

### Run Task 2

All commands are executed from the project root.

```powershell
py -3.11 src\task2\dataset.py
py -3.11 src\task2\model.py
py -3.11 src\task2\train_ssl.py
py -3.11 src\task2\linear_probe.py
py -3.11 src\task2\random_encoder_baseline.py
py -3.11 src\task2\handcrafted_baseline.py
py -3.11 src\task2\check_leakage.py
py -3.11 src\task2\visualize_embeddings.py
```

Further technical details are documented in `src/task2/README.md`.

## 7. Task 2 Experimental Results

The following results were obtained on the available 272 fixed validation windows.

| Method | Macro-F1 | Balanced accuracy |
|---|---:|---:|
| Self-supervised CNN encoder | 0.8546 | 0.8753 |
| Random-weight CNN encoder | 0.8611 | 0.8757 |
| Handcrafted statistical features | **0.9592** | **0.9648** |

The handcrafted-feature baseline achieved the highest macro-F1.

The self-supervised reconstruction objective reduced its training loss, but the learned encoder did not outperform the random-weight baseline on the reported linear-probe evaluation.

These results are validation measurements and are not claimed as unseen-test performance.

## 8. Representation Visualization

The self-supervised encoder produces 64-dimensional embeddings.

t-SNE was used to visualize the development embeddings in two dimensions.

Generated figures:

- `results/task2/tsne_by_room.png`
- `results/task2/tsne_by_activity.png`

The visualizations indicate substantial separation between environments and some activity-dependent clustering. These plots are qualitative diagnostics rather than direct measures of cross-room generalization.

## 9. Data Integrity and Limitations

The preliminary audit found no duplicate capture-relative paths or byte-identical windows between development and validation.

The original capture manifests also showed no duplicate capture IDs.

Remaining limitations include:

- Three Room A validation captures are unavailable under strict 150-observation windowing.
- Room A normalization consistency with the corrected decoder must be confirmed.
- The baseline uses separately fitted room normalization statistics.
- The original recording sessions may be confounded with physical-room identity.
- Task 3 cross-room transfer performance has not yet been established.

No validation windows are intentionally used for self-supervised pretraining or classifier fitting.

## 10. Repository and Reproducibility

The repository should contain source code, configuration, experiment results, small model checkpoints, and documentation.

Raw or processed CSI datasets must not be committed to GitHub.

Important generated checkpoints:

- `checkpoints/task2/ssl_model.pt`
- `checkpoints/task2/csi_encoder.pt`

The pretrained encoder is intended to be reused by the cross-room few-shot adaptation component.

## 11. Team Contributions

**Member 1 — Preprocessing**

Implemented CSI ingestion, Room A decoding, amplitude processing, dataset validation, windowing, normalization, and capture-level split manifests.

**Member 2 — Self-Supervised Learning**

Implemented the CNN encoder, masked-reconstruction training, frozen linear classifier, random encoder baseline, handcrafted-feature baseline, preliminary leakage checks, and t-SNE visualizations.

**Other components**

Room identification, few-shot adaptation, end-to-end inference, and final integration should be attributed to their implementing contributors when completed.

## 12. Academic Integrity and AI Disclosure

The project follows the challenge restriction against copying or translating the source authors' MATLAB preprocessing implementation.

AI-assisted code drafting, debugging, and documentation. Experimental results are based on the locally executed scripts and reported outputs.

## 13. References

- CS60055 Ubiquitous Computing — Hackathon Challenge 2.
- Schäfer et al., *Human Activity Recognition Using CSI Information with Nexmon*, Applied Sciences, 2021. https://doi.org/10.3390/app11198860
- CSI Human Activity Dataset: https://ieee-dataport.org/open-access/csi-human-activity
- PyTorch: https://pytorch.org/
- scikit-learn: https://scikit-learn.org/
