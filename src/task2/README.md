# Task 2 — Self-Supervised CSI Representation Learning

**CS60055 Ubiquitous Computing — Hackathon Challenge 2**

## 1. Objective

Task 2 investigates whether a neural network can learn useful representations from unlabeled WiFi CSI recordings.

The official requirements are:

1. Train a self-supervised encoder using development recordings from both rooms without using activity or room labels.
2. Freeze the pretrained encoder.
3. Fit a linear classifier using labeled development data.
4. Evaluate on the fixed validation dataset.
5. Compare against handcrafted features and the same randomly initialized encoder architecture.

The primary activity-recognition metric is macro-F1, with balanced accuracy also reported.

## 2. Input Dataset

The implementation uses Member 1's preprocessed NumPy arrays.

| Room | Development | Validation | Dataset root |
|---|---:|---:|---|
| A | 925 | 147 | `data/strict_normalized/` |
| B | 1,076 | 125 | `data/normalized/` |
| Total | 2,001 | 272 | — |

Each window has:

- Shape `(150, 256)`
- Data type `float32`
- Finite CSI amplitude values
- One original capture identity

Only development windows are used during self-supervised training.

Room and activity metadata are retained for permitted downstream classification and evaluation, but are not passed into the self-supervised training objective.

## 3. Implementation Files

| File | Purpose |
|---|---|
| `dataset.py` | Locate, load, and validate CSI windows |
| `model.py` | Define the CNN encoder and reconstruction model |
| `train_ssl.py` | Train the masked-reconstruction model |
| `linear_probe.py` | Evaluate a frozen pretrained encoder |
| `random_encoder_baseline.py` | Evaluate an untrained random encoder |
| `handcrafted_baseline.py` | Evaluate handcrafted statistical features |
| `check_leakage.py` | Preliminary development/validation overlap audit |
| `visualize_embeddings.py` | Generate t-SNE visualizations |

## 4. Self-Supervised Model Architecture

The encoder is a lightweight temporal 1D CNN.

The input CSI window has shape `(150, 256)`.

Before convolution, the tensor dimensions are rearranged so that the 256 subcarriers act as input channels and the 150 observations act as the temporal dimension.

### Encoder

| Layer | Configuration |
|---|---|
| Conv1D 1 | 256 → 128 channels, kernel 5, padding 2 |
| GroupNorm + ReLU | 8 groups |
| Conv1D 2 | 128 → 128 channels, kernel 5, padding 2 |
| GroupNorm + ReLU | 8 groups |
| Conv1D 3 | 128 → 64 channels, kernel 3, padding 1 |
| GroupNorm + ReLU | 8 groups |
| AdaptiveAvgPool1D | Temporal dimension → 1 |

**Final embedding size: 64 dimensions.**

The reconstruction model uses a 1×1 convolutional decoder that transforms the temporal feature map into 256 reconstructed CSI subcarrier channels.

## 5. Self-Supervised Training Method

### Masked reconstruction

For every training batch, 20% of the temporal positions are randomly selected.

At each selected time position, all 256 subcarrier values are replaced with zero.

The reconstruction model predicts CSI values from the masked input.

The training loss is mean squared error calculated only at masked positions:

\[
L_{\mathrm{mask}} =
\frac{1}{|M|}
\sum_{(t,s)\in M}
(\hat{x}_{t,s}-x_{t,s})^2
\]

where \(M\) represents the masked CSI positions.

No room labels or activity labels are used during encoder training.

### Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Device | NVIDIA RTX 3050 Laptop GPU |
| CUDA enabled | Yes |
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | AdamW |
| Mask ratio | 20% |
| Gradient clipping | Maximum norm 1.0 |
| Random seed | 42 |

### Training Loss

| Epoch | Reconstruction loss |
|---|---:|
| 1 | 0.673425 |
| 2 | 0.558528 |
| 3 | 0.543517 |
| 4 | 0.529142 |
| 5 | 0.525876 |
| 6 | 0.511604 |
| 7 | 0.509815 |
| 8 | 0.502548 |
| 9 | 0.503627 |
| 10 | 0.498258 |

The reconstruction loss decreased by approximately 26% from the first to the final epoch.

This shows improvement on the training objective, but does not by itself establish better activity recognition.

## 6. Frozen Linear-Probe Evaluation

After pretraining:

1. Load the pretrained encoder.
2. Freeze all encoder parameters.
3. Extract one 64-dimensional embedding per CSI window.
4. Fit a linear classifier using labeled development embeddings.
5. Evaluate on the fixed validation windows.

### Linear Classifier

The classifier uses:

- `StandardScaler`
- `LogisticRegression`
- `max_iter=2000`
- `class_weight="balanced"`
- `random_state=42`

Only the scaler and linear classifier are fitted using labeled development data.

The evaluation contains eight distinct activity labels:

EMPTY, FALL, LYING, SIT, SIT-DOWN, STAND, STAND-UP, and WALK.

### Results

**Macro-F1:** 0.8546

**Balanced accuracy:** 0.8753

Validation accuracy is approximately 85%.

### Per-Class F1

| Activity | F1 |
|---|---:|
| EMPTY | 0.91 |
| FALL | 0.91 |
| LYING | 0.94 |
| SIT | 0.71 |
| SIT-DOWN | 0.75 |
| STAND | 0.80 |
| STAND-UP | 0.93 |
| WALK | 0.90 |

The model performs comparatively well on LYING, STAND-UP, FALL, and EMPTY.

SIT and SIT-DOWN are more difficult, with lower recall and F1.

## 7. Random-Weight Encoder Baseline

A second encoder is created using the same CNN architecture, but with randomly initialized weights.

Its weights are frozen without self-supervised training.

The embeddings are evaluated using the same linear-classifier configuration.

**Results:**

- Macro-F1: 0.8611
- Balanced accuracy: 0.8757

The random encoder slightly outperforms the pretrained SSL encoder on this validation evaluation.

Therefore, the current experiment does not demonstrate a benefit from masked-reconstruction pretraining over random convolutional features.

## 8. Handcrafted-Feature Baseline

A non-neural baseline is implemented using three statistical measurements per subcarrier:

1. Mean amplitude across time.
2. Temporal standard deviation.
3. Mean absolute difference between consecutive observations.

Since there are 256 subcarriers:

\[
256 \times 3 = 768
\]

Each window produces 768 handcrafted features.

The features are standardized, and the same logistic-regression classifier configuration is used.

**Results:**

- Macro-F1: 0.9592
- Balanced accuracy: 0.9648

The handcrafted approach substantially outperforms both encoder-based approaches on the available validation dataset.

## 9. Model Comparison

| Method | Macro-F1 | Balanced accuracy |
|---|---:|---:|
| Pretrained SSL encoder | 0.8546 | 0.8753 |
| Random encoder | 0.8611 | 0.8757 |
| Handcrafted features | **0.9592** | **0.9648** |

The handcrafted features outperform the SSL encoder by 10.46 percentage points in macro-F1.

The results indicate that time-domain statistical properties of the CSI amplitude contain strong activity-related information.

However, this comparison does not establish performance in unseen rooms or recording sessions.

## 10. t-SNE Visualization

The pretrained encoder's 64-dimensional development embeddings are projected into two dimensions using t-SNE.

Parameters include:

- Perplexity: 30
- Random seed: 42
- PCA initialization
- Automatic learning rate

Generated images:

- `results/task2/tsne_by_room.png`
- `results/task2/tsne_by_activity.png`

### Room-Wise Interpretation

The embeddings form visibly separated regions for Rooms A and B.

This indicates substantial room/session-dependent structure in the representation.

### Activity-Wise Interpretation

Some activities show relatively distinct clusters, while SIT and STAND overlap more noticeably. Certain activities form multiple separated groups.

The plots suggest the representation contains both environment-specific and activity-related information.

The projections are qualitative and should not be used as independent proof of transfer performance.

## 11. Data Leakage and Evaluation Integrity

The preliminary check found:

| Test | Room A | Room B |
|---|---:|---:|
| Duplicate relative paths across splits | 0 | 0 |
| Identical CSI windows across splits | 0 | 0 |
| Duplicate capture IDs in manifests | 0 | 0 |

The preprocessing manifests preserve the original capture-level development and validation assignments.

These preliminary checks do not independently prove that normalization and all source-capture provenance are correct.

### Unresolved Issues

- Three official Room A validation captures have fewer than 150 valid observations and were excluded by strict windowing.
- Nine additional short Room A development captures were excluded.
- The Room A normalization file requires confirmation against the corrected decoder's strict-window output.
- Room A and B use room-specific normalization parameters.
- The validation results describe the available 272-window subset, not a complete official benchmark result.
- High performance with handcrafted features does not establish cross-room few-shot transfer performance.

## 12. Reproducing Task 2

Run the following commands from the project root.

### Verify dataset loading

```powershell
py -3.11 src\task2\dataset.py
```

### Test model architecture

```powershell
py -3.11 src\task2\model.py
```

### Train self-supervised encoder

```powershell
py -3.11 src\task2\train_ssl.py
```

### Run frozen linear probe

```powershell
py -3.11 src\task2\linear_probe.py
```

### Evaluate random encoder

```powershell
py -3.11 src\task2\random_encoder_baseline.py
```

### Evaluate handcrafted features

```powershell
py -3.11 src\task2\handcrafted_baseline.py
```

### Check preliminary data leakage

```powershell
py -3.11 src\task2\check_leakage.py
```

### Generate t-SNE visualizations

```powershell
py -3.11 src\task2\visualize_embeddings.py
```

## 13. Generated Artifacts

### Checkpoints

```text
checkpoints/task2/
├── ssl_model.pt
└── csi_encoder.pt
```

`ssl_model.pt` contains the full reconstruction model weights.

`csi_encoder.pt` contains the pretrained feature extractor weights.

### Evaluation Results

```text
results/task2/
├── linear_classifier.joblib
├── linear_probe_metrics.csv
├── linear_probe_predictions.csv
├── random_encoder_metrics.csv
├── handcrafted_metrics.csv
├── tsne_by_room.png
└── tsne_by_activity.png
```

## 14. Handover to Task 3

The downstream few-shot adaptation implementation can reuse the trained encoder.

Required components:

- `src/task2/model.py`
- `src/task2/dataset.py`
- `checkpoints/task2/csi_encoder.pt`
- Compatible preprocessing and normalization configuration
- The activity-label and capture-split manifests

### Example Encoder Loading

```python
import torch
from src.task2.model import CSIEncoder

encoder = CSIEncoder()

state = torch.load(
    "checkpoints/task2/csi_encoder.pt",
    map_location="cpu",
    weights_only=True
)

encoder.load_state_dict(state)
encoder.eval()
encoder.requires_grad_(False)
```

The encoder expects CSI windows of shape `(batch_size, 150, 256)`.

Its output has shape `(batch_size, 64)`.

The frozen encoder can provide representations for few-shot classifiers and cross-room adaptation experiments.

## 15. Conclusion

The Task 2 implementation provides a reproducible masked-reconstruction encoder, frozen linear-probe evaluation, two comparison baselines, and representation visualizations.

The key experimental finding is that handcrafted CSI statistics outperform the current self-supervised CNN representation on the available validation data.

The pretrained encoder remains available for subsequent transfer-learning experiments.

Future work should investigate whether improved representation-learning objectives, normalization strategies, or temporal feature aggregation improve cross-room transfer.

## 16. Contribution and AI Disclosure

Task 2 work covers dataset loading for modeling, CNN architecture, masked-reconstruction pretraining, linear-probe evaluation, random encoder evaluation, handcrafted features, preliminary leakage checks, and t-SNE visualizations.

AI-assisted code drafting, debugging, and documentation. Experimental results are based on the locally executed scripts and reported outputs.
