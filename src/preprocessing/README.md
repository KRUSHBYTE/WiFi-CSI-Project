# WiFi CSI Preprocessing

## Objective

Convert Experiment-2 (Room A) and Experiment-3 (Room B) recordings into fixed-size CSI amplitude windows for machine learning.

## Input Datasets

- Room A: 1,084 PCAP captures
- Room B: 1,201 CSV captures

## Preprocessing

**Room A**
1. Extract UDP payloads from PCAP files.
2. Validate packet structure.
3. Decode BCM4366c0 packed CSI.
4. Calculate CSI amplitude.
5. Select 150 consecutive observations.
6. Normalize using development-set statistics.

**Room B**
1. Read amplitude CSV files.
2. Validate 256 columns and numerical values.
3. Select the central 150 observations.
4. Normalize using development-set statistics.

## Output

Each processed window is a NumPy array with:

- Shape: `(150, 256)`
- Type: `float32`

## Dataset Splits

| Room | Development | Validation |
|---|---:|---:|
| A | 925 | 147 |
| B | 1,076 | 125 |

Room A preserves the supplied split. Room B uses a deterministic class-wise split with random seed 42.

## Known Exceptions

Twelve Room A captures have fewer than 150 valid CSI observations. They are excluded from the strict-window output pending an approved handling policy.

The original captures remain unchanged.

## Decoder Verification

The independent Python BCM4366c0 decoder was compared against `csiread`. Eight sampled packets from `fall10.pcap` showed agreement to six displayed decimal places following the packet-wise scaling correction.

## Reproducibility

Run the preprocessing scripts from the project root with the Python environment specified in `requirements.txt`.

Raw and processed datasets are excluded from GitHub.