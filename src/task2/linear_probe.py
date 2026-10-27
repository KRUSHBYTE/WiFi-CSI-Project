
from pathlib import Path
import csv
import joblib
import numpy as np
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report
)

from dataset import get_window_paths, load_window
from model import CSIEncoder


ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = ROOT / "checkpoints/task2/csi_encoder.pt"
RESULTS = ROOT / "results/task2"

ROOM_A_LABELS = {
    "fall": "FALL",
    "sit": "SIT",
    "sitdown": "SIT-DOWN",
    "stand": "STAND",
    "standup": "STAND-UP",
    "walk": "WALK"
}

ROOM_B_LABELS = {
    "empty": "EMPTY",
    "lying": "LYING",
    "sitting": "SIT",
    "standing": "STAND",
    "walking": "WALK"
}

ALL_LABELS = sorted(
    set(ROOM_A_LABELS.values()) |
    set(ROOM_B_LABELS.values())
)


def get_activity(path, room):
    """Read the activity label for evaluation."""

    if room == "room_A":
        for part in reversed(path.parent.parts):
            key = part.lower().replace("-", "").replace("_", "")
            if key in ROOM_A_LABELS:
                return ROOM_A_LABELS[key]

    elif room == "room_B":
        key = path.stem.split("_")[0].lower()
        if key in ROOM_B_LABELS:
            return ROOM_B_LABELS[key]

    raise ValueError(f"Unknown activity label: {path}")


@torch.no_grad()
def extract_embeddings(encoder, records, batch_size=32):
    """Extract frozen encoder representations."""

    embeddings = []
    labels = []

    encoder.eval()

    for start in range(0, len(records), batch_size):

        batch_records = records[start:start + batch_size]

        windows = np.stack([
            load_window(path)
            for path, room in batch_records
        ])

        x = torch.from_numpy(windows).to(DEVICE)

        features = encoder(x)

        embeddings.append(features.cpu().numpy())

        labels.extend([
            get_activity(path, room)
            for path, room in batch_records
        ])

    return np.concatenate(embeddings), np.array(labels)


def main():

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)

    # Load the pretrained encoder.
    encoder = CSIEncoder().to(DEVICE)

    encoder.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True
        )
    )

    # Freeze all encoder parameters.
    encoder.requires_grad_(False)

    development = get_window_paths("development")
    validation = get_window_paths("validation")

    print("\nExtracting development embeddings...")
    X_train, y_train = extract_embeddings(
        encoder, development
    )

    print("Extracting validation embeddings...")
    X_val, y_val = extract_embeddings(
        encoder, validation
    )

    print("\nTraining embeddings:", X_train.shape)
    print("Validation embeddings:", X_val.shape)
    print("Activities:", ALL_LABELS)

    # Train only the linear classifier.
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_val)
    probabilities = classifier.predict_proba(X_val)

    macro_f1 = f1_score(
        y_val, predictions,
        labels=ALL_LABELS,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_val, predictions
    )

    print("\nTASK 2 LINEAR PROBE RESULTS")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print("\nPer-class results:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=ALL_LABELS,
            zero_division=0
        )
    )

    # Save the fitted classifier.
    joblib.dump(
        classifier,
        RESULTS / "linear_classifier.joblib"
    )

    # Save metrics.
    with open(RESULTS / "linear_probe_metrics.csv",
              "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["macro_f1", macro_f1])
        writer.writerow(["balanced_accuracy", balanced_acc])

    # Save validation predictions.
    with open(RESULTS / "linear_probe_predictions.csv",
              "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow([
            "window_id", "room", "label",
            "confidence", "true_label"
        ])

        for i, (path, room) in enumerate(validation):
            writer.writerow([
                str(path.relative_to(ROOT)),
                room,
                predictions[i],
                float(probabilities[i].max()),
                y_val[i]
            ])

    print("\nResults saved:", RESULTS)


if __name__ == "__main__":
    main()
