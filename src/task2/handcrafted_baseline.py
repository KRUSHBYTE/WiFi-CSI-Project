
import csv
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report
)

from dataset import get_window_paths, load_window
from linear_probe import get_activity, ALL_LABELS, RESULTS


def extract_features(window):
    """Extract statistical features from a CSI window."""

    # Mean amplitude of each subcarrier.
    mean = np.mean(window, axis=0)

    # Temporal standard deviation per subcarrier.
    std = np.std(window, axis=0)

    # Average absolute change between consecutive observations.
    temporal_diff = np.mean(
        np.abs(np.diff(window, axis=0)),
        axis=0
    )

    features = np.concatenate([
        mean, std, temporal_diff
    ])

    return features.astype(np.float32)


def prepare_dataset(split):
    """Create handcrafted features and activity labels."""

    records = get_window_paths(split)

    features = []
    labels = []

    for path, room in records:
        window = load_window(path)

        features.append(extract_features(window))
        labels.append(get_activity(path, room))

    return np.stack(features), np.array(labels)


def main():

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Preparing development features...")
    X_train, y_train = prepare_dataset("development")

    print("\nPreparing validation features...")
    X_val, y_val = prepare_dataset("validation")

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # Same classifier configuration as the other baselines.
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

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=ALL_LABELS,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_val, predictions
    )

    print("\nHANDCRAFTED BASELINE RESULTS")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=ALL_LABELS,
            zero_division=0
        )
    )

    # Save evaluation results.
    with open(
        RESULTS / "handcrafted_metrics.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["macro_f1", macro_f1])
        writer.writerow(["balanced_accuracy", balanced_acc])

    print("\nResults saved.")


if __name__ == "__main__":
    main()
