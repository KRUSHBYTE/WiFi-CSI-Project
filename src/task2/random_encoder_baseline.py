
import csv
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report
)

from dataset import get_window_paths
from model import CSIEncoder
from linear_probe import (
    extract_embeddings,
    ALL_LABELS,
    DEVICE,
    RESULTS
)


def main():

    torch.manual_seed(42)

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Initialize the same encoder with random weights.
    encoder = CSIEncoder().to(DEVICE)

    # Freeze random weights.
    encoder.requires_grad_(False)
    encoder.eval()

    development = get_window_paths("development")
    validation = get_window_paths("validation")

    print("\nExtracting random encoder embeddings...")

    X_train, y_train = extract_embeddings(
        encoder, development
    )

    X_val, y_val = extract_embeddings(
        encoder, validation
    )

    # Use the same linear classifier as the SSL evaluation.
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

    print("\nRANDOM ENCODER BASELINE")
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

    # Save baseline results.
    with open(
        RESULTS / "random_encoder_metrics.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["metric", "value"])
        writer.writerow(["macro_f1", macro_f1])
        writer.writerow(["balanced_accuracy", balanced_acc])

    print("\nBaseline results saved.")


if __name__ == "__main__":
    main()
