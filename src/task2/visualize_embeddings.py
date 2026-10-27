
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from dataset import get_window_paths
from model import CSIEncoder
from linear_probe import (
    extract_embeddings,
    ROOT,
    RESULTS,
    DEVICE
)


def plot_embeddings(points, labels, title, filename):

    plt.figure(figsize=(9, 7))

    for label in sorted(set(labels)):
        mask = labels == label

        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=12,
            alpha=0.65,
            label=label
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Load the pretrained encoder.
    encoder = CSIEncoder().to(DEVICE)

    encoder.load_state_dict(
        torch.load(
            ROOT / "checkpoints/task2/csi_encoder.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    encoder.requires_grad_(False)
    encoder.eval()

    # Use development data only.
    records = get_window_paths("development")

    print("Extracting encoder embeddings...")

    features, activities = extract_embeddings(
        encoder, records
    )

    rooms = np.array([
        room for _, room in records
    ])

    print("Embedding shape:", features.shape)

    # Compute one 2-D projection for both visualizations.
    print("Computing t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )

    points = tsne.fit_transform(features)

    plot_embeddings(
        points,
        rooms,
        "CSI Encoder Embeddings by Room",
        RESULTS / "tsne_by_room.png"
    )

    plot_embeddings(
        points,
        activities,
        "CSI Encoder Embeddings by Activity",
        RESULTS / "tsne_by_activity.png"
    )

    print("\nVisualizations saved:")
    print(RESULTS / "tsne_by_room.png")
    print(RESULTS / "tsne_by_activity.png")


if __name__ == "__main__":
    main()
