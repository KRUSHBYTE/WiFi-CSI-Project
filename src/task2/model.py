
import torch
import torch.nn as nn


class CSIEncoder(nn.Module):
    """Extract a 64-dimensional representation from a CSI window."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def temporal_features(self, x):
        # Convert (batch, time, subcarriers)
        # into (batch, subcarriers, time).
        x = x.transpose(1, 2)

        return self.features(x)

    def forward(self, x):
        features = self.temporal_features(x)

        # Aggregate temporal features into an embedding.
        embedding = self.pool(features).squeeze(-1)

        return embedding


class MaskedReconstructionModel(nn.Module):
    """Reconstruct CSI data from masked inputs."""

    def __init__(self):
        super().__init__()

        self.encoder = CSIEncoder()

        self.decoder = nn.Conv1d(
            in_channels=64,
            out_channels=256,
            kernel_size=1
        )

    def forward(self, x):
        features = self.encoder.temporal_features(x)

        reconstructed = self.decoder(features)

        return reconstructed.transpose(1, 2)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MaskedReconstructionModel().to(device)

    sample = torch.randn(2, 150, 256).to(device)

    reconstruction = model(sample)
    embeddings = model.encoder(sample)

    print("Device:", device)
    print("Input:", sample.shape)
    print("Reconstruction:", reconstruction.shape)
    print("Embedding:", embeddings.shape)