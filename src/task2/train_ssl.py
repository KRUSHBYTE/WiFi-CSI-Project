
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import get_window_paths, load_window
from model import MaskedReconstructionModel


# Training configuration
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MASK_RATIO = 0.20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = PROJECT_ROOT / "checkpoints" / "task2"

torch.manual_seed(42)


class CSIDataset(Dataset):

    def __init__(self):
        # Use only development windows.
        self.records = get_window_paths("development")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, _room = self.records[index]

        # Return CSI values only, never labels.
        window = load_window(path)

        return torch.from_numpy(window)


def mask_csi(x):
    """Randomly mask complete time observations."""

    batch_size, time_steps, _ = x.shape

    mask = torch.rand(
        batch_size, time_steps,
        device=x.device
    ) < MASK_RATIO

    masked_x = x.clone()

    # Hide all 256 subcarriers at selected time steps.
    masked_x[mask] = 0.0

    return masked_x, mask


def train():

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = CSIDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = MaskedReconstructionModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Training device:", DEVICE)
    print("Total windows:", len(dataset))
    print("Total epochs:", EPOCHS)

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for x in loader:

            x = x.to(DEVICE)

            masked_x, mask = mask_csi(x)

            optimizer.zero_grad()

            reconstruction = model(masked_x)

            # Evaluate errors only at hidden positions.
            loss = F.mse_loss(
                reconstruction[mask],
                x[mask]
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {avg_loss:.6f}"
        )

    # Save the complete reconstruction model.
    torch.save(
        model.state_dict(),
        SAVE_DIR / "ssl_model.pt"
    )

    # Save encoder separately for downstream tasks.
    torch.save(
        model.encoder.state_dict(),
        SAVE_DIR / "csi_encoder.pt"
    )

    print("\nTraining completed.")
    print("Model saved to:", SAVE_DIR)


if __name__ == "__main__":
    train()