
from pathlib import Path
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Member 1's preprocessed datasets
DATA_PATHS = {
    "room_A": PROJECT_ROOT / "data" / "strict_normalized",
    "room_B": PROJECT_ROOT / "data" / "normalized",
}

WINDOW_SHAPE = (150, 256)


def get_window_paths(split="development"):
    """Collect CSI window paths from both rooms."""

    if split not in ("development", "validation"):
        raise ValueError("Invalid dataset split")

    records = []

    for room, root in DATA_PATHS.items():
        folder = root / split / room

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        paths = sorted(folder.rglob("*.npy"))

        for path in paths:
            records.append((path, room))

        print(f"{room} {split}: {len(paths)} windows")

    return records


def load_window(path):
    """Load and verify a single CSI window."""

    window = np.load(path, allow_pickle=False)

    if window.shape != WINDOW_SHAPE:
        raise ValueError(
            f"Invalid shape {window.shape}: {path}"
        )

    if window.dtype != np.float32:
        raise ValueError(
            f"Invalid dtype {window.dtype}: {path}"
        )

    if not np.isfinite(window).all():
        raise ValueError(f"Non-finite values: {path}")

    return window


if __name__ == "__main__":

    # Only development data is loaded for SSL training.
    development = get_window_paths("development")

    print(f"\nTotal development windows: {len(development)}")

    # Verify one sample from each room.
    for room in DATA_PATHS:
        path = next(
            p for p, r in development if r == room
        )

        window = load_window(path)

        print(f"\n{room}")
        print("Shape:", window.shape)
        print("Dtype:", window.dtype)
        print("Sample:", path.name)
