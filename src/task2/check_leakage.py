
import hashlib
from dataset import DATA_PATHS, get_window_paths, load_window


def check_leakage():
    development = get_window_paths("development")
    validation = get_window_paths("validation")

    for room in DATA_PATHS:
        dev = [p for p, r in development if r == room]
        val = [p for p, r in validation if r == room]

        def capture_id(path, split):
            root = DATA_PATHS[room] / split / room
            return str(path.relative_to(root)).lower()

        dev_ids = {
            capture_id(p, "development") for p in dev
        }

        val_ids = {
            capture_id(p, "validation") for p in val
        }

        overlap = dev_ids & val_ids

        def window_hash(path):
            x = load_window(path)
            return hashlib.sha256(x.tobytes()).hexdigest()

        dev_hashes = {window_hash(p) for p in dev}
        val_hashes = {window_hash(p) for p in val}

        duplicate_windows = dev_hashes & val_hashes

        print(f"\n{room}")
        print("Development:", len(dev))
        print("Validation:", len(val))
        print("Overlapping capture paths:", len(overlap))
        print("Identical CSI windows:", len(duplicate_windows))


if __name__ == "__main__":
    check_leakage()
