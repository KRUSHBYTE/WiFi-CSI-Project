"""Window -> feature vector extractors used by Task 1, Task 3 and run_episode.py.

* handcrafted : per-subcarrier temporal mean, standard deviation and mean absolute
                first difference (768 values) - same definition as Task 2's baseline.
* dynamics    : handcrafted without the per-subcarrier mean (512 values), i.e. no
                static-multipath fingerprint.
* ssl         : Task 2's pretrained CSIEncoder (64-d), frozen.
* random      : the same CSIEncoder architecture with random weights (seed 42).

The CSIEncoder is evaluated with a NumPy re-implementation of its forward pass
(Conv1d -> GroupNorm -> ReLU x3 -> global average pool). Weights are read from
the Task 2 checkpoint with torch when it is installed, otherwise with a small
reader for PyTorch's zip checkpoint format, so inference needs no GPU or torch.
`scripts/check_numpy_encoder.py` compares it with the PyTorch module.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np

# Layer specification of src/task2/model.py::CSIEncoder (index in nn.Sequential).
ENCODER_LAYERS = [  # (conv index, in, out, kernel, padding, groupnorm index)
    (0, 256, 128, 5, 2, 1),
    (3, 128, 128, 5, 2, 4),
    (6, 128, 64, 3, 1, 7),
]
GROUPS = 8
EPS = 1e-5


# ---------------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------------

def handcrafted_features(windows: np.ndarray) -> np.ndarray:
    """(N, T, 256) -> (N, 768)."""
    windows = np.asarray(windows, dtype=np.float64)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    if windows.shape[1] > 1:
        diff = np.abs(np.diff(windows, axis=1)).mean(axis=1)
    else:
        diff = np.zeros_like(mean)
    return np.concatenate([mean, std, diff], axis=1).astype(np.float32)


def dynamics_features(windows: np.ndarray) -> np.ndarray:
    """(N, T, 256) -> (N, 512): handcrafted features WITHOUT the per-subcarrier mean.

    The temporal mean of a room-normalized window is dominated by the room's static
    multipath (its fingerprint) and by where in the room the person is. Dropping it
    keeps only motion-related statistics (temporal std and mean |first difference|),
    which are the parts of the signal most likely to transfer across rooms.
    """
    return handcrafted_features(windows)[:, 256:]


# ---------------------------------------------------------------------------
# Checkpoint reading without torch
# ---------------------------------------------------------------------------

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "BFloat16Storage": None}


class _Storage:
    def __init__(self, dtype, key):
        self.dtype, self.key = dtype, key


def _rebuild_tensor(storage, offset, size, stride, *args):
    array = storage.array
    if not size:
        return array[offset].copy()
    itemsize = array.itemsize
    view = np.lib.stride_tricks.as_strided(
        array[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride))
    return np.array(view)


class _TorchZipUnpickler(pickle.Unpickler):
    def __init__(self, handle, archive: zipfile.ZipFile, prefix: str):
        super().__init__(handle)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name in _DTYPES:
            return name
        raise pickle.UnpicklingError(f"unsupported object in checkpoint: {module}.{name}")

    def persistent_load(self, pid):
        _, storage_type, key, _location, _numel = pid
        dtype = _DTYPES.get(storage_type)
        if dtype is None:
            raise pickle.UnpicklingError(f"unsupported storage {storage_type}")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        storage = _Storage(dtype, key)
        storage.array = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))
        return storage


def load_state_dict(path: str | Path) -> dict:
    """Return {name: np.ndarray} from a torch.save()'d state_dict."""
    path = Path(path)
    try:
        import torch  # noqa: F401
        state = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.detach().cpu().numpy() for k, v in state.items()}
    except ImportError:
        pass
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with archive.open(pkl) as handle:
            state = _TorchZipUnpickler(handle, archive, prefix).load()
    return {k: np.asarray(v) for k, v in state.items()}


def random_state_dict(seed: int = 42) -> dict:
    """PyTorch-default initialisation (kaiming_uniform a=sqrt(5) => U(+-1/sqrt(fan_in)))."""
    rng = np.random.default_rng(seed)
    state = {}
    for conv, cin, cout, k, _pad, gn in ENCODER_LAYERS:
        bound = 1.0 / np.sqrt(cin * k)
        state[f"features.{conv}.weight"] = rng.uniform(-bound, bound, (cout, cin, k)).astype(np.float32)
        state[f"features.{conv}.bias"] = rng.uniform(-bound, bound, cout).astype(np.float32)
        state[f"features.{gn}.weight"] = np.ones(cout, np.float32)
        state[f"features.{gn}.bias"] = np.zeros(cout, np.float32)
    return state


# ---------------------------------------------------------------------------
# NumPy CSIEncoder forward pass
# ---------------------------------------------------------------------------

class NumpyCSIEncoder:
    def __init__(self, state: dict):
        self.layers = []
        for conv, cin, cout, k, pad, gn in ENCODER_LAYERS:
            w = np.asarray(state[f"features.{conv}.weight"], dtype=np.float32)
            if w.shape != (cout, cin, k):
                raise ValueError(f"checkpoint layer {conv} has shape {w.shape}")
            self.layers.append((
                w, np.asarray(state[f"features.{conv}.bias"], np.float32), k, pad,
                np.asarray(state[f"features.{gn}.weight"], np.float32),
                np.asarray(state[f"features.{gn}.bias"], np.float32),
            ))

    @staticmethod
    def _conv1d(x, w, b, k, pad):
        # x: (N, C, L) -> (N, Cout, L) with 'same' zero padding.
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
        cols = np.lib.stride_tricks.sliding_window_view(x, k, axis=2)  # (N, C, L, k)
        return np.einsum("nclk,ock->nol", cols, w, optimize=True) + b[None, :, None]

    @staticmethod
    def _group_norm(x, gamma, beta):
        n, c, length = x.shape
        g = x.reshape(n, GROUPS, c // GROUPS, length)
        mean = g.mean(axis=(2, 3), keepdims=True)
        var = g.var(axis=(2, 3), keepdims=True)
        g = (g - mean) / np.sqrt(var + EPS)
        return g.reshape(n, c, length) * gamma[None, :, None] + beta[None, :, None]

    def __call__(self, windows: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """(N, 150, 256) -> (N, 64)."""
        windows = np.asarray(windows, dtype=np.float32)
        out = []
        for start in range(0, len(windows), batch_size):
            x = windows[start:start + batch_size].transpose(0, 2, 1)  # (N, 256, T)
            for w, b, k, pad, gamma, beta in self.layers:
                x = np.maximum(self._group_norm(self._conv1d(x, w, b, k, pad), gamma, beta), 0.0)
            out.append(x.mean(axis=2))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 64), np.float32)


class FeatureExtractor:
    """Callable (N,150,256) normalized windows -> (N, D) features."""

    def __init__(self, kind: str, cfg=None):
        self.kind = kind
        if kind == "handcrafted":
            self._fn = handcrafted_features
        elif kind == "dynamics":
            self._fn = dynamics_features
        elif kind == "ssl":
            self._fn = NumpyCSIEncoder(load_state_dict(cfg.path("encoder_checkpoint")))
        elif kind == "random":
            self._fn = NumpyCSIEncoder(random_state_dict(42))
        else:
            raise ValueError(f"unknown feature extractor {kind}")

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        return self._fn(windows)
