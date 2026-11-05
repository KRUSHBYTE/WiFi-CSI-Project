"""Configuration loading, project paths and label definitions shared by all stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"

ROOMS = ("A", "B")

# Activity vocabulary per room. Labels come from the capture manifests
# (configs/room_*_split.csv), never from signal values.
ROOM_LABELS = {
    "A": ["FALL", "SIT", "SIT-DOWN", "STAND", "STAND-UP", "WALK"],
    "B": ["EMPTY", "LYING", "SIT", "STAND", "WALK"],
}
ALL_LABELS = sorted(set(ROOM_LABELS["A"]) | set(ROOM_LABELS["B"]))

# Folder names of the source datasets -> canonical labels.
ROOM_A_FOLDERS = {
    "fall": "FALL",
    "sit": "SIT",
    "sitdown": "SIT-DOWN",
    "stand": "STAND",
    "standup": "STAND-UP",
    "walk": "WALK",
}
ROOM_B_PREFIXES = {
    "EMPTY": "Empty",
    "LYING": "Lying",
    "SIT": "Sitting",
    "STAND": "Standing",
    "WALK": "Walking",
}


class Config(dict):
    """dict with attribute-style access to the top-level sections."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as error:
            raise AttributeError(item) from error

    def path(self, key: str) -> Path:
        value = Path(self["paths"][key])
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def preprocessing_hash(self) -> str:
        blob = json.dumps(self["preprocessing"], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    # Allow a small override file to inherit from the main config.
    base = data.pop("inherit", None)
    if base:
        parent = load_config((path.parent / base).resolve())
        merged = _deep_merge(dict(parent), data)
        return Config(merged)
    return Config(data)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def other_room(room: str) -> str:
    return "B" if room == "A" else "A"
