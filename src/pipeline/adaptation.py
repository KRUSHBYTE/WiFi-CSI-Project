"""Task 3 cross-room few-label adaptation (shared by experiments and run_episode.py).

For target room T and source room S (the other room):

* Shared activities (SIT, STAND, WALK) are learned ONLY from labeled source-room
  development captures - no target labels for them.
* Target-exclusive activities (A: SIT-DOWN, STAND-UP, FALL; B: EMPTY, LYING) are
  learned from K labeled target DEVELOPMENT captures per activity (one window per
  capture, so K captures = K shots). Validation captures are never support.
* Environment awareness ("routing"): every window is assigned a domain that decides
  (1) which development statistics standardize its CSI input, and
  (2) which per-domain feature statistics align its features
      (unsupervised alignment: mean/std of that domain's unlabeled development features).
  Domain = true room (oracle), Task-1 predicted room (predicted) or one pooled
  domain for everything (none = the same model without routing).

Feature alignment uses only unlabeled development windows of each room, which the
challenge allows for representation learning (Task 2 already pretrains on them).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score

from pipeline.config import ROOM_LABELS, other_room


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class PrototypeClassifier:
    """Nearest class mean in the aligned feature space (prototypical classifier)."""

    def fit(self, x, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        self.centroids_ = np.stack([x[y == c].mean(axis=0) for c in self.classes_])
        d = self._dist(x)
        self.temperature_ = float(np.median(d)) or 1.0
        return self

    def _dist(self, x):
        return np.sqrt(((x[:, None, :] - self.centroids_[None]) ** 2).sum(-1))

    def predict_proba(self, x):
        logits = -self._dist(x) / self.temperature_
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


def make_classifier(kind: str):
    if kind == "logreg":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
    if kind == "prototype":
        return PrototypeClassifier()
    raise ValueError(f"unknown classifier {kind}")


# ---------------------------------------------------------------------------
# Feature cache: features of every window under every normalization
# ---------------------------------------------------------------------------

class FeatureBank:
    """features[(norm_name)] -> (N, D) for one extractor; plus per-domain alignment stats."""

    def __init__(self, windows, meta, normalizers, extractor, alignment: bool = True):
        self.meta = meta
        self.alignment = alignment
        self.features = {}
        for name, norm in normalizers.items():
            normalized = np.stack([norm(w) for w in windows]) if len(windows) else windows
            self.features[name] = extractor(normalized)
        dev = (meta.split == "development").to_numpy()
        self.stats = {}
        for name in normalizers:
            rows = dev & ((meta.room == name).to_numpy() if name in ("A", "B") else True)
            self.stats[name] = feature_stats(self.features[name][rows])

    def get(self, index, domain):
        """Aligned features for window rows ``index`` interpreted in ``domain``."""
        x = self.features[domain][index]
        return align(x, self.stats[domain]) if self.alignment else x


def feature_stats(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, np.where(std < 1e-8, 1.0, std)


def align(x, stats):
    mean, std = stats
    return ((x - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Support selection, fitting and prediction
# ---------------------------------------------------------------------------

def select_support(meta, target, activities, k, seed):
    """K distinct target DEVELOPMENT captures per target-exclusive activity."""
    rng = np.random.default_rng(seed * 1009 + (0 if target == "A" else 1))
    chosen = []
    for activity in activities:
        pool = meta[(meta.room == target) & (meta.split == "development") & (meta.activity == activity)]
        pool = pool.sort_values("capture_id")
        if len(pool) < k:
            raise ValueError(f"room {target} has only {len(pool)} development captures of {activity}, K={k}")
        chosen.extend(pool.index[rng.choice(len(pool), size=k, replace=False)].tolist())
    return np.array(chosen, dtype=int)


def source_rows(meta, target, shared):
    source = other_room(target)
    mask = (meta.room == source) & (meta.split == "development") & (meta.activity.isin(shared))
    return np.flatnonzero(mask.to_numpy())


def fit_target_model(bank, target, support_index, shared, classifier, mode):
    """Train the adapted classifier for one target room."""
    source = other_room(target)
    src_domain = "pooled" if mode == "none" else source
    tgt_domain = "pooled" if mode == "none" else target
    src = source_rows(bank.meta, target, shared)
    x = np.concatenate([bank.get(src, src_domain), bank.get(support_index, tgt_domain)])
    y = np.concatenate([bank.meta.activity.to_numpy()[src], bank.meta.activity.to_numpy()[support_index]])
    model = make_classifier(classifier).fit(x, y)
    return {"target": target, "classifier": model, "labels": ROOM_LABELS[target],
            "support_index": support_index, "mode": mode}


def predict(bank, models, index, routed_rooms, mode):
    """Predict activities for rows ``index``.

    models: {"A": model for target A, "B": model for target B}
    routed_rooms: room per row used for routing (true room for oracle,
                  router output for predicted; ignored for none, which uses
                  the model of the evaluated target and the pooled domain).
    """
    labels = np.empty(len(index), dtype=object)
    confidence = np.zeros(len(index))
    routed_rooms = np.asarray(routed_rooms)
    for room in np.unique(routed_rooms):
        rows = np.flatnonzero(routed_rooms == room)
        domain = "pooled" if mode == "none" else room
        model = models[room]["classifier"]
        proba = model.predict_proba(bank.get(index[rows], domain))
        labels[rows] = model.classes_[proba.argmax(axis=1)]
        confidence[rows] = proba.max(axis=1)
    return labels, confidence


def activity_metrics(y_true, y_pred, labels, exclusive, shared):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    present = sorted(set(y_true))
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "target_exclusive_macro_f1": f1_score(y_true, y_pred, labels=exclusive, average="macro", zero_division=0),
        "shared_macro_f1": f1_score(y_true, y_pred, labels=shared, average="macro", zero_division=0),
        "balanced_accuracy": recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "accuracy": float(np.mean(y_true == y_pred)),
    }
