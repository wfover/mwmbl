"""XGBoost contextual-bandit model for source selection.

A pointwise ``XGBRegressor`` predicts the judge reward of querying a source
for the current query. Its input is the shared context feature vector
(``features.FEATURE_NAMES``) concatenated with a per-source identity one-hot
block (``src_{name}``): with source identity in the trees, the model can learn
interactions like intent_code x github or cos_bow x arxiv that a within-query
ranker is otherwise structurally blind to. Unknown sources get an all-zero
identity block and back off to the shared features.

There is no per-request update: the model is retrained in batch from
``SuperSearchImpression`` rows (which store per-source shared vectors keyed by
source name, plus judge rewards, and never the query text) or from an offline
``RewardMatrix``. Artifacts are a directory holding ``model.json`` (XGBoost's
native, version-stable format) and ``meta.json`` (source vocabulary, feature
names, provenance); writes are atomic (tmp + ``os.replace``, meta last) so
serving processes can hot-reload safely.

Loading is fail-fast by design: a corrupt artifact, a feature-set mismatch
with the running code, or no artifact anywhere raises a plain exception that
propagates out of source selection. The only quiet path is the designed one —
no artifact in the runtime dir yet (normal before the first online retrain),
in which case the repo-bundled warm-start artifact is used.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from django.conf import settings

from mwmbl.tinysearchengine.super_search_select.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MODEL_FILE = "model.json"
META_FILE = "meta.json"

# Repo-bundled warm-start artifact, committed inside the package so it ships
# with every deployment (unlike devdata, which may not be mounted).
BUNDLED_DIR = Path(__file__).parent / "artifacts" / "xgb"

# How often get_model() re-stats meta.json to pick up a retrain.
RELOAD_CHECK_SECONDS = 60.0

# Tuned offline via evaluation.evaluate_holdout; change only with offline evidence.
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": 0,
}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def build_vocab(names: Iterable[str]) -> list[str]:
    """Frozen, sorted source vocabulary for the identity one-hot block."""
    return sorted(set(names))


def full_feature_names(vocab: Sequence[str]) -> list[str]:
    return list(FEATURE_NAMES) + [f"src_{name}" for name in vocab]


def encode(shared: Sequence[float], source: str, vocab_index: dict[str, int]) -> np.ndarray:
    """shared context vector ++ source identity one-hot.

    Shared vectors shorter than ``NUM_FEATURES`` are zero-padded: features are
    only ever appended to ``FEATURE_NAMES``, so older logged vectors align
    exactly with zeros in the newer slots. A *longer* vector means the data
    came from a newer feature set than this code and is an error.
    """
    v = np.asarray(shared, dtype=np.float64)
    if v.ndim != 1 or v.size > NUM_FEATURES:
        raise ValueError(
            f"shared feature vector has {v.shape} shape; expected <= {NUM_FEATURES} values")
    x = np.zeros(NUM_FEATURES + len(vocab_index), dtype=np.float64)
    x[:v.size] = v
    idx = vocab_index.get(source)
    if idx is not None:
        x[NUM_FEATURES + idx] = 1.0
    return x


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def build_training_data(
    rows: Iterable[tuple[dict[str, Sequence[float]], dict[str, float]]],
    vocab: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) from impression rows: each row is (features by source, rewards by source).

    One training pair per source with both a stored feature vector and a
    reward — exactly what ``SuperSearchImpression`` logs per request.
    """
    vocab_index = {name: i for i, name in enumerate(vocab)}
    xs, ys = [], []
    for features, rewards in rows:
        for name, reward in rewards.items():
            shared = features.get(name)
            if shared is None:
                continue
            xs.append(encode(shared, name, vocab_index))
            ys.append(float(reward))
    if not xs:
        return (np.zeros((0, NUM_FEATURES + len(vocab))), np.zeros(0))
    return np.stack(xs), np.asarray(ys)


def build_training_data_from_matrix(matrix) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, y, vocab) from a dense offline ``evaluation.RewardMatrix`` (masked cells)."""
    if list(matrix.feature_names) != list(FEATURE_NAMES):
        raise ValueError(
            f"matrix feature names {matrix.feature_names} do not match the "
            f"running code's FEATURE_NAMES {list(FEATURE_NAMES)}; rebuild the matrix")
    vocab = build_vocab(matrix.sources)
    vocab_index = {name: i for i, name in enumerate(vocab)}
    xs, ys = [], []
    for q in range(matrix.X.shape[0]):
        for s in range(matrix.X.shape[1]):
            if matrix.mask[q, s]:
                xs.append(encode(matrix.X[q, s], matrix.sources[s], vocab_index))
                ys.append(float(matrix.R[q, s]))
    return np.stack(xs), np.asarray(ys), vocab


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None):
    from xgboost import XGBRegressor

    model = XGBRegressor(**(params or XGB_PARAMS))
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Artifact save / load
# ---------------------------------------------------------------------------

def save_artifact(
    model,
    vocab: Sequence[str],
    model_dir: str | Path,
    reward_kind: str,
    n_rows: int,
    metrics: dict | None = None,
) -> None:
    """Atomically write ``model.json`` + ``meta.json`` (meta last: its presence
    and mtime are the load/reload signal)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    # tmp name keeps the .json extension so xgboost writes the JSON format
    tmp_model = model_dir / ("tmp." + MODEL_FILE)
    model.save_model(tmp_model)
    os.replace(tmp_model, model_dir / MODEL_FILE)

    meta = {
        "format_version": FORMAT_VERSION,
        "shared_feature_names": list(FEATURE_NAMES),
        "source_vocab": list(vocab),
        "reward_kind": reward_kind,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(n_rows),
        "metrics": metrics or {},
    }
    tmp_meta = model_dir / (META_FILE + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, model_dir / META_FILE)
    logger.info("saved super-search xgb artifact to %s (%d rows, reward=%s)",
                model_dir, n_rows, reward_kind)


@dataclass
class XgbSourceModel:
    model: object            # fitted XGBRegressor
    vocab: list[str]
    vocab_index: dict[str, int]
    meta: dict

    def score(self, feats: dict[str, Sequence[float]]) -> dict[str, float]:
        """Predicted reward for each source given its shared feature vector."""
        if not feats:
            return {}
        names = list(feats)
        X = np.stack([encode(feats[n], n, self.vocab_index) for n in names])
        preds = self.model.predict(X)
        return {name: float(p) for name, p in zip(names, preds)}


def load_artifact(model_dir: str | Path) -> XgbSourceModel:
    """Load and validate an artifact directory. Raises on anything unexpected:
    missing files, unreadable JSON, format or feature-set mismatch."""
    from xgboost import XGBRegressor

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / META_FILE).read_text())
    if meta["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"artifact {model_dir} has format_version {meta['format_version']}, "
            f"expected {FORMAT_VERSION}")
    if meta["shared_feature_names"] != list(FEATURE_NAMES):
        raise ValueError(
            f"artifact {model_dir} was trained on features "
            f"{meta['shared_feature_names']} but the running code has "
            f"{list(FEATURE_NAMES)}; retrain the artifact")
    model = XGBRegressor()
    model.load_model(model_dir / MODEL_FILE)
    vocab = list(meta["source_vocab"])
    return XgbSourceModel(
        model=model,
        vocab=vocab,
        vocab_index={name: i for i, name in enumerate(vocab)},
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Serving singleton
# ---------------------------------------------------------------------------

_cached: XgbSourceModel | None = None
_cached_key: tuple[Path, float] | None = None
_next_check: float = 0.0
_lock = threading.Lock()


def _artifact_dir() -> Path:
    """Runtime dir if it holds an artifact (i.e. an online retrain has run),
    else the repo-bundled warm-start artifact."""
    runtime = Path(settings.SUPER_SEARCH_XGB_MODEL_DIR)
    if (runtime / META_FILE).exists():
        return runtime
    return BUNDLED_DIR


def get_model() -> XgbSourceModel:
    """Shared per-process model, hot-reloaded when a retrain replaces the artifact.

    Fail-fast: if no artifact exists anywhere or the artifact is invalid this
    raises (FileNotFoundError / ValueError / json errors) rather than falling
    back — a missing model is a deployment bug to fix, not a state to mask.
    """
    global _cached, _cached_key, _next_check
    now = time.monotonic()
    if _cached is not None and now < _next_check:
        return _cached
    with _lock:
        if _cached is not None and time.monotonic() < _next_check:
            return _cached
        model_dir = _artifact_dir()
        mtime = (model_dir / META_FILE).stat().st_mtime
        key = (model_dir, mtime)
        if key != _cached_key:
            _cached = load_artifact(model_dir)
            _cached_key = key
            logger.info("super-search xgb model loaded from %s (trained %s, %d sources)",
                        model_dir, _cached.meta.get("trained_at"), len(_cached.vocab))
        _next_check = time.monotonic() + RELOAD_CHECK_SECONDS
        return _cached


def reset_model_cache() -> None:
    """Drop the cached model (tests / after an in-process retrain)."""
    global _cached, _cached_key, _next_check
    with _lock:
        _cached = None
        _cached_key = None
        _next_check = 0.0
