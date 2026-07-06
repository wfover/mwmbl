"""Tests for the XGBoost contextual-bandit source model and the xgb policy."""
import json

import fakeredis
import numpy as np
import pytest

from mwmbl.tinysearchengine.super_search_select import policy, profiles, xgb_model
from mwmbl.tinysearchengine.super_search_select.evaluation import RewardMatrix
from mwmbl.tinysearchengine.super_search_select.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
)
from mwmbl.tinysearchengine.super_search_select.rewards import SelectionContext

INTENT_CODE = FEATURE_NAMES.index("intent_code")

FAST_PARAMS = {**xgb_model.XGB_PARAMS, "n_estimators": 50}


@pytest.fixture(autouse=True)
def clean_model_cache():
    xgb_model.reset_model_cache()
    yield
    xgb_model.reset_model_cache()


def _shared(intent_code: float = 0.0) -> list[float]:
    x = [0.0] * NUM_FEATURES
    x[0] = 1.0  # bias
    x[INTENT_CODE] = intent_code
    return x


def _interaction_rows(n: int = 60):
    """Planted structure: 'github' pays off iff intent_code, 'recipes' iff not."""
    rows = []
    for i in range(n):
        code = float(i % 2)
        features = {"github": _shared(code), "recipes": _shared(code)}
        rewards = {"github": code, "recipes": 1.0 - code}
        rows.append((features, rewards))
    return rows


def _trained_model(tmp_path, sources=("github", "recipes"), reward_kind="test"):
    vocab = xgb_model.build_vocab(sources)
    X, y = xgb_model.build_training_data(_interaction_rows(), vocab)
    model = xgb_model.train(X, y, params=FAST_PARAMS)
    xgb_model.save_artifact(model, vocab, tmp_path, reward_kind=reward_kind, n_rows=len(y))
    return vocab


# ---------------------------------------------------------------------------
# Encoding / training data
# ---------------------------------------------------------------------------

def test_encode_shape_and_identity_block():
    vocab_index = {"a": 0, "b": 1}
    x = xgb_model.encode(_shared(), "b", vocab_index)
    assert x.shape == (NUM_FEATURES + 2,)
    assert x[NUM_FEATURES] == 0.0 and x[NUM_FEATURES + 1] == 1.0


def test_encode_unknown_source_gets_zero_block():
    x = xgb_model.encode(_shared(), "never_seen", {"a": 0})
    assert np.all(x[NUM_FEATURES:] == 0.0)


def test_encode_zero_pads_old_short_vectors():
    # Pre-intent impressions stored 10-dim vectors; the intent block was
    # appended last, so zero-padding aligns exactly.
    old = [1.0] * 10
    x = xgb_model.encode(old, "a", {"a": 0})
    assert np.all(x[:10] == 1.0)
    assert np.all(x[10:NUM_FEATURES] == 0.0)


def test_encode_rejects_vectors_from_a_newer_feature_set():
    with pytest.raises(ValueError):
        xgb_model.encode([0.0] * (NUM_FEATURES + 1), "a", {"a": 0})


def test_build_training_data_skips_rewards_without_features():
    rows = [({"a": _shared()}, {"a": 1.0, "phantom": 0.5})]
    X, y = xgb_model.build_training_data(rows, ["a", "phantom"])
    assert X.shape == (1, NUM_FEATURES + 2)
    assert y.tolist() == [1.0]


def test_build_training_data_from_matrix():
    Q, S = 3, 2
    X = np.zeros((Q, S, NUM_FEATURES))
    X[:, :, 0] = 1.0
    R = np.array([[1.0, 0.0], [0.5, 0.2], [0.0, 0.9]])
    mask = np.ones((Q, S), dtype=bool)
    mask[2, 0] = False
    matrix = RewardMatrix(queries=["q1", "q2", "q3"], sources=["b", "a"],
                          feature_names=list(FEATURE_NAMES), X=X, R=R, mask=mask)
    Xf, y, vocab = xgb_model.build_training_data_from_matrix(matrix)
    assert vocab == ["a", "b"]
    assert Xf.shape == (5, NUM_FEATURES + 2)
    assert set(y.tolist()) == {1.0, 0.0, 0.5, 0.2, 0.9}


def test_matrix_with_mismatched_features_raises():
    matrix = RewardMatrix(queries=["q"], sources=["a"], feature_names=["bias"],
                          X=np.zeros((1, 1, 1)), R=np.zeros((1, 1)),
                          mask=np.ones((1, 1), dtype=bool))
    with pytest.raises(ValueError, match="feature names"):
        xgb_model.build_training_data_from_matrix(matrix)


# ---------------------------------------------------------------------------
# Model learns identity x context interactions
# ---------------------------------------------------------------------------

def test_model_learns_intent_source_interaction(tmp_path):
    _trained_model(tmp_path)
    loaded = xgb_model.load_artifact(tmp_path)
    code_scores = loaded.score({"github": _shared(1.0), "recipes": _shared(1.0)})
    plain_scores = loaded.score({"github": _shared(0.0), "recipes": _shared(0.0)})
    assert code_scores["github"] > code_scores["recipes"]
    assert plain_scores["recipes"] > plain_scores["github"]


# ---------------------------------------------------------------------------
# Artifact save / load / get_model
# ---------------------------------------------------------------------------

def test_artifact_roundtrip_meta(tmp_path):
    vocab = _trained_model(tmp_path, reward_kind="judge")
    loaded = xgb_model.load_artifact(tmp_path)
    assert loaded.vocab == list(vocab)
    assert loaded.meta["reward_kind"] == "judge"
    assert loaded.meta["shared_feature_names"] == list(FEATURE_NAMES)
    assert loaded.meta["n_rows"] == 120


def test_load_artifact_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        xgb_model.load_artifact(tmp_path / "nothing_here")


def test_load_artifact_feature_mismatch_raises(tmp_path):
    _trained_model(tmp_path)
    meta_path = tmp_path / xgb_model.META_FILE
    meta = json.loads(meta_path.read_text())
    meta["shared_feature_names"] = meta["shared_feature_names"][:-1]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="retrain"):
        xgb_model.load_artifact(tmp_path)


def test_load_artifact_bad_format_version_raises(tmp_path):
    _trained_model(tmp_path)
    meta_path = tmp_path / xgb_model.META_FILE
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 999
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version"):
        xgb_model.load_artifact(tmp_path)


def test_get_model_uses_runtime_dir_over_bundle(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    _trained_model(runtime)
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_MODEL_DIR", str(runtime))
    model = xgb_model.get_model()
    assert model.vocab == ["github", "recipes"]


def test_get_model_falls_back_to_bundle_when_runtime_empty(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    _trained_model(bundle, sources=("bundled_source",))
    monkeypatch.setattr(xgb_model, "BUNDLED_DIR", bundle)
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_MODEL_DIR",
                        str(tmp_path / "empty_runtime"))
    assert xgb_model.get_model().vocab == ["bundled_source"]


def test_get_model_no_artifact_anywhere_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(xgb_model, "BUNDLED_DIR", tmp_path / "no_bundle")
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_MODEL_DIR",
                        str(tmp_path / "no_runtime"))
    with pytest.raises(FileNotFoundError):
        xgb_model.get_model()


def test_get_model_hot_reloads_after_retrain(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    _trained_model(runtime)
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_MODEL_DIR", str(runtime))
    monkeypatch.setattr(xgb_model, "RELOAD_CHECK_SECONDS", 0.0)
    assert xgb_model.get_model().vocab == ["github", "recipes"]

    _trained_model(runtime, sources=("github", "recipes", "arxiv"))
    meta_path = runtime / xgb_model.META_FILE
    # ensure the mtime moves even on coarse-grained filesystems
    stat = meta_path.stat()
    import os
    os.utime(meta_path, (stat.st_atime, stat.st_mtime + 1))
    assert xgb_model.get_model().vocab == ["arxiv", "github", "recipes"]


# ---------------------------------------------------------------------------
# xgb policy
# ---------------------------------------------------------------------------

@pytest.fixture
def xgb_policy_env(tmp_path, monkeypatch):
    """xgb selection mode with a toy artifact trained over the policy's sources."""
    monkeypatch.setattr(profiles, "_redis", fakeredis.FakeRedis())
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_SELECTION_MODE", "xgb")
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_MODEL_DIR", str(tmp_path))
    names = ["mwmbl", "hn"] + [f"site{i}" for i in range(20)]
    _trained_model(tmp_path, sources=names)
    return names


def test_policy_xgb_selects_and_records_features(xgb_policy_env, monkeypatch):
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_EPSILON", 0.0)
    ctx = SelectionContext()
    chosen = policy.select_sources("python testing tools", xgb_policy_env, k=5, ctx=ctx)
    assert len(chosen) == 5
    assert "mwmbl" in chosen and "hn" in chosen  # always-on included
    assert set(chosen) <= set(ctx.features)
    assert all(len(v) == NUM_FEATURES for v in ctx.features.values())


def test_policy_xgb_greedy_is_deterministic(xgb_policy_env, monkeypatch):
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_EPSILON", 0.0)
    runs = {tuple(policy.select_sources("some query", xgb_policy_env, k=6))
            for _ in range(5)}
    assert len(runs) == 1


def test_policy_xgb_full_explore_still_valid(xgb_policy_env, monkeypatch):
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_EPSILON", 1.0)
    seen = set()
    for _ in range(10):
        chosen = policy.select_sources("some query", xgb_policy_env, k=5)
        assert len(chosen) == 5
        assert len(set(chosen)) == 5
        assert "mwmbl" in chosen and "hn" in chosen
        seen.add(tuple(chosen))
    assert len(seen) > 1  # actually exploring


def test_policy_xgb_missing_artifact_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(profiles, "_redis", fakeredis.FakeRedis())
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_SELECTION_MODE", "xgb")
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_XGB_MODEL_DIR",
                        str(tmp_path / "empty"))
    monkeypatch.setattr(xgb_model, "BUNDLED_DIR", tmp_path / "no_bundle")
    names = ["mwmbl", "hn"] + [f"site{i}" for i in range(20)]
    with pytest.raises(FileNotFoundError):
        policy.select_sources("some query", names, k=5)


def test_policy_unknown_mode_raises(monkeypatch):
    monkeypatch.setattr(profiles, "_redis", fakeredis.FakeRedis())
    monkeypatch.setattr("django.conf.settings.SUPER_SEARCH_SELECTION_MODE", "bogus")
    names = ["mwmbl", "hn"] + [f"site{i}" for i in range(20)]
    with pytest.raises(ValueError, match="SUPER_SEARCH_SELECTION_MODE"):
        policy.select_sources("some query", names, k=5)


# ---------------------------------------------------------------------------
# Online retrain from the impression log
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_retrain_from_impressions_writes_artifact(tmp_path, monkeypatch):
    from mwmbl.models import SuperSearchImpression

    monkeypatch.setattr(xgb_model, "XGB_PARAMS", FAST_PARAMS)
    for features, rewards in _interaction_rows(30):
        SuperSearchImpression.objects.create(
            candidates=list(features), selected=list(features),
            features={k: list(v) for k, v in features.items()}, rewards=rewards)
    # A pre-intent row with a short (10-dim) vector must zero-pad, not break.
    SuperSearchImpression.objects.create(
        candidates=["old"], selected=["old"],
        features={"old": [1.0] * 10}, rewards={"old": 0.5})

    metrics = xgb_model.train_and_save_from_impressions(
        window_days=7, min_rows=10, out_dir=tmp_path)
    assert metrics is not None and "train_rmse" in metrics
    loaded = xgb_model.load_artifact(tmp_path)
    assert {"github", "recipes", "old"} <= set(loaded.vocab)
    assert loaded.meta["reward_kind"] == "judge"
    assert loaded.meta["n_rows"] == 61


@pytest.mark.django_db
def test_retrain_skips_below_min_rows(tmp_path):
    from mwmbl.models import SuperSearchImpression

    SuperSearchImpression.objects.create(
        candidates=["a"], selected=["a"],
        features={"a": [0.0] * NUM_FEATURES}, rewards={"a": 1.0})
    assert xgb_model.train_and_save_from_impressions(
        window_days=7, min_rows=100, out_dir=tmp_path) is None
    assert not (tmp_path / xgb_model.META_FILE).exists()


# ---------------------------------------------------------------------------
# Privacy tripwire: nothing vector-valued may enter the persisted features
# ---------------------------------------------------------------------------

def test_persisted_features_are_scalar_and_reconstruction_safe():
    """Impressions persist the feature vector per source; every entry must be a
    named scalar that cannot reconstruct the query. A name matching the pattern
    below suggests someone is about to log a raw projection/embedding — that
    needs conscious review, hence this tripwire."""
    import re

    from mwmbl.tinysearchengine.super_search_select.features import (
        QueryContext, feature_vector,
    )
    from mwmbl.tinysearchengine.super_search_select.registry import get_meta

    assert not any(re.search(r"proj|embed|vec|hash", name) for name in FEATURE_NAMES)
    qctx = QueryContext.build("some test query", np.zeros(64), np.zeros(64))
    x = feature_vector(qctx, get_meta("github"), (None, None))
    assert x.shape == (len(FEATURE_NAMES),)
