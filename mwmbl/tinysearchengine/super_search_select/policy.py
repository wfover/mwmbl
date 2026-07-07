"""Source-selection policy: pick which ~10 of ~100 sources to query.

Three interchangeable scorings over the same context features, chosen by
``SUPER_SEARCH_SELECTION_MODE``:

- **cosine** (default): greedy query-vs-profile cosine, with a few reserved
  slots for cold sites so new sources are discovered.
- **lints**: per-arm linear-Gaussian Thompson sampling, which explores cold
  arms automatically via their wide posterior.
- **xgb**: contextual bandit — an XGBoost model predicts each source's judge
  reward from the context features plus a source-identity one-hot, with
  epsilon-greedy exploration. Learns in batch from the impression log.

Whichever runs, the always-on global sources (own index, HN) are included for
free, and the feature vectors used for the decision are stashed on the
``SelectionContext`` so the reward attribution at request completion is
consistent with the action taken.
"""
from __future__ import annotations

import random

import numpy as np
from django.conf import settings

from mwmbl.tinysearchengine.super_search_select import bandit, profiles, rstats, xgb_model
from mwmbl.tinysearchengine.super_search_select.features import (
    QueryContext,
    cosine_relevance,
    feature_vector,
)
from mwmbl.tinysearchengine.super_search_select.registry import get_meta
from mwmbl.tinysearchengine.super_search_select.rewards import SelectionContext


def select_sources(
    query: str,
    source_names: list[str],
    k: int | None = None,
    ctx: SelectionContext | None = None,
) -> list[str]:
    """Return up to ``k`` source names to query for ``query``.

    If ``ctx`` is given, the feature vector each selected source was scored on is
    recorded in ``ctx.features`` for a consistent bandit update later.
    """
    if k is None:
        k = settings.SUPER_SEARCH_SOURCES_TO_QUERY
    if len(source_names) <= k:
        if ctx is not None:
            _record_features(ctx, query, source_names)
        return list(source_names)

    # Pinned sources are always queried: the global always-on sources plus any
    # high-value sources named in SUPER_SEARCH_FORCE_INCLUDE (the offline-chosen
    # sources that carry real gold relevance but a content-blind policy would skip).
    force_include = getattr(settings, "SUPER_SEARCH_FORCE_INCLUDE", []) or []
    pinned = [n for n in source_names if get_meta(n).always_on]
    pinned += [n for n in source_names if n in force_include and n not in pinned]
    selectable = [n for n in source_names if n not in set(pinned)]
    budget = max(k - len(pinned), 0)
    if budget == 0:
        chosen = pinned[:k]
        if ctx is not None:
            _record_features(ctx, query, chosen)
        return chosen

    bow, cng = profiles.get_query_vectors(query)
    qctx = QueryContext.build(query, bow, cng)
    profs = profiles.get_profiles(selectable)
    stats = rstats.get_stats(selectable)
    feats = {n: feature_vector(qctx, get_meta(n), profs[n], stats[n]) for n in selectable}

    mode = settings.SUPER_SEARCH_SELECTION_MODE
    if mode == "xgb":
        chosen = _select_xgb(selectable, feats, budget)
    elif mode == "lints":
        chosen = _select_bandit(selectable, feats, budget)
    elif mode == "cosine":
        chosen = _select_cosine(qctx, selectable, profs, budget)
    else:
        raise ValueError(f"unknown SUPER_SEARCH_SELECTION_MODE {mode!r}")

    if ctx is not None:
        pinned_stats = rstats.get_stats([n for n in pinned if n not in feats])
        for name in pinned + chosen:
            if name in feats:
                ctx.features[name] = feats[name].tolist()
            elif name not in ctx.features:
                # pinned sources weren't scored; compute their features too.
                ctx.features[name] = feature_vector(
                    qctx, get_meta(name), profs.get(name, (None, None)),
                    pinned_stats.get(name),
                ).tolist()

    return pinned + chosen


def _select_cosine(qctx, selectable, profs, budget) -> list[str]:
    warm = [n for n in selectable if profs[n][0] is not None]
    cold = [n for n in selectable if profs[n][0] is None]
    explore_n = min(getattr(settings, "SUPER_SEARCH_EXPLORE_FLOOR", 0), len(cold), budget)
    exploit_n = budget - explore_n
    warm.sort(key=lambda n: cosine_relevance(qctx, profs[n]), reverse=True)
    chosen = warm[:exploit_n]
    chosen += random.sample(cold, explore_n) if explore_n else []
    if len(chosen) < budget:
        remaining = [n for n in warm[exploit_n:] + cold if n not in set(chosen)]
        chosen += remaining[: budget - len(chosen)]
    return chosen


def _select_xgb(selectable, feats, budget) -> list[str]:
    """Epsilon-greedy over the XGBoost model's predicted rewards.

    Each of the ``budget`` slots independently explores with probability
    epsilon: the non-explore slots take the top-scored sources, the explore
    slots are filled uniformly at random from the rest (cold sources included,
    so this subsumes the cosine path's explore floor and keeps randomized
    pairs flowing into the training log).
    """
    scores = xgb_model.get_model().score(feats)
    ranked = sorted(selectable, key=lambda n: scores[n], reverse=True)
    epsilon = settings.SUPER_SEARCH_XGB_EPSILON
    n_explore = sum(random.random() < epsilon for _ in range(budget))
    chosen = ranked[:budget - n_explore]
    rest = ranked[budget - n_explore:]
    chosen += random.sample(rest, min(n_explore, len(rest)))
    return chosen


def _select_bandit(selectable, feats, budget) -> list[str]:
    rng = np.random.default_rng()
    states = bandit.get_states(selectable)
    scored = sorted(
        selectable,
        key=lambda n: bandit.sample_score(states[n], np.asarray(feats[n]), rng),
        reverse=True,
    )
    return scored[:budget]


def _record_features(ctx: SelectionContext, query: str, names: list[str]) -> None:
    """Compute and stash feature vectors for ``names`` (small-fanout / all-selected case)."""
    bow, cng = profiles.get_query_vectors(query)
    qctx = QueryContext.build(query, bow, cng)
    profs = profiles.get_profiles(names)
    stats = rstats.get_stats(names)
    for name in names:
        ctx.features[name] = feature_vector(
            qctx, get_meta(name), profs.get(name, (None, None)), stats.get(name)
        ).tolist()
