"""
Tests for the incremental judge worker (_JudgeWorker) and the hybrid/straggler
scoring paths in _emit_final_results.
"""
import asyncio

import mwmbl.tinysearchengine.super_search as ss
from mwmbl.tinysearchengine.indexer import Document
from mwmbl.tinysearchengine.super_search_select.rewards import SelectionContext

QUERY = "testing"


def _doc(url: str) -> Document:
    return Document(title=f"Result about testing {url}", url=url, extract="testing extract")


async def _wait_for(condition, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not condition():
        assert asyncio.get_event_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


def _collector():
    events = []

    async def emit(event_type, data):
        events.append((event_type, data))

    return events, emit


def test_worker_scores_and_dedupes(monkeypatch):
    calls = []

    def fake_judge(query, docs):
        calls.append([d.url for d in docs])
        return [0.5] * len(docs)

    monkeypatch.setattr(ss, "_judge_score_docs", fake_judge)

    async def scenario():
        ctx = SelectionContext()
        worker = ss._JudgeWorker(QUERY, ctx)
        worker.submit([_doc("http://a"), _doc("http://b")])
        worker.submit([_doc("http://a"), _doc("http://c")])
        await _wait_for(lambda: len(ctx.judge_scores) == 3)
        await worker.aclose()
        return ctx

    ctx = asyncio.run(scenario())
    assert set(ctx.judge_scores) == {"http://a", "http://b", "http://c"}
    scored = [url for batch in calls for url in batch]
    assert scored.count("http://a") == 1


def test_worker_unavailable_falls_back_to_ltr(monkeypatch):
    monkeypatch.setattr(ss, "_judge_score_docs", lambda query, docs: None)
    monkeypatch.setattr(ss, "score_documents", lambda model, query, docs: [0.1] * len(docs))

    async def scenario():
        ctx = SelectionContext()
        worker = ss._JudgeWorker(QUERY, ctx)
        worker.submit([_doc("http://a")])
        await _wait_for(lambda: worker.unavailable)

        events, emit = _collector()
        all_docs = [_doc("http://a"), _doc("http://b")]
        await ss._emit_final_results(QUERY, all_docs, emit, [None], asyncio.Lock(), ctx,
                                     use_judge=True, judge_worker=worker)
        await worker.aclose()
        return ctx, events

    ctx, events = asyncio.run(scenario())
    # No judge scores recorded, so judge rewards stay on their survival fallback.
    assert ctx.judge_scores == {}
    assert [e for e, _ in events] == ["results"]


def test_final_frame_judges_stragglers_only(monkeypatch):
    calls = []

    def fake_judge(query, docs):
        calls.append([d.url for d in docs])
        return [0.9 if d.url == "http://b" else 0.2 for d in docs]

    monkeypatch.setattr(ss, "_judge_score_docs", fake_judge)

    async def scenario():
        ctx = SelectionContext()
        worker = ss._JudgeWorker(QUERY, ctx)
        worker.submit([_doc("http://a")])
        await _wait_for(lambda: "http://a" in ctx.judge_scores)

        events, emit = _collector()
        all_docs = [_doc("http://a"), _doc("http://b"), _doc("http://c")]
        await ss._emit_final_results(QUERY, all_docs, emit, [None], asyncio.Lock(), ctx,
                                     use_judge=True, judge_worker=worker)
        return ctx, events, calls

    ctx, events, calls = asyncio.run(scenario())
    assert set(ctx.judge_scores) == {"http://a", "http://b", "http://c"}
    # The final pass scored only the docs the worker had not already covered.
    assert calls[-1] == ["http://b", "http://c"]
    results = events[-1][1].results
    assert results[0].url == "http://b"


def test_hybrid_scores_mix_judge_and_ltr(monkeypatch):
    monkeypatch.setattr(ss, "score_documents", lambda model, query, docs: [0.3] * len(docs))

    async def scenario():
        ctx = SelectionContext(judge_scores={"http://a": 0.8})
        docs = [_doc("http://a"), _doc("http://b")]
        return await ss._hybrid_scores(QUERY, docs, ctx)

    scores = asyncio.run(scenario())
    assert scores == [0.8, 0.3]


def test_progressive_frame_feeds_worker(monkeypatch):
    monkeypatch.setattr(ss, "_judge_score_docs", lambda query, docs: [0.7] * len(docs))
    monkeypatch.setattr(ss, "score_documents", lambda model, query, docs: [0.1] * len(docs))

    async def scenario():
        ctx = SelectionContext()
        worker = ss._JudgeWorker(QUERY, ctx)
        events, emit = _collector()
        all_docs = [_doc("http://a"), _doc("http://b")]
        await ss._emit_final_results(QUERY, all_docs, emit, [None], asyncio.Lock(), ctx,
                                     judge_worker=worker)
        await _wait_for(lambda: len(ctx.judge_scores) == 2)
        await worker.aclose()
        return ctx, events

    ctx, events = asyncio.run(scenario())
    assert set(ctx.judge_scores) == {"http://a", "http://b"}
    assert [e for e, _ in events] == ["results"]
