"""Tests for the MCP server skin over MetaBrain.

Every tool is called directly against a temp database (no subprocess), and one
end-to-end test spawns ``metabrain-mcp`` over stdio and lists its tools with the
real MCP client, which is the only way to prove the wiring a client will hit.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from metabrain import MetaBrain

mcp_server = pytest.importorskip(
    "metabrain.mcp_server", reason="needs the optional 'mcp' extra"
)

TOOL_NAMES = {
    "recall",
    "learn",
    "hypotheses",
    "verdict",
    "start_brief",
    "stats",
    "capture_error",
}


@pytest.fixture()
def brain(tmp_path):
    db = MetaBrain(str(tmp_path / "agent.db"))
    mcp_server.set_brain(db)
    try:
        yield db
    finally:
        mcp_server.set_brain(None)
        db.close()


def _json_roundtrip(value):
    """Every tool return must survive JSON serialization untouched."""
    return json.loads(json.dumps(value))


# -- individual tools -----------------------------------------------------


def test_learn_then_recall_round_trips(brain):
    made = mcp_server.learn("gotcha", "WAL mode needs a busy timeout", domain="sqlite")
    assert _json_roundtrip(made)["insight"] == "WAL mode needs a busy timeout"

    hits = mcp_server.recall("busy timeout")
    assert [h["insight"] for h in _json_roundtrip(hits)] == [
        "WAL mode needs a busy timeout"
    ]
    assert hits[0]["domain"] == "sqlite"


def test_learn_maps_context_onto_evidence(brain):
    made = mcp_server.learn("pattern", "short prompts win", context="3 sessions")
    assert made["evidence"] == "3 sessions"


def test_learn_rejects_an_unknown_type(brain):
    with pytest.raises(ValueError, match="type must be one of"):
        mcp_server.learn("hunch", "not a real type")
    assert mcp_server.recall("not a real type") == []


def test_recall_honours_limit(brain):
    for i in range(5):
        mcp_server.learn("pattern", f"limit probe {i}")
    assert len(mcp_server.recall("limit probe", limit=2)) == 2


def test_hypotheses_lists_a_promoted_pattern(brain):
    for _ in range(3):
        mcp_server.learn("pattern", "question hooks lift saves", domain="ig")

    everything = _json_roundtrip(mcp_server.hypotheses())
    assert [h["statement"] for h in everything] == ["question hooks lift saves"]
    assert everything[0]["status"] == "testing"
    assert mcp_server.hypotheses(status="graduated") == []
    assert len(mcp_server.hypotheses(status="testing")) == 1


def test_verdict_records_an_experiment_against_a_hypothesis(brain):
    for _ in range(3):
        mcp_server.learn("pattern", "carousels outperform stills", domain="ig")
    hyp = mcp_server.hypotheses(status="testing")[0]["id"]

    entry = _json_roundtrip(
        mcp_server.verdict("pass", hypothesis=hyp, evidence="1,240 saves")
    )
    assert entry["type"] == "verdict"
    assert entry["content"] == {"result": "pass", "evidence": "1,240 saves"}

    assert brain.experiments(hypothesis=hyp)[0].result == "supports"
    assert mcp_server.hypotheses(status="testing")[0]["evidence_for"] == 1


def test_verdict_on_a_unit(brain):
    unit = brain.unit("ship the MCP server")
    entry = mcp_server.verdict("fail", unit=unit, evidence="tests red")
    assert entry["unit_id"] == unit
    assert _json_roundtrip(entry)["content"] == {"result": "fail", "evidence": "tests red"}


def test_verdict_rejects_a_bad_result(brain):
    with pytest.raises(ValueError, match="result must be one of"):
        mcp_server.verdict("maybe")


def test_start_brief_surfaces_preferences_and_open_work(brain):
    mcp_server.learn("preference", "always recall first")
    mcp_server.learn("gotcha", "sdists leak absolute paths")
    brain.unit("an unfinished unit")

    out = _json_roundtrip(mcp_server.start_brief())
    assert [p["insight"] for p in out["preferences"]] == ["always recall first"]
    assert "sdists leak absolute paths" in [l["insight"] for l in out["learnings"]]
    assert [u["content"] for u in out["open_units"]] == ["an unfinished unit"]
    assert out["last_checkpoint"] is None
    assert out["recent_errors"] == []


def test_stats_counts_rows(brain):
    before = mcp_server.stats()
    mcp_server.learn("failure", "counted me")
    after = mcp_server.stats()

    assert set(after) >= {"learnings", "hypotheses", "experiments", "errors"}
    assert after["learnings"] == before["learnings"] + 1
    assert _json_roundtrip(after) == after


def test_capture_error_records_and_surfaces(brain):
    rec = _json_roundtrip(
        mcp_server.capture_error("pytest", "boom", context="while building")
    )
    assert rec["tool"] == "pytest"
    assert rec["error"] == "boom"
    assert rec["context"] == "while building"

    assert mcp_server.start_brief()["recent_errors"][0]["error"] == "boom"


def test_tools_refuse_to_run_without_an_open_database():
    mcp_server.set_brain(None)
    with pytest.raises(RuntimeError, match="no metabrain database is open"):
        mcp_server.stats()


def test_the_library_types_are_the_validation_source():
    from metabrain.db import _LEARNING_TYPES

    assert set(mcp_server.LEARNING_TYPES) == set(_LEARNING_TYPES)


def test_build_server_registers_every_tool(brain):
    server = mcp_server.build_server()
    listed = {t.name for t in _list_tools_sync(server)}
    assert listed == TOOL_NAMES


def _list_tools_sync(server):
    """``list_tools`` is sync in mcp 1.x and 2.x alike; tolerate a coroutine anyway."""
    import inspect

    result = server.list_tools()
    if inspect.isawaitable(result):
        import anyio

        async def _await():
            return await result

        return anyio.run(_await)
    return result


def _errored(result) -> bool:
    """``CallToolResult.isError`` in mcp 1.x, ``.is_error`` in 2.x."""
    return bool(getattr(result, "is_error", False) or getattr(result, "isError", False))


# -- end to end over stdio ------------------------------------------------


def test_stdio_server_lists_its_tools(tmp_path):
    """Spawn the real console script and speak MCP to it over stdio."""
    anyio = pytest.importorskip("anyio")
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    db_path = tmp_path / "stdio.db"
    MetaBrain(str(db_path)).close()

    # Prefer the installed console script, which is what a client actually spawns;
    # fall back to -m so the test still runs from a plain source checkout.
    script = pathlib.Path(sys.executable).parent / "metabrain-mcp"
    if script.exists():
        params = StdioServerParameters(command=str(script), args=["--db", str(db_path)])
    else:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "metabrain.mcp_server", "--db", str(db_path)],
        )

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = {t.name for t in listed.tools}

                called = await session.call_tool(
                    "learn", {"type": "pattern", "insight": "spoken over stdio"}
                )
                recalled = await session.call_tool("recall", {"query": "over stdio"})
                return names, called, recalled

    names, called, recalled = anyio.run(run)
    assert names == TOOL_NAMES
    assert not _errored(called)
    assert not _errored(recalled)
    assert "spoken over stdio" in recalled.content[0].text
