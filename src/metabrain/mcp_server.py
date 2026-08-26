"""An MCP server that exposes a metabrain database to any MCP client.

The library is the source of truth; this module is a thin, JSON-shaped skin over
it. Every tool calls one :class:`~metabrain.MetaBrain` method on a single shared
instance and returns plain dicts/lists, so a client never sees a dataclass.

Run it over stdio::

    metabrain-mcp --db ./agent.db

Register it with Claude Code::

    claude mcp add metabrain -- metabrain-mcp --db ./agent.db

Requires the optional extra: ``pip install 'metabrain[mcp]'``.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict
from typing import Any

from .db import _LEARNING_TYPES, MetaBrain

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _McpServer
except ModuleNotFoundError:  # pragma: no cover - depends on installed major
    try:  # mcp >= 2.0 renamed FastMCP to MCPServer
        from mcp.server.mcpserver import MCPServer as _McpServer
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "metabrain's MCP server needs the 'mcp' SDK. "
            "Install it with: pip install 'metabrain[mcp]'"
        ) from exc

__all__ = [
    "LEARNING_TYPES",
    "build_server",
    "capture_error",
    "hypotheses",
    "learn",
    "main",
    "open_brain",
    "recall",
    "set_brain",
    "start_brief",
    "stats",
    "TOOLS",
]

#: The learning types the library accepts, re-exported so validation cannot drift.
LEARNING_TYPES: tuple[str, ...] = tuple(sorted(_LEARNING_TYPES))

INSTRUCTIONS = (
    "metabrain is a memory layer that learns what works. Call start_brief at the "
    "beginning of a task to load proven preferences and open hypotheses, recall "
    "before acting, learn after discovering something durable, and verdict once a "
    "unit of work resolves so the learn -> hypothesis -> experiment loop turns."
)

_brain: MetaBrain | None = None


# -- shared instance ------------------------------------------------------


def set_brain(brain: MetaBrain | None) -> None:
    """Install the MetaBrain instance every tool reads and writes through."""
    global _brain
    _brain = brain


def open_brain(db_path: str) -> MetaBrain:
    """Open ``db_path``, install it as the shared instance, and return it."""
    brain = MetaBrain(db_path)
    set_brain(brain)
    return brain


def close_brain() -> None:
    """Close the shared instance if one is open, and forget it."""
    global _brain
    if _brain is not None:
        _brain.close()
        _brain = None


def _db() -> MetaBrain:
    if _brain is None:
        raise RuntimeError("no metabrain database is open; start the server with --db PATH")
    return _brain


# -- tools ----------------------------------------------------------------


def recall(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search stored lessons for the substring `query` and return up to `limit`
    matching lesson objects, newest first."""
    return [asdict(item) for item in _db().recall(query, limit=limit)]


def learn(
    type: str,
    insight: str,
    domain: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Record or reinforce one lesson from `type` (failure, pattern, gotcha, or
    preference), `insight`, and optional `domain` and `context`, returning the
    stored lesson."""
    if type not in _LEARNING_TYPES:
        raise ValueError(f"type must be one of {list(LEARNING_TYPES)}, got {type!r}")
    return asdict(_db().learn(type, insight, domain=domain, evidence=context))


def hypotheses(status: str | None = None) -> list[dict[str, Any]]:
    """List hypothesis objects, optionally filtered by `status` (testing,
    graduated, or rejected)."""
    return [asdict(item) for item in _db().hypotheses(status=status)]


def verdict(
    result: str,
    unit: str | None = None,
    evidence: str | None = None,
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """Record `result` ("pass" or "fail") against an optional `unit` or
    `hypothesis` with optional `evidence`, and return the stored entry (an
    experiment is written when a hypothesis is in play)."""
    return asdict(
        _db().verdict(result, unit=unit, hypothesis=hypothesis, evidence=evidence)
    )


def start_brief() -> dict[str, Any]:
    """Return the start-of-task digest (preferences, learnings, open_hypotheses,
    open_units, last_checkpoint, recent_errors), taking no arguments."""
    return asdict(_db().read_start())


def stats() -> dict[str, int]:
    """Return a mapping of every metabrain table name to its row count, taking no
    arguments."""
    return _db().stats()


def capture_error(tool: str, error: str, context: str | None = None) -> dict[str, Any]:
    """Record a failure from `tool` name, `error` text, and optional `context`,
    returning the stored error entry."""
    return asdict(_db().capture_error(tool, error, context=context))


#: The tool functions registered on the server, in the order they are exposed.
TOOLS = (recall, learn, hypotheses, verdict, start_brief, stats, capture_error)


# -- server ---------------------------------------------------------------


def build_server() -> Any:
    """Build the MCP server with every metabrain tool registered."""
    server = _McpServer(name="metabrain", instructions=INSTRUCTIONS)
    for fn in TOOLS:
        server.tool()(fn)
    return server


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``metabrain-mcp`` console script."""
    parser = argparse.ArgumentParser(
        prog="metabrain-mcp",
        description="Serve a metabrain database over MCP (stdio transport).",
    )
    parser.add_argument("--db", required=True, help="path to the metabrain SQLite file")
    args = parser.parse_args(argv)

    open_brain(args.db)
    atexit.register(close_brain)
    try:
        build_server().run(transport="stdio")
    finally:
        close_brain()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
