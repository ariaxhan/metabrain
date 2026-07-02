"""The :class:`MetaBrain` memory layer and its self-improving loop.

A thin, typed wrapper over a single SQLite file. Design goals:

- **Library-first.** Import it and call methods that return objects; no
  subprocess, no required CLI.
- **Zero dependencies.** Only the standard-library ``sqlite3``.
- **Deterministic population.** Using the API correctly fills every table as a
  side effect, sessions on open, events on every write, hypotheses and
  experiments as the learn/verdict loop turns. Nothing relies on the caller
  remembering to log.
- **Safe under concurrency.** WAL plus ``busy_timeout`` let multiple agent
  processes share one file; an in-process lock guards the single connection so
  one instance is safe across threads.
- **Injection-proof.** Every value is bound as a parameter.

The loop is the point::

    learn("pattern", insight)  →  recurs (hit_count ≥ promote_at)
        →  graduates to a hypothesis (status: testing)
        →  each verdict on it is an experiment (supports / refutes)
        →  confidence ≥ graduate_at over a min sample
        →  re-emitted as a learning of type 'preference', a *proven* rule.

That is what separates metabrain from a store that only remembers what you told
it: it discovers what actually works and promotes it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import (
    ContextEntry,
    ErrorRecord,
    Experiment,
    Hypothesis,
    Learning,
    StartBrief,
)
from .schema import apply_schema

# Default vocabularies. Validation is *advisory*: callers may pass other values
# (the DB has no closed CHECK set), but a typo in a known verb should still be
# caught early, so the explicit verbs validate against these.
_LEARNING_TYPES = frozenset({"failure", "pattern", "gotcha", "preference"})
_VERDICTS = frozenset({"pass", "fail"})
_UNIT_KINDS = frozenset({"spec", "contract"})

# Thresholds, mined from 5,066 real learnings (not guessed):
#   hit_count sits at 1-2 for 57% of patterns, then drops sharply, the
#   recurring tail begins at 3. So a pattern observed 3× is worth testing.
DEFAULT_PROMOTE_AT = 3
#   confidence has no historical data (experiments never ran in the bash era),
#   so 0.8 is an honest default; require a minimum sample so 1/1 can't graduate.
DEFAULT_GRADUATE_AT = 0.8
DEFAULT_MIN_EXPERIMENTS = 3

# Tolerance so a hypothesis sitting exactly on a threshold (e.g. confidence 0.2
# vs a float-underflowed 0.199999…) lands on the intended side of the line.
_EPS = 1e-9


def _now() -> str:
    """Current time as an ISO-8601 UTC string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _new_memory_event_id(source_db: str | None) -> str:
    """A globally-unique memory_events id shaped ``<db-short>-<ts>-<rand>``.

    The short tag is the origin db's basename when known, so an event id is
    self-describing across federated stores; the ts + random suffix guarantee
    uniqueness even for many events in the same millisecond.
    """
    short = "local"
    if source_db:
        base = Path(source_db).name or source_db
        cleaned = "".join(c for c in base if c.isalnum() or c in "-_")
        short = cleaned[:16] or "local"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{short}-{ts}-{uuid.uuid4().hex[:8]}"


def _dump(content: Any) -> str:
    """Serialize a payload to JSON; strings pass through so callers can store text."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


class MetaBrain:
    """A SQLite-backed memory store with a built-in learn → prove → graduate loop.

    Open one per project::

        db = MetaBrain("agent.db")
        with db.session(task="content") as s:
            s.learn("pattern", "question hooks lift saves", domain="ig")
            ...
        brief = db.read_start()           # proven preferences first

    The flat API works without a session too, writes attach to a lazily-opened
    *ambient* session, so ``sessions`` and ``events`` still populate::

        db.learn("gotcha", "WAL needed for concurrent agents", domain="db")

    The instance is a context manager and closes its connection on exit.
    """

    def __init__(
        self,
        path: str | Path = "agent.db",
        *,
        agent: str | None = None,
        promote_at: int = DEFAULT_PROMOTE_AT,
        graduate_at: float = DEFAULT_GRADUATE_AT,
        min_experiments: int = DEFAULT_MIN_EXPERIMENTS,
    ) -> None:
        """Open (creating if needed) the database at ``path``.

        ``agent`` is an optional default label stamped on entries that don't
        specify their own. ``promote_at`` / ``graduate_at`` / ``min_experiments``
        tune the self-improving loop.
        """
        if promote_at < 1:
            raise ValueError(f"promote_at must be >= 1, got {promote_at}")
        if min_experiments < 1:
            raise ValueError(f"min_experiments must be >= 1, got {min_experiments}")
        if not 0.5 < graduate_at <= 1.0:
            # Above 0.5 keeps the graduate and reject thresholds disjoint, so a
            # 50/50 coin-flip can never count as a "proven" preference.
            raise ValueError(f"graduate_at must be in (0.5, 1.0], got {graduate_at}")
        self.path = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        self.agent = agent
        self.promote_at = promote_at
        self.graduate_at = graduate_at
        self.min_experiments = min_experiments
        self._lock = threading.RLock()
        self._ambient_id: str | None = None
        # check_same_thread=False + our own lock makes one instance usable from
        # several threads; WAL + busy_timeout handle several processes.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            apply_schema(self._conn, _now())

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._ambient_id is not None:
                self._end_session(self._ambient_id, success=None, outcome=None, tokens=None)
                self._ambient_id = None
            self._conn.close()

    def __enter__(self) -> "MetaBrain":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- low-level helpers -------------------------------------------------

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _emit(
        self, kind: str, *, session_id: str | None, ref_id: str | None = None, data: Any = None
    ) -> None:
        """Append a telemetry event. Called by every write so ``events`` is total."""
        self._execute(
            "INSERT INTO events (ts, session_id, kind, ref_id, data) VALUES (?, ?, ?, ?, ?)",
            (_now(), session_id, kind, ref_id, _dump(data) if data is not None else None),
        )

    def _emit_memory_event(
        self,
        kind: str,
        *,
        subject_id: str | None = None,
        object_id: str | None = None,
        actor: str | None = None,
        source_db: str | None = None,
        source_row_id: str | None = None,
        import_batch_id: str | None = None,
        reason: str | None = None,
        inference_method: str | None = None,
        payload: Any = None,
    ) -> str:
        """Append an observation to the append-only ``memory_events`` spine.

        This is the memory-lifecycle log (design section 2): the ground truth the
        derived caches (hit_count, domain, archived state) are recomputed from.
        Never UPDATE, never DELETE; a correction is a new event. Returns the id.
        """
        event_id = _new_memory_event_id(source_db)
        self._execute(
            """INSERT INTO memory_events
               (event_id, ts, schema_version, kind, actor, source_db, source_row_id,
                import_batch_id, subject_id, object_id, reason, inference_method, payload)
               VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                _now(),
                kind,
                actor,
                source_db,
                source_row_id,
                import_batch_id,
                subject_id,
                object_id,
                reason,
                inference_method,
                _dump(payload) if payload is not None else None,
            ),
        )
        return event_id

    # -- sessions ----------------------------------------------------------

    def session(
        self,
        *,
        task: str | None = None,
        tier: int | None = None,
        agent: str | None = None,
        meta: Any = None,
    ) -> "Session":
        """Open a session. Use as a context manager so close is recorded::

            with db.session(task="lead-capture") as s:
                s.learn(...)
        """
        sid = self._open_session(task=task, tier=tier, agent=agent or self.agent, meta=meta)
        return Session(self, sid)

    def _open_session(
        self,
        *,
        task: str | None,
        tier: int | None,
        agent: str | None,
        meta: Any,
    ) -> str:
        sid = _new_id("sess")
        self._execute(
            """INSERT INTO sessions (id, started_at, task, tier, agent, meta)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (sid, _now(), task, tier, agent, _dump(meta) if meta is not None else None),
        )
        self._emit("session", session_id=sid, ref_id=sid, data={"event": "start", "task": task})
        return sid

    def _end_session(
        self,
        sid: str,
        *,
        success: bool | None,
        outcome: str | None,
        tokens: int | None,
    ) -> None:
        self._execute(
            """UPDATE sessions
               SET ended_at = ?, success = ?, outcome = COALESCE(?, outcome),
                   tokens_used = COALESCE(?, tokens_used)
               WHERE id = ?""",
            (_now(), None if success is None else int(success), outcome, tokens, sid),
        )
        self._emit("session", session_id=sid, ref_id=sid, data={"event": "end", "success": success})

    def _ambient(self) -> str:
        """The implicit session backing flat (session-less) API calls."""
        with self._lock:
            if self._ambient_id is None:
                self._ambient_id = self._open_session(
                    task="ambient", tier=None, agent=self.agent, meta=None
                )
            return self._ambient_id

    def _sid(self, session_id: str | None) -> str:
        return session_id if session_id is not None else self._ambient()

    # -- learnings (cross-session memory) ----------------------------------

    def learn(
        self,
        type: str,
        insight: str,
        *,
        evidence: str | None = None,
        domain: str | None = None,
        visibility: str = "agent",
        sensitivity: str = "low",
        session_id: str | None = None,
        hit_count: int | None = None,
        last_hit: str | None = None,
        source_db: str | None = None,
        source_row_id: str | None = None,
        import_batch_id: str | None = None,
    ) -> Learning:
        """Record (or reinforce) a durable lesson.

        Re-learning an identical ``(type, insight)`` bumps its ``hit_count``
        rather than inserting a duplicate, the same forcing function as
        :meth:`recall`. When a learning crosses ``promote_at`` it graduates
        into a hypothesis automatically.

        The observatory params make an import lossless and replayable (design
        1.1, 3.1/3.2). ``hit_count``/``last_hit`` seed the row with the source's
        real observed reinforcement instead of resetting it to 1. The provenance
        (``source_db``/``source_row_id``/``import_batch_id``) is stamped on the
        row and recorded in the origin ``memory_events`` observation, so the seed
        is an OBSERVATION, not a bare scalar. Passing any provenance marks the
        call an import: its origin event is ``imported`` and an exact-string
        collision becomes a ``merged`` event (both provenance chains, SUM rule),
        never a silent ``+1`` or ``MAX``.
        """
        if type not in _LEARNING_TYPES:
            raise ValueError(f"type must be one of {sorted(_LEARNING_TYPES)}, got {type!r}")
        if not insight or not insight.strip():
            raise ValueError("insight must be a non-empty string")
        sid = self._sid(session_id)
        is_import = bool(import_batch_id or source_db or source_row_id)
        seed_count = hit_count if hit_count is not None else 1
        if seed_count < 1:
            seed_count = 1
        existing = self._query(
            "SELECT * FROM learnings WHERE type = ? AND insight = ? LIMIT 1", (type, insight)
        )
        if existing:
            row = existing[0]
            now = last_hit or _now()
            if is_import:
                # Two distinct source rows with the identical insight collapse
                # into one: a MERGE, not a silent +1 and not a MAX. Preserve both
                # provenance chains + both observed counts in the event, and SUM
                # the reinforcement into the survivor (design 3.1 / 2.2). The
                # survivor is always the live existing row, so a chain cannot
                # form (every collision resolves to this single surviving id).
                survivor_seed = row["hit_count"]
                merged_count = survivor_seed + seed_count
                self._emit_memory_event(
                    "merged",
                    subject_id=row["id"],
                    object_id=source_row_id,
                    actor="metabrain-import",
                    source_db=source_db,
                    source_row_id=source_row_id,
                    import_batch_id=import_batch_id,
                    reason="exact-string dedup collision on import",
                    payload={
                        "survivor": {
                            "source_db": row["source_db"],
                            "source_row_id": row["source_row_id"],
                            "observed_hit_count": survivor_seed,
                        },
                        "merged_away": {
                            "source_db": source_db,
                            "source_row_id": source_row_id,
                            "observed_hit_count": seed_count,
                        },
                    },
                )
                self._execute(
                    """UPDATE learnings
                       SET hit_count = ?, last_hit = ?,
                           evidence = COALESCE(?, evidence), domain = COALESCE(?, domain)
                       WHERE id = ?""",
                    (merged_count, now, evidence, domain, row["id"]),
                )
            else:
                # A live re-learn of the identical insight is a genuine
                # reinforcement: bump by one and record it as an observation so
                # hit_count stays replayable event-by-event going forward.
                self._execute(
                    """UPDATE learnings
                       SET hit_count = hit_count + 1, last_hit = ?,
                           evidence = COALESCE(?, evidence), domain = COALESCE(?, domain)
                       WHERE id = ?""",
                    (now, evidence, domain, row["id"]),
                )
                self._emit_memory_event(
                    "reinforced",
                    subject_id=row["id"],
                    actor=self.agent,
                    reason="live re-learn of identical insight",
                )
            self._emit("learn", session_id=sid, ref_id=row["id"], data={"reinforced": True})
            self._maybe_promote(row["id"])
            return self.get_learning(row["id"])  # type: ignore[return-value]

        lid = _new_id("learn")
        now = _now()
        seed_last_hit = last_hit or now
        self._execute(
            """INSERT INTO learnings
               (id, ts, session_id, type, insight, evidence, domain,
                hit_count, last_hit, visibility, sensitivity,
                source_db, source_row_id, import_batch_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                lid, now, sid, type, insight, evidence, domain,
                seed_count, seed_last_hit, visibility, sensitivity,
                source_db, source_row_id, import_batch_id,
            ),
        )
        # Origin observation: the seed hit_count recorded AS AN EVENT (design 1.1,
        # kill 1). 'imported' for a rebuild import, 'created' for a live learn.
        self._emit_memory_event(
            "imported" if is_import else "created",
            subject_id=lid,
            actor="metabrain-import" if is_import else self.agent,
            source_db=source_db,
            source_row_id=source_row_id,
            import_batch_id=import_batch_id,
            reason="seed import of pre-repair hit_count" if is_import else "live learn",
            inference_method=(
                "seed: pre-repair hit_count, no per-event history available"
                if is_import
                else None
            ),
            payload={"observed_hit_count": seed_count, "type": type, "domain": domain},
        )
        self._emit("learn", session_id=sid, ref_id=lid, data={"type": type})
        self._maybe_promote(lid)
        return self.get_learning(lid)  # type: ignore[return-value]

    def get_learning(self, learning_id: str) -> Learning | None:
        rows = self._query("SELECT * FROM learnings WHERE id = ?", (learning_id,))
        return Learning.from_row(rows[0]) if rows else None

    def learnings(
        self,
        *,
        type: str | None = None,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[Learning]:
        """Fetch stored learnings, newest first, optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._query(f"SELECT * FROM learnings {where} ORDER BY ts DESC LIMIT ?", params)
        return [Learning.from_row(r) for r in rows]

    def recall(self, query: str, *, limit: int = 20, session_id: str | None = None) -> list[Learning]:
        """Search learnings by substring across insight and evidence.

        Matches have their ``hit_count`` bumped and ``last_hit`` stamped, so a
        frequently-useful pattern surfaces its value over time, and may cross
        ``promote_at`` and graduate to a hypothesis as a result.
        """
        sid = self._sid(session_id)
        like = f"%{query}%"
        rows = self._query(
            """SELECT * FROM learnings
               WHERE insight LIKE ? OR evidence LIKE ?
               ORDER BY ts DESC LIMIT ?""",
            (like, like, limit),
        )
        results = [Learning.from_row(r) for r in rows]
        self._emit("recall", session_id=sid, data={"query": query, "hits": len(results)})
        if results:
            now = _now()
            ids = [r.id for r in results]
            placeholders = ",".join("?" for _ in ids)
            self._execute(
                f"""UPDATE learnings SET hit_count = hit_count + 1, last_hit = ?
                    WHERE id IN ({placeholders})""",
                [now, *ids],
            )
            for lid in ids:
                self._maybe_promote(lid)
            # Re-read so returned objects reflect the bumped counts.
            results = [self.get_learning(i) for i in ids]  # type: ignore[misc]
            results = [r for r in results if r is not None]
        return results

    def forget(self, learning_id: str) -> bool:
        """Delete a learning by id. Returns True if a row was removed."""
        cur = self._execute("DELETE FROM learnings WHERE id = ?", (learning_id,))
        return cur.rowcount > 0

    # -- the loop, stage 1: pattern → hypothesis ---------------------------

    def _maybe_promote(self, learning_id: str) -> None:
        """Graduate a recurring learning into a hypothesis once it crosses the bar.

        The old ``type == 'pattern'`` hard gate is removed (design 3.4): a
        recurring gotcha or failure is exactly what a "do not repeat this" memory
        should promote. The uniform ``promote_at`` threshold applies to every
        type; per-type thresholds are deliberately out of scope (Appendix A).
        """
        rows = self._query("SELECT * FROM learnings WHERE id = ?", (learning_id,))
        if not rows:
            return
        row = rows[0]
        if (
            row["hit_count"] < self.promote_at
            or row["hypothesis_id"] is not None
        ):
            return
        hid = _new_id("hyp")
        self._execute(
            """INSERT INTO hypotheses
               (id, ts, session_id, statement, domain, source_learning_id, status, confidence)
               VALUES (?, ?, ?, ?, ?, ?, 'testing', 0.0)""",
            (hid, _now(), row["session_id"], row["insight"], row["domain"], row["id"]),
        )
        self._execute(
            "UPDATE learnings SET hypothesis_id = ? WHERE id = ?", (hid, row["id"])
        )
        self._emit(
            "graduate",
            session_id=row["session_id"],
            ref_id=hid,
            data={"from": row["type"], "to": "hypothesis", "learning_id": row["id"]},
        )

    def hypotheses(self, *, status: str | None = None, limit: int = 50) -> list[Hypothesis]:
        where = "WHERE status = ?" if status else ""
        params: list[Any] = [status] if status else []
        params.append(limit)
        rows = self._query(
            f"SELECT * FROM hypotheses {where} ORDER BY ts DESC LIMIT ?", params
        )
        return [Hypothesis.from_row(r) for r in rows]

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        rows = self._query("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
        return Hypothesis.from_row(rows[0]) if rows else None

    def experiments(self, *, hypothesis: str | None = None, limit: int = 100) -> list[Experiment]:
        where = "WHERE hypothesis_id = ?" if hypothesis else ""
        params: list[Any] = [hypothesis] if hypothesis else []
        params.append(limit)
        rows = self._query(
            f"SELECT * FROM experiments {where} ORDER BY ts DESC LIMIT ?", params
        )
        return [Experiment.from_row(r) for r in rows]

    # -- context (work state) ----------------------------------------------

    def _write_context(
        self,
        type: str,
        content: Any,
        *,
        kind: str | None = None,
        unit_id: str | None = None,
        hypothesis_id: str | None = None,
        acceptance: Sequence[str] | None = None,
        agent: str | None = None,
        session_id: str | None = None,
    ) -> ContextEntry:
        sid = self._sid(session_id)
        entry_id = _new_id(type)
        self._execute(
            """INSERT INTO context
               (id, ts, session_id, type, kind, contract_id, hypothesis_id, acceptance, agent, content)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                _now(),
                sid,
                type,
                kind,
                unit_id,
                hypothesis_id,
                _dump(list(acceptance)) if acceptance is not None else None,
                agent if agent is not None else self.agent,
                _dump(content),
            ),
        )
        self._emit(type, session_id=sid, ref_id=entry_id, data={"kind": kind})
        return self.get_context(entry_id)  # type: ignore[return-value]

    def get_context(self, entry_id: str) -> ContextEntry | None:
        rows = self._query("SELECT * FROM context WHERE id = ?", (entry_id,))
        return ContextEntry.from_row(rows[0]) if rows else None

    def unit(
        self,
        statement: Any,
        *,
        kind: str = "contract",
        acceptance: Sequence[str] | None = None,
        hypothesis: str | None = None,
        agent: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Open a unit of work and return its id.

        ``kind="spec"`` describes code work and **requires** ``acceptance=[...]``
        (the checks that must pass); ``kind="contract"`` describes non-code work
        and leaves acceptance optional. A unit may link to a ``hypothesis`` it
        tests, which is what lets a later verdict feed the loop.
        """
        if kind not in _UNIT_KINDS:
            raise ValueError(f"kind must be one of {sorted(_UNIT_KINDS)}, got {kind!r}")
        if kind == "spec" and not acceptance:
            raise ValueError("kind='spec' requires a non-empty acceptance=[...] list")
        entry = self._write_context(
            "unit",
            statement,
            kind=kind,
            hypothesis_id=hypothesis,
            acceptance=acceptance,
            agent=agent,
            session_id=session_id,
        )
        # A unit references itself so it surfaces when querying its own id.
        self._execute("UPDATE context SET contract_id = ? WHERE id = ?", (entry.id, entry.id))
        return entry.id

    # Backwards-friendly alias: a contract is a unit of kind 'contract'.
    def contract(self, content: Any, *, agent: str | None = None, session_id: str | None = None) -> str:
        """Alias for :meth:`unit` with ``kind="contract"``."""
        return self.unit(content, kind="contract", agent=agent, session_id=session_id)

    def checkpoint(
        self,
        content: Any,
        *,
        unit: str | None = None,
        agent: str | None = None,
        session_id: str | None = None,
    ) -> ContextEntry:
        """Record progress mid-work."""
        return self._write_context(
            "checkpoint", content, unit_id=unit, agent=agent, session_id=session_id
        )

    def handoff(
        self,
        content: Any,
        *,
        unit: str | None = None,
        agent: str | None = None,
        session_id: str | None = None,
    ) -> ContextEntry:
        """Record a handoff brief for the next session or agent."""
        return self._write_context(
            "handoff", content, unit_id=unit, agent=agent, session_id=session_id
        )

    # -- the loop, stage 2: verdict → experiment → graduation --------------

    def verdict(
        self,
        result: str,
        *,
        unit: str | None = None,
        hypothesis: str | None = None,
        evidence: str | None = None,
        agent: str | None = None,
        session_id: str | None = None,
    ) -> ContextEntry:
        """Record a ``"pass"``/``"fail"`` outcome for a unit of work.

        If the unit (or the explicit ``hypothesis``) tests a hypothesis, the
        verdict also becomes an **experiment**: ``pass`` supports it, ``fail``
        refutes it. The hypothesis's confidence updates, and once it clears
        ``graduate_at`` over ``min_experiments`` samples it graduates into a
        proven ``preference`` learning. That is the loop closing itself.
        """
        if result not in _VERDICTS:
            raise ValueError(f"result must be one of {sorted(_VERDICTS)}, got {result!r}")
        sid = self._sid(session_id)
        unit_hyp = self._unit_hypothesis(unit)
        if hypothesis is not None and unit_hyp is not None and hypothesis != unit_hyp:
            raise ValueError(
                "unit tests a different hypothesis than the one passed; "
                "pass only `unit` or only `hypothesis`, or make them agree"
            )
        hyp_id = hypothesis or unit_hyp
        # Hold the lock across the whole verdict→experiment→graduation sequence so
        # concurrent verdicts on the same hypothesis can't both pass the
        # graduation gate (the RLock is reentrant, so nested writes are fine).
        with self._lock:
            entry = self._write_context(
                "verdict",
                {"result": result, "evidence": evidence},
                unit_id=unit,
                hypothesis_id=hyp_id,
                agent=agent,
                session_id=sid,
            )
            if hyp_id is not None:
                self._record_experiment(hyp_id, unit, result, evidence, sid)
        return entry

    def _unit_hypothesis(self, unit_id: str | None) -> str | None:
        if unit_id is None:
            return None
        rows = self._query(
            "SELECT hypothesis_id FROM context WHERE id = ? AND type = 'unit'", (unit_id,)
        )
        return rows[0]["hypothesis_id"] if rows else None

    def _record_experiment(
        self,
        hypothesis_id: str,
        unit_id: str | None,
        result: str,
        evidence: str | None,
        session_id: str,
    ) -> None:
        outcome = "supports" if result == "pass" else "refutes"
        eid = _new_id("exp")
        self._execute(
            """INSERT INTO experiments
               (id, ts, session_id, hypothesis_id, unit_id, result, evidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, _now(), session_id, hypothesis_id, unit_id, outcome, evidence),
        )
        col = "evidence_for" if outcome == "supports" else "evidence_against"
        self._execute(f"UPDATE hypotheses SET {col} = {col} + 1 WHERE id = ?", (hypothesis_id,))
        rows = self._query("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,))
        if not rows:
            return
        h = rows[0]
        total = h["evidence_for"] + h["evidence_against"]
        confidence = (h["evidence_for"] / total) if total else 0.0
        self._execute(
            "UPDATE hypotheses SET confidence = ? WHERE id = ?", (confidence, hypothesis_id)
        )
        self._emit(
            "experiment",
            session_id=session_id,
            ref_id=eid,
            data={"hypothesis_id": hypothesis_id, "result": outcome, "confidence": confidence},
        )
        self._maybe_graduate(hypothesis_id, h, total, confidence)

    def _maybe_graduate(
        self, hypothesis_id: str, h: sqlite3.Row, total: int, confidence: float
    ) -> None:
        """Promote a proven hypothesis to a preference, or reject a refuted one.

        The status transition is a conditional UPDATE (``WHERE status='testing'``)
        whose ``rowcount`` is the single source of truth: only the call that
        actually flips the row does the follow-on work, so a hypothesis can never
        double-graduate even if two verdicts race. ``_EPS`` keeps exact-boundary
        confidences on the intended side of the line.
        """
        if h["status"] != "testing" or total < self.min_experiments:
            return
        if confidence + _EPS >= self.graduate_at:
            cur = self._execute(
                "UPDATE hypotheses SET status = 'graduated' WHERE id = ? AND status = 'testing'",
                (hypothesis_id,),
            )
            if cur.rowcount != 1:
                return  # someone else already transitioned it
            evidence = (
                f"proven: {h['evidence_for']}/{total} experiments support "
                f"(conf {confidence:.2f})"
            )
            lid = self._emit_preference(
                h["statement"], h["domain"], h["session_id"], hypothesis_id, evidence
            )
            self._execute(
                "UPDATE hypotheses SET graduated_learning_id = ? WHERE id = ?", (lid, hypothesis_id)
            )
            self._emit(
                "graduate",
                session_id=h["session_id"],
                ref_id=lid,
                data={"from": "hypothesis", "to": "preference", "hypothesis_id": hypothesis_id},
            )
        elif (1.0 - confidence) + _EPS >= self.graduate_at:
            cur = self._execute(
                "UPDATE hypotheses SET status = 'rejected' WHERE id = ? AND status = 'testing'",
                (hypothesis_id,),
            )
            if cur.rowcount != 1:
                return
            self._emit(
                "graduate",
                session_id=h["session_id"],
                ref_id=hypothesis_id,
                data={"from": "hypothesis", "to": "rejected"},
            )

    def _emit_preference(
        self,
        statement: str,
        domain: str | None,
        session_id: str | None,
        hypothesis_id: str,
        evidence: str,
    ) -> str:
        """Create (or reinforce) the proven ``preference`` learning for a graduation.

        Routes through the same ``(type, insight)`` dedup as :meth:`learn` so two
        hypotheses with the same statement don't produce duplicate preferences.
        """
        existing = self._query(
            "SELECT id FROM learnings WHERE type = 'preference' AND insight = ? LIMIT 1",
            (statement,),
        )
        now = _now()
        if existing:
            lid = existing[0]["id"]
            self._execute(
                """UPDATE learnings
                   SET hit_count = hit_count + 1, last_hit = ?, evidence = ?, hypothesis_id = ?
                   WHERE id = ?""",
                (now, evidence, hypothesis_id, lid),
            )
            return lid
        lid = _new_id("learn")
        self._execute(
            """INSERT INTO learnings
               (id, ts, session_id, type, insight, evidence, domain,
                hit_count, last_hit, hypothesis_id)
               VALUES (?, ?, ?, 'preference', ?, ?, ?, 1, ?, ?)""",
            (lid, now, session_id, statement, evidence, domain, now, hypothesis_id),
        )
        return lid

    def context(
        self,
        *,
        type: str | None = None,
        unit: str | None = None,
        limit: int = 50,
    ) -> list[ContextEntry]:
        """Fetch context entries, newest first, optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if unit is not None:
            clauses.append("contract_id = ?")
            params.append(unit)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._query(f"SELECT * FROM context {where} ORDER BY ts DESC LIMIT ?", params)
        return [ContextEntry.from_row(r) for r in rows]

    # -- errors ------------------------------------------------------------

    def capture_error(
        self,
        tool: str,
        error: str,
        *,
        file: str | None = None,
        context: str | None = None,
        domain: str | None = None,
        session_id: str | None = None,
    ) -> ErrorRecord:
        """Record a failure for later diagnosis."""
        sid = self._sid(session_id)
        ts = _now()
        cur = self._execute(
            """INSERT INTO errors (ts, session_id, tool, error, file, context, domain)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts, sid, tool, error, file, context, domain),
        )
        self._emit("error", session_id=sid, ref_id=str(cur.lastrowid), data={"tool": tool})
        return ErrorRecord(
            id=cur.lastrowid or 0,
            ts=ts,
            tool=tool,
            error=error,
            file=file,
            context=context,
            domain=domain,
            session_id=sid,
        )

    def errors(self, *, limit: int = 50) -> list[ErrorRecord]:
        """Fetch captured errors, newest first."""
        rows = self._query("SELECT * FROM errors ORDER BY ts DESC LIMIT ?", (limit,))
        return [ErrorRecord.from_row(r) for r in rows]

    # -- session digest ----------------------------------------------------

    def read_start(self, *, learnings_limit: int = 20) -> StartBrief:
        """Build the "what to know before working" digest.

        Proven ``preferences`` come first, the rules metabrain earned through the
        loop, then recent learnings, hypotheses still under test, open units
        without a verdict, the latest checkpoint, and recent errors.
        """
        preferences = self.learnings(type="preference", limit=learnings_limit)
        recent = self._query(
            "SELECT * FROM learnings WHERE type != 'preference' ORDER BY ts DESC LIMIT ?",
            (learnings_limit,),
        )
        recent_learnings = [Learning.from_row(r) for r in recent]

        open_hypotheses = self.hypotheses(status="testing")

        unit_rows = self._query(
            """SELECT c.* FROM context c
               WHERE c.type = 'unit'
                 AND NOT EXISTS (
                   SELECT 1 FROM context v
                   WHERE v.type = 'verdict' AND v.contract_id = c.contract_id
                 )
               ORDER BY c.ts DESC"""
        )
        open_units = [ContextEntry.from_row(r) for r in unit_rows]

        last_rows = self._query(
            "SELECT * FROM context WHERE type = 'checkpoint' ORDER BY ts DESC LIMIT 1"
        )
        last_checkpoint = ContextEntry.from_row(last_rows[0]) if last_rows else None

        return StartBrief(
            preferences=preferences,
            learnings=recent_learnings,
            open_hypotheses=open_hypotheses,
            open_units=open_units,
            last_checkpoint=last_checkpoint,
            recent_errors=self.errors(limit=5),
        )

    def write_end(self, content: Any, *, unit: str | None = None) -> ContextEntry:
        """Checkpoint before stopping. Shorthand for :meth:`checkpoint`."""
        return self.checkpoint(content, unit=unit)

    # -- maintenance -------------------------------------------------------

    def prune(self, *, keep: int = 50) -> int:
        """Trim checkpoint history to the most recent ``keep`` entries.

        Units, handoffs, verdicts, learnings, hypotheses, and experiments are
        never pruned, only the high-churn checkpoint trail. Returns the count
        deleted.
        """
        cur = self._execute(
            """DELETE FROM context
               WHERE type = 'checkpoint' AND id NOT IN (
                 SELECT id FROM context WHERE type = 'checkpoint'
                 ORDER BY ts DESC LIMIT ?
               )""",
            (keep,),
        )
        return cur.rowcount

    def stats(self) -> dict[str, int]:
        """Row counts per table, a quick health check that the loop is turning."""
        out: dict[str, int] = {}
        for table in (
            "sessions",
            "events",
            "learnings",
            "context",
            "hypotheses",
            "experiments",
            "errors",
        ):
            rows = self._query(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = rows[0]["n"]
        return out


class Session:
    """A handle on one open session. Use it as a context manager.

    Every method delegates to :class:`MetaBrain` with this session's id attached,
    so all writes inherit a ``session_id`` and the session's outcome is recorded
    on exit. An exception inside the block is captured as an error and marks the
    session unsuccessful before propagating.
    """

    __slots__ = ("db", "id", "_outcome", "_success", "_tokens", "_closed")

    def __init__(self, db: MetaBrain, session_id: str) -> None:
        self.db = db
        self.id = session_id
        self._outcome: str | None = None
        self._success: bool | None = None
        self._tokens: int | None = None
        self._closed = False

    # context-manager protocol
    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, _tb: object) -> bool:
        if exc is not None:
            self.db.capture_error(
                tool=getattr(exc_type, "__name__", "exception"),
                error=str(exc),
                context="session",
                session_id=self.id,
            )
            if self._success is None:
                self._success = False
            if self._outcome is None:
                self._outcome = f"{getattr(exc_type, '__name__', 'error')}: {exc}"
        elif self._success is None:
            self._success = True
        self.close()
        return False  # never suppress

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.db._end_session(
            self.id, success=self._success, outcome=self._outcome, tokens=self._tokens
        )

    # outcome setters
    def outcome(self, text: str, *, success: bool = True) -> None:
        """Record how the session went; filled into the row on close."""
        self._outcome = text
        self._success = success

    def tokens(self, n: int) -> None:
        """Record tokens used this session."""
        self._tokens = n

    # delegated writes (all carry this session's id)
    def learn(self, type: str, insight: str, **kw: Any) -> Learning:
        return self.db.learn(type, insight, session_id=self.id, **kw)

    def recall(self, query: str, **kw: Any) -> list[Learning]:
        return self.db.recall(query, session_id=self.id, **kw)

    def unit(self, statement: Any, **kw: Any) -> str:
        return self.db.unit(statement, session_id=self.id, **kw)

    def contract(self, content: Any, **kw: Any) -> str:
        return self.db.contract(content, session_id=self.id, **kw)

    def checkpoint(self, content: Any, **kw: Any) -> ContextEntry:
        return self.db.checkpoint(content, session_id=self.id, **kw)

    def handoff(self, content: Any, **kw: Any) -> ContextEntry:
        return self.db.handoff(content, session_id=self.id, **kw)

    def verdict(self, result: str, **kw: Any) -> ContextEntry:
        return self.db.verdict(result, session_id=self.id, **kw)

    def error(self, tool: str, error: str, **kw: Any) -> ErrorRecord:
        return self.db.capture_error(tool, error, session_id=self.id, **kw)
