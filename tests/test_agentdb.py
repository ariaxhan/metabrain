"""Behavioural tests for metabrain. Each test proves a contract of the API, not
its implementation, so the schema can evolve underneath them.

The headline tests are the loop (``test_full_loop_*``) and the three target use
cases (content / lead capture / job applications), those are what metabrain
exists to do, so they are tested as first-class behaviour.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from metabrain import MetaBrain, ContextEntry, IncompatibleDatabaseError, Learning


@pytest.fixture()
def db(tmp_path):
    database = MetaBrain(tmp_path / "agent.db", agent="tester")
    yield database
    database.close()


# -- learnings -------------------------------------------------------------

def test_learn_returns_stored_record(db):
    rec = db.learn("gotcha", "WAL needed for concurrent agents", domain="db")
    assert isinstance(rec, Learning)
    assert rec.id.startswith("learn_")
    assert rec.type == "gotcha"
    assert rec.domain == "db"
    assert rec.hit_count == 1


def test_learnings_round_trip_and_filter(db):
    db.learn("failure", "auth token expired silently", domain="auth")
    db.learn("pattern", "retry with backoff", domain="net")
    assert len(db.learnings()) == 2
    assert len(db.learnings(domain="auth")) == 1
    assert len(db.learnings(type="pattern")) == 1


def test_learn_rejects_bad_type(db):
    with pytest.raises(ValueError):
        db.learn("nonsense", "x")


def test_learn_rejects_empty_insight(db):
    with pytest.raises(ValueError):
        db.learn("pattern", "   ")


def test_relearning_same_insight_bumps_instead_of_duplicating(db):
    db.learn("pattern", "same thing", domain="x")
    db.learn("pattern", "same thing")
    rows = db.learnings(type="pattern")
    assert len(rows) == 1
    assert rows[0].hit_count == 2
    assert rows[0].domain == "x"  # original domain preserved via COALESCE


def test_recall_searches_and_bumps_hit_count(db):
    db.learn("gotcha", "sqlite locks under concurrent writes")
    db.learn("pattern", "unrelated thing")
    hits = db.recall("concurrent")
    assert len(hits) == 1
    assert hits[0].insight.startswith("sqlite locks")
    assert hits[0].hit_count == 2  # 1 on create + 1 on recall
    assert hits[0].last_hit is not None


def test_forget(db):
    rec = db.learn("pattern", "temporary")
    assert db.forget(rec.id) is True
    assert db.forget(rec.id) is False
    assert db.learnings() == []


# -- the self-improving loop (the moat) ------------------------------------

def test_pattern_graduates_to_hypothesis_at_promote_at(tmp_path):
    db = MetaBrain(tmp_path / "a.db", promote_at=3)
    db.learn("pattern", "X works", domain="d")  # hit 1
    assert db.hypotheses() == []
    db.learn("pattern", "X works")              # hit 2
    assert db.hypotheses() == []
    db.learn("pattern", "X works")              # hit 3 -> graduates
    hyps = db.hypotheses(status="testing")
    assert len(hyps) == 1
    assert hyps[0].statement == "X works"
    # the source learning now points at its hypothesis
    learning = db.learnings(type="pattern")[0]
    assert learning.hypothesis_id == hyps[0].id
    db.close()


def test_recall_can_trigger_graduation(tmp_path):
    db = MetaBrain(tmp_path / "a.db", promote_at=3)
    db.learn("pattern", "recall me")            # hit 1
    db.recall("recall")                         # hit 2
    db.recall("recall")                         # hit 3 -> graduates
    assert len(db.hypotheses(status="testing")) == 1
    db.close()


def test_all_types_graduate_at_uniform_threshold(tmp_path):
    # design 3.4: the type=='pattern' hard gate is removed, so a recurring
    # gotcha (or failure) graduates at the same uniform promote_at as a pattern.
    # A recurring "do not repeat this mistake" is exactly what should promote.
    db = MetaBrain(tmp_path / "a.db", promote_at=2)
    for _ in range(2):
        db.learn("gotcha", "recurring gotcha worth promoting")
    hyps = db.hypotheses()
    assert len(hyps) == 1
    assert hyps[0].statement == "recurring gotcha worth promoting"
    db.close()


def test_full_loop_pattern_to_proven_preference(tmp_path):
    """The headline behaviour: a recurring pattern, tested by verdicts, becomes
    a proven `preference` with zero manual bookkeeping."""
    db = MetaBrain(tmp_path / "loop.db", promote_at=2, graduate_at=0.8, min_experiments=3)

    # 1. a pattern recurs and graduates to a hypothesis
    db.learn("pattern", "ship small PRs", domain="eng")
    db.learn("pattern", "ship small PRs")
    h = db.hypotheses(status="testing")[0]

    # 2. verdicts on units that test it become experiments
    u1 = db.unit("split the auth PR", kind="contract", hypothesis=h.id)
    db.verdict("pass", unit=u1, evidence="merged clean")
    u2 = db.unit("split the billing PR", kind="contract", hypothesis=h.id)
    db.verdict("pass", unit=u2, evidence="merged clean")
    # not graduated yet: only 2 experiments, below min_experiments=3
    assert db.get_hypothesis(h.id).status == "testing"
    u3 = db.unit("split the api PR", kind="contract", hypothesis=h.id)
    db.verdict("pass", unit=u3, evidence="merged clean")

    # 3. the hypothesis graduated into a proven preference
    refreshed = db.get_hypothesis(h.id)
    assert refreshed.status == "graduated"
    assert refreshed.confidence == 1.0
    prefs = db.learnings(type="preference")
    assert len(prefs) == 1
    assert prefs[0].insight == "ship small PRs"
    assert "experiments support" in prefs[0].evidence

    # 4. experiments were recorded
    assert len(db.experiments(hypothesis=h.id)) == 3
    db.close()


def test_hypothesis_rejected_when_evidence_refutes(tmp_path):
    db = MetaBrain(tmp_path / "r.db", promote_at=2, graduate_at=0.8, min_experiments=3)
    db.learn("pattern", "premature idea")
    db.learn("pattern", "premature idea")
    h = db.hypotheses(status="testing")[0]
    for _ in range(3):
        u = db.unit("test it", kind="contract", hypothesis=h.id)
        db.verdict("fail", unit=u, evidence="did not help")
    assert db.get_hypothesis(h.id).status == "rejected"
    assert db.learnings(type="preference") == []
    db.close()


def test_verdict_on_explicit_hypothesis(tmp_path):
    db = MetaBrain(tmp_path / "e.db", promote_at=1, min_experiments=2, graduate_at=0.8)
    db.learn("pattern", "direct")               # graduates immediately (promote_at=1)
    h = db.hypotheses(status="testing")[0]
    db.verdict("pass", hypothesis=h.id, evidence="a")
    db.verdict("pass", hypothesis=h.id, evidence="b")
    assert db.get_hypothesis(h.id).status == "graduated"
    db.close()


# -- units / context -------------------------------------------------------

def test_unit_checkpoint_verdict_flow(db):
    uid = db.unit({"goal": "ship auth"}, kind="contract")
    assert uid.startswith("unit_")
    cp = db.checkpoint({"did": "wired login"}, unit=uid)
    assert isinstance(cp, ContextEntry)
    assert cp.unit_id == uid
    db.verdict("pass", unit=uid, evidence="12 tests green")
    entries = db.context(unit=uid)
    types = {e.type for e in entries}
    assert types == {"unit", "checkpoint", "verdict"}


def test_spec_requires_acceptance(db):
    with pytest.raises(ValueError):
        db.unit("add timeout", kind="spec")
    uid = db.unit("add timeout", kind="spec", acceptance=["fires at 5s", "tests pass"])
    entry = db.get_context(uid)
    assert entry.kind == "spec"
    assert entry.acceptance == ["fires at 5s", "tests pass"]


def test_contract_alias(db):
    uid = db.contract({"goal": "x"})
    assert db.get_context(uid).kind == "contract"


def test_checkpoint_preserves_dict_content(db):
    db.checkpoint({"did": "X", "learned": ["Y", "Z"]})
    fetched = db.context(type="checkpoint")[0]
    assert fetched.content == {"did": "X", "learned": ["Y", "Z"]}


def test_checkpoint_preserves_string_content(db):
    db.checkpoint("just a note")
    assert db.context(type="checkpoint")[0].content == "just a note"


def test_default_agent_label_applied(db):
    cp = db.checkpoint({"did": "x"})
    assert cp.agent == "tester"


def test_verdict_rejects_bad_result(db):
    with pytest.raises(ValueError):
        db.verdict("maybe")


# -- sessions & deterministic population ------------------------------------

def test_session_writes_and_closes_row(tmp_path):
    db = MetaBrain(tmp_path / "s.db")
    with db.session(task="feature", tier=2) as s:
        s.learn("pattern", "inside a session")
        sid = s.id
    rows = db._query("SELECT * FROM sessions WHERE id = ?", (sid,))
    assert rows[0]["ended_at"] is not None
    assert rows[0]["success"] == 1
    assert rows[0]["task"] == "feature"
    db.close()


def test_session_records_exception_as_error_and_failure(tmp_path):
    db = MetaBrain(tmp_path / "x.db")
    with pytest.raises(RuntimeError):
        with db.session(task="risky") as s:
            sid = s.id
            raise RuntimeError("boom")
    row = db._query("SELECT * FROM sessions WHERE id = ?", (sid,))[0]
    assert row["success"] == 0
    assert "boom" in row["outcome"]
    assert any("boom" in e.error for e in db.errors())
    db.close()


def test_every_write_emits_an_event(tmp_path):
    db = MetaBrain(tmp_path / "ev.db")
    with db.session(task="t") as s:
        s.learn("pattern", "a")
        s.checkpoint({"did": "b"})
    kinds = {r["kind"] for r in db._query("SELECT kind FROM events")}
    assert {"session", "learn", "checkpoint"} <= kinds
    db.close()


def test_flat_api_uses_ambient_session(tmp_path):
    db = MetaBrain(tmp_path / "amb.db")
    db.learn("pattern", "no explicit session")
    sessions = db._query("SELECT * FROM sessions WHERE task = 'ambient'")
    assert len(sessions) == 1
    events = db._query("SELECT * FROM events WHERE session_id = ?", (sessions[0]["id"],))
    assert len(events) >= 1
    db.close()


def test_stats_covers_all_core_tables(db):
    with db.session(task="t", tier=1) as s:
        s.learn("pattern", "a")
        s.learn("pattern", "a")  # bump; promote_at default 3 so no hypothesis yet
        uid = s.unit("do x", kind="contract")
        s.checkpoint({"n": 1}, unit=uid)
        s.error("Bash", "boom")
    stats = db.stats()
    assert stats["sessions"] >= 1
    assert stats["events"] >= 1
    assert stats["learnings"] == 1
    assert stats["context"] >= 2
    assert stats["errors"] == 1


# -- read_start digest -----------------------------------------------------

def test_read_start_surfaces_preferences_first(tmp_path):
    db = MetaBrain(tmp_path / "b.db", promote_at=1, min_experiments=2, graduate_at=0.8)
    db.learn("pattern", "proven rule")          # graduates to hypothesis immediately
    h = db.hypotheses(status="testing")[0]
    db.verdict("pass", hypothesis=h.id)
    db.verdict("pass", hypothesis=h.id)          # -> preference
    db.learn("gotcha", "unproven note")

    brief = db.read_start()
    assert any(p.insight == "proven rule" for p in brief.preferences)
    assert all(p.type == "preference" for p in brief.preferences)
    assert any(l.insight == "unproven note" for l in brief.learnings)
    assert all(l.type != "preference" for l in brief.learnings)
    db.close()


def test_read_start_open_units_excludes_verdicted(db):
    open_uid = db.unit({"goal": "open"}, kind="contract")
    closed_uid = db.unit({"goal": "done"}, kind="contract")
    db.verdict("pass", unit=closed_uid)
    db.checkpoint({"did": "latest"})

    brief = db.read_start()
    open_ids = {u.unit_id for u in brief.open_units}
    assert open_uid in open_ids
    assert closed_uid not in open_ids
    assert brief.last_checkpoint.content == {"did": "latest"}


# -- errors ----------------------------------------------------------------

def test_capture_and_list_errors(db):
    db.capture_error("Edit", "file not found", file="a.py", domain="fs")
    errs = db.errors()
    assert len(errs) == 1
    assert errs[0].tool == "Edit"
    assert errs[0].file == "a.py"


# -- maintenance -----------------------------------------------------------

def test_prune_keeps_recent_checkpoints_only(db):
    for i in range(10):
        db.checkpoint({"n": i})
    db.unit({"keep": "me"}, kind="contract")  # non-checkpoint, must survive
    deleted = db.prune(keep=3)
    assert deleted == 7
    assert len(db.context(type="checkpoint")) == 3
    assert len(db.context(type="unit")) == 1


# -- persistence & concurrency --------------------------------------------

def test_persists_across_reopen(tmp_path):
    path = tmp_path / "p.db"
    with MetaBrain(path) as db:
        db.learn("pattern", "durable")
    with MetaBrain(path) as db2:
        assert len(db2.learnings()) == 1


def test_thread_safe_writes(tmp_path):
    db = MetaBrain(tmp_path / "c.db")

    def worker(n: int) -> None:
        for i in range(20):
            db.learn("pattern", f"thread {n} item {i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(db.learnings(limit=1000)) == 80
    db.close()


def test_in_memory_database():
    db = MetaBrain(":memory:")
    db.learn("pattern", "ephemeral")
    assert len(db.learnings()) == 1
    db.close()


# -- read-compatibility with a bash-era database ---------------------------

def test_opens_and_migrates_legacy_database(tmp_path):
    """A pre-v1 database (no sessions/events/hypotheses, old columns) must open,
    migrate forward, and keep its data."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE learnings (
          id TEXT PRIMARY KEY, ts TEXT NOT NULL, type TEXT NOT NULL,
          insight TEXT NOT NULL, evidence TEXT, domain TEXT,
          hit_count INTEGER NOT NULL DEFAULT 0, last_hit TEXT,
          visibility TEXT NOT NULL DEFAULT 'agent',
          sensitivity TEXT NOT NULL DEFAULT 'low'
        );
        CREATE TABLE context (
          id TEXT PRIMARY KEY, ts TEXT NOT NULL, type TEXT NOT NULL,
          contract_id TEXT, agent TEXT, content TEXT NOT NULL
        );
        CREATE TABLE errors (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
          tool TEXT NOT NULL, error TEXT NOT NULL, file TEXT, context TEXT, domain TEXT
        );
        INSERT INTO learnings (id, ts, type, insight) VALUES ('l1', '2025-01-01T00:00:00Z', 'design_decision', 'legacy insight');
        INSERT INTO context (id, ts, type, content) VALUES ('c1', '2025-01-01T00:00:00Z', 'checkpoint', 'old note');
        """
    )
    conn.commit()
    conn.close()

    db = MetaBrain(path)
    # legacy rows survived, including the out-of-vocabulary type
    assert any(l.insight == "legacy insight" for l in db.learnings())
    assert db.context(type="checkpoint")[0].content == "old note"
    # new features work on the migrated DB
    db.learn("pattern", "post-migration")
    assert db.stats()["sessions"] >= 1
    db.close()


# -- use case 1: self-learning content engine (maerai) ---------------------

def test_use_case_content_engine(tmp_path):
    """A pattern about what drives engagement gets proven by post outcomes and
    graduates into the brand's playbook, replacing maerai's patterns.json."""
    db = MetaBrain(tmp_path / "content.db", promote_at=2, min_experiments=3, graduate_at=0.66)
    with db.session(task="content", agent="maerai") as s:
        s.learn("pattern", "question hooks lift saves", domain="instagram")
        s.learn("pattern", "question hooks lift saves", domain="instagram")  # -> hypothesis
        h = db.hypotheses(status="testing")[0]
        for saves, ok in [(1200, "pass"), (90, "fail"), (1500, "pass"), (1100, "pass")]:
            post = s.unit(f"carousel, {saves} saves", kind="contract", hypothesis=h.id)
            s.verdict(ok, unit=post, evidence=f"{saves} saves")
    # 3 of 4 supported -> confidence 0.75 >= 0.66 -> graduated playbook rule
    assert db.get_hypothesis(h.id).status == "graduated"
    playbook = [p.insight for p in db.read_start().preferences]
    assert "question hooks lift saves" in playbook


# -- use case 2: lead capture ----------------------------------------------

def test_use_case_lead_capture(tmp_path):
    """Per-lead memory plus a learned rule about what converts."""
    db = MetaBrain(tmp_path / "leads.db", promote_at=2, min_experiments=2, graduate_at=0.8)
    with db.session(task="lead-capture") as s:
        lead = s.unit({"name": "Acme", "source": "webinar"}, kind="contract")
        s.checkpoint({"stage": "contacted"}, unit=lead)
        s.checkpoint({"stage": "demo booked"}, unit=lead)
        s.verdict("pass", unit=lead, evidence="closed")
        # learn + prove what converts
        s.learn("pattern", "follow up within 1h converts")
        s.learn("pattern", "follow up within 1h converts")
        h = db.hypotheses(status="testing")[0]
        s.verdict("pass", hypothesis=h.id, evidence="lead A closed")
        s.verdict("pass", hypothesis=h.id, evidence="lead B closed")
    history = [c.content for c in db.context(unit=lead)]
    assert {"stage": "contacted"} in history
    assert db.get_hypothesis(h.id).status == "graduated"


# -- use case 3: self-improving job applications ---------------------------

def test_use_case_job_applications(tmp_path):
    """Each application is a unit; a learned tactic about what gets responses
    graduates once enough applications confirm it."""
    db = MetaBrain(tmp_path / "jobs.db", promote_at=2, min_experiments=3, graduate_at=0.6)
    with db.session(task="job-app") as s:
        s.learn("pattern", "lead with a shipped metric", domain="resume")
        s.learn("pattern", "lead with a shipped metric", domain="resume")
        h = db.hypotheses(status="testing")[0]
        for company, got_reply in [("A", "pass"), ("B", "fail"), ("C", "pass"), ("D", "pass")]:
            app = s.unit({"company": company}, kind="contract", hypothesis=h.id)
            s.verdict(got_reply, unit=app, evidence=f"{company} reply={got_reply}")
    assert db.get_hypothesis(h.id).status == "graduated"
    assert any("shipped metric" in p.insight for p in db.read_start().preferences)


# -- adversarial regressions (confirmed bugs, now guarded) -----------------

def test_concurrent_verdicts_graduate_exactly_once(tmp_path):
    """C1: racing verdicts on one hypothesis must not double-graduate."""
    for _ in range(8):
        db = MetaBrain(tmp_path / f"g{_}.db", promote_at=1, min_experiments=3, graduate_at=0.8)
        db.learn("pattern", "grace")
        h = db.hypotheses(status="testing")[0]
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            for _ in range(10):
                db.verdict("pass", hypothesis=h.id)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(db.learnings(type="preference")) == 1
        db.close()


def test_two_hypotheses_same_statement_dedupe_preference(tmp_path):
    """H1: two hypotheses with identical statements graduate to one preference."""
    db = MetaBrain(tmp_path / "d.db", promote_at=1, min_experiments=2, graduate_at=0.8)
    db.learn("pattern", "same statement")
    h1 = db.hypotheses(status="testing")[0]
    db.verdict("pass", hypothesis=h1.id)
    db.verdict("pass", hypothesis=h1.id)
    db.forget(db.learnings(type="pattern")[0].id)
    db.learn("pattern", "same statement")
    h2 = [h for h in db.hypotheses(status="testing")][0]
    db.verdict("pass", hypothesis=h2.id)
    db.verdict("pass", hypothesis=h2.id)
    prefs = db.learnings(type="preference")
    assert len(prefs) == 1
    assert prefs[0].hit_count == 2  # reinforced, not duplicated
    db.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"promote_at": 0},
        {"min_experiments": 0},
        {"min_experiments": -5},
        {"graduate_at": 0.5},
        {"graduate_at": -0.2},
        {"graduate_at": 1.5},
    ],
)
def test_rejects_out_of_range_loop_params(tmp_path, kwargs):
    """H2: nonsense tuning parameters are refused at construction."""
    with pytest.raises(ValueError):
        MetaBrain(tmp_path / "p.db", **kwargs)


def test_exact_reject_boundary_rejects(tmp_path):
    """H3: 1 pass / 4 fail at graduate_at=0.8 is an 80% failure → rejected, not stuck."""
    db = MetaBrain(tmp_path / "h3.db", promote_at=1, min_experiments=5, graduate_at=0.8)
    db.learn("pattern", "boundary")
    h = db.hypotheses(status="testing")[0]
    db.verdict("pass", hypothesis=h.id)
    for _ in range(4):
        db.verdict("fail", hypothesis=h.id)
    assert db.get_hypothesis(h.id).status == "rejected"
    db.close()


def test_exact_graduate_boundary_graduates(tmp_path):
    """H3 mirror: 4 pass / 1 fail = 0.8 exactly should graduate."""
    db = MetaBrain(tmp_path / "h3b.db", promote_at=1, min_experiments=5, graduate_at=0.8)
    db.learn("pattern", "boundary2")
    h = db.hypotheses(status="testing")[0]
    for _ in range(4):
        db.verdict("pass", hypothesis=h.id)
    db.verdict("fail", hypothesis=h.id)
    assert db.get_hypothesis(h.id).status == "graduated"
    db.close()


def test_foreign_shaped_table_is_rejected(tmp_path):
    """M1: a pre-existing loop table with an incompatible shape is refused, not
    silently kept until the first write crashes."""
    path = tmp_path / "foreign.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE hypotheses (id TEXT PRIMARY KEY, statement TEXT);")
    conn.commit()
    conn.close()
    with pytest.raises(IncompatibleDatabaseError):
        MetaBrain(path)


def test_legacy_context_without_contract_id_migrates(tmp_path):
    """M2: a legacy context table missing contract_id gains it on open."""
    path = tmp_path / "ctx.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE context (id TEXT PRIMARY KEY, ts TEXT NOT NULL, "
        "type TEXT NOT NULL, content TEXT NOT NULL);"
    )
    conn.commit()
    conn.close()
    db = MetaBrain(path)
    uid = db.unit("works now", kind="contract")
    assert db.get_context(uid) is not None
    db.close()


def test_verdict_conflicting_unit_and_hypothesis_raises(tmp_path):
    """M3: a unit linked to hypothesis A plus an explicit hypothesis B is ambiguous."""
    db = MetaBrain(tmp_path / "m3.db", promote_at=1)
    db.learn("pattern", "A")
    db.learn("pattern", "B")
    ha, hb = db.hypotheses()[0].id, db.hypotheses()[1].id
    u = db.unit("test A", kind="contract", hypothesis=ha)
    with pytest.raises(ValueError):
        db.verdict("pass", unit=u, hypothesis=hb)
    db.close()
