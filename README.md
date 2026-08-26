# metabrain

**A SQLite memory layer for AI agents that learns what works.** Zero dependencies. One file.

Most agent-memory tools store what you *tell* them and hand it back later. metabrain does that too — but it also closes the loop: a pattern you record enough times graduates into a **hypothesis**, every outcome you log becomes an **experiment** for or against it, and once the evidence clears the bar it graduates again into a proven **preference**. Your agent stops guessing and starts running on rules it earned.

```
learn(pattern)  →  recurs  →  hypothesis (under test)
        →  each verdict is an experiment (supports / refutes)
        →  evidence clears the bar  →  preference  (a proven rule)
```

That loop is the whole point. It runs on the Python standard library — no vector database, no server, no API keys.

## Install

```bash
pip install metabrain
```

Python 3.10+. No dependencies beyond the standard library. (Import name is `metabrain`.)

## Quick start

```python
from metabrain import MetaBrain

db = MetaBrain("agent.db")

with db.session(task="content") as s:
    # A hunch. Record it as you notice it — three times and it's worth testing.
    s.learn("pattern", "question hooks lift saves", domain="instagram")
    s.learn("pattern", "question hooks lift saves", domain="instagram")
    s.learn("pattern", "question hooks lift saves", domain="instagram")

    # It just graduated into a hypothesis. Now test it against reality.
    h = db.hypotheses(status="testing")[0]
    post = s.unit("carousel with a question hook", kind="contract", hypothesis=h.id)
    s.verdict("pass", unit=post, evidence="1,240 saves")

# Next session: the proven rules come first.
brief = db.read_start()
for rule in brief.preferences:        # things metabrain has *proven*
    print("PROVEN:", rule.insight)
for h in brief.open_hypotheses:       # things it's still testing
    print("testing:", h.statement, f"({h.confidence:.0%})")
```

You don't have to open a session — the flat API (`db.learn(...)`, `db.verdict(...)`) works too and attaches to an ambient session automatically, so the telemetry still fills.

## Why it's different

|  | metabrain | typical vector-memory store |
| --- | --- | --- |
| Remembers what you tell it | ✅ | ✅ |
| **Proves which memories actually work** | ✅ the learn→experiment→graduate loop | ❌ |
| Working state + telemetry, not just recall | ✅ units, checkpoints, sessions, events | ❌ |
| Infrastructure | a single SQLite file | vector DB / server / API key |
| Dependencies | none (stdlib `sqlite3`) | several |

Recall stays deliberately simple — substring + a hit counter — because the moat is the loop, not embedding search. (Semantic recall may arrive later as an opt-in `metabrain[embeddings]` extra; the core will always be zero-dependency.)

## Built for real, stateful products

The loop is general. Three shapes it was designed against:

**Self-learning content engine.** Each post is a unit; engagement is the verdict. Hooks that keep winning graduate into the brand's proven playbook.
```python
s.learn("pattern", "carousels outperform single images", domain="ig")  # ...×3 → hypothesis
for saves, ok in [(1200,"pass"), (90,"fail"), (1500,"pass"), (1100,"pass")]:
    post = s.unit(f"carousel ({saves} saves)", kind="contract", hypothesis=h.id)
    s.verdict(ok, unit=post, evidence=f"{saves} saves")
# 3/4 supported → graduates into the playbook
```

**Lead capture.** Each lead is a unit with its own checkpoint trail; a tactic about what converts graduates once enough leads confirm it.
```python
lead = s.unit({"name": "Acme", "source": "webinar"}, kind="contract")
s.checkpoint({"stage": "demo booked"}, unit=lead)
s.verdict("pass", unit=lead, evidence="closed")
```

**Self-improving job applications.** Each application is a unit; "lead with a shipped metric" stays a guess until enough replies prove it, then becomes a rule.
```python
app = s.unit({"company": "Acme"}, kind="contract", hypothesis=h.id)
s.verdict("pass", unit=app, evidence="recruiter replied")
```

## How the tables fill themselves

metabrain has seven tables, and you never write to them directly — **correct use of the API fills every one as a side effect.** Open a session and each write inherits its id, emits an event, and turns the loop:

| Table | Filled by | When |
| --- | --- | --- |
| `sessions` | `db.session()` open/close | every run |
| `events` | every write method | always (telemetry is automatic) |
| `learnings` | `learn()` — `preference` rows are *graduated* | always |
| `context` | `unit()`, `checkpoint()`, `handoff()`, `verdict()` | always |
| `hypotheses` | a `pattern` crossing `promote_at` (default 3 hits) | automatic |
| `experiments` | a `verdict()` on a unit/hypothesis under test | automatic |
| `errors` | `capture_error()`, and any exception inside a session | automatic |

The thresholds are tunable and were calibrated on 5,066 real learnings, not guessed: `promote_at=3` (where the recurring-pattern tail actually begins), `graduate_at=0.8` over a minimum of 3 experiments so a single lucky result can't graduate.

```python
db = MetaBrain("agent.db", promote_at=3, graduate_at=0.8, min_experiments=3)
```

## API

| Method | What it does |
| --- | --- |
| `session(*, task, tier, agent, meta)` | Open a session (context manager); records the outcome on close |
| `learn(type, insight, *, evidence, domain, ...)` | Record/reinforce a lesson; recurring `pattern`s graduate to hypotheses |
| `recall(query, *, limit)` | Substring-search lessons; bumps hit count (can trigger graduation) |
| `learnings(*, type, domain, limit)` | Fetch lessons, newest first |
| `forget(id)` | Delete a lesson |
| `unit(statement, *, kind, acceptance, hypothesis)` | Open a unit of work; `kind="spec"` requires `acceptance=[...]` |
| `checkpoint(content, *, unit, agent)` | Record progress mid-work |
| `handoff(content, *, unit, agent)` | Record a brief for the next session |
| `verdict(result, *, unit, hypothesis, evidence)` | `"pass"`/`"fail"`; becomes an experiment when a hypothesis is in play |
| `hypotheses(*, status, limit)` / `experiments(*, hypothesis)` | Inspect the loop |
| `context(*, type, unit, limit)` | Fetch work-state entries |
| `read_start(*, learnings_limit)` | The "what to know" digest — proven preferences first |
| `capture_error(tool, error, ...)` / `errors(*, limit)` | Record / fetch failures |
| `prune(*, keep)` / `stats()` | Trim old checkpoints / row counts per table |

Use `MetaBrain(":memory:")` for an ephemeral in-process store (handy in tests).

## Concurrency & safety

Built for multiple agents sharing one file. SQLite runs in WAL mode with a busy timeout so several processes read and write concurrently; within a process a single connection is lock-guarded, and the verdict→graduation path is one critical section so racing verdicts can never double-graduate a hypothesis. Every value is bound as a query parameter — caller strings never reach the SQL text.

It can open and migrate an older metabrain / base-schema database (learnings, context, errors) forward in place. A database created by a different tool whose `events`/`hypotheses`/`experiments` tables have an incompatible shape is detected on open and rejected with a clear `IncompatibleDatabaseError`, rather than corrupting it.

## Use as an MCP server

<!-- mcp-name: io.github.ariaxhan/metabrain -->

Point Claude Code, Codex, or any MCP client at a metabrain file and the loop runs from inside the agent — no glue code.

```bash
pip install 'metabrain[mcp]'
claude mcp add metabrain -- metabrain-mcp --db ./agent.db
```

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.metabrain]
command = "metabrain-mcp"
args = ["--db", "./agent.db"]
```

`metabrain-mcp` speaks stdio, opens one shared `MetaBrain` on the `--db` path, and closes it on exit. Seven tools, thin wrappers over the library:

| Tool | Calls |
| --- | --- |
| `start_brief()` | `read_start()` — proven preferences first; run it before you work |
| `recall(query, limit=20)` | `recall()` |
| `learn(type, insight, domain?, context?)` | `learn()`; `type` is `failure` / `pattern` / `gotcha` / `preference` |
| `hypotheses(status?)` | `hypotheses()` |
| `verdict(result, unit?, evidence?, hypothesis?)` | `verdict()` — closes the loop |
| `stats()` | `stats()` |
| `capture_error(tool, error, context?)` | `capture_error()` |

The core package stays zero-dependency; the `mcp` SDK arrives only with the extra, and works on both `mcp` 1.x and 2.x.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT © Aria Han
