"""metabrain — a SQLite-backed memory layer with a self-improving loop.

Not just a place to store what you tell it. metabrain *discovers what works*:
a pattern you record enough times graduates into a hypothesis, every verdict on
it is an experiment, and once the evidence clears the bar it graduates again
into a proven ``preference``. That loop — learn → prove → graduate — is what
sets it apart from a plain memory store, and it runs on stdlib ``sqlite3`` with
zero dependencies and no server.

Using the API correctly fills every table as a side effect: open a session and
each write inherits its id, emits an event, and turns the loop.

    from metabrain import MetaBrain

    db = MetaBrain("agent.db")
    with db.session(task="content") as s:
        s.learn("pattern", "question hooks lift saves", domain="ig")  # ×3 → hypothesis
        h = db.hypotheses(status="testing")[0]
        u = s.unit("post a question-hook carousel", kind="contract", hypothesis=h.id)
        s.verdict("pass", unit=u, evidence="1.2k saves")             # experiment → ...
    brief = db.read_start()   # ...and proven preferences surface first
"""

from .db import (
    DEFAULT_GRADUATE_AT,
    DEFAULT_MIN_EXPERIMENTS,
    DEFAULT_PROMOTE_AT,
    MetaBrain,
    Session,
)
from .models import (
    ContextEntry,
    ErrorRecord,
    Experiment,
    Hypothesis,
    Learning,
    StartBrief,
)
from .schema import IncompatibleDatabaseError

__version__ = "1.1.1"

__all__ = [
    "MetaBrain",
    "Session",
    "Learning",
    "ContextEntry",
    "Hypothesis",
    "Experiment",
    "ErrorRecord",
    "StartBrief",
    "IncompatibleDatabaseError",
    "DEFAULT_PROMOTE_AT",
    "DEFAULT_GRADUATE_AT",
    "DEFAULT_MIN_EXPERIMENTS",
    "__version__",
]
