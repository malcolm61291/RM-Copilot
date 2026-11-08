"""Offline evaluation harness: is the agent selecting well, writing safely and running fast?

    python scripts/evaluate.py

Reports (1) propensity-model quality on a holdout, (2) lift of the shortlist vs the book,
(3) guardrail pass rate and personalisation of drafts, (4) latency per step from the trace.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from crm_agent.config import settings
from crm_agent.orchestration.session import AgentSession

PROMPTS = [
    "Find high-value customers likely to convert for a personal loan this month and generate personalized "
    "WhatsApp messages.",
    "only salaried customers in Mumbai, top 5, in Hindi",
    "make them more formal",
    "Which customers should I pitch a credit card to? top 5, draft messages",
]


def main() -> None:
    console = Console()
    card = json.loads((settings.model_dir / "propensity_personal_loan_card.json").read_text())
    t = Table(title="1 · Propensity model (holdout)")
    for k in card["metrics"]:
        t.add_column(k)
    t.add_row(*[str(v) for v in card["metrics"].values()])
    console.print(t)

    session = AgentSession("eval")
    drafts_seen, ready, unique_bodies, named, lang_ok = 0, 0, set(), 0, 0
    lift_rows = []
    for prompt in PROMPTS:
        st = session.send(prompt)
        if st.get("shortlist"):
            prop = st["propensity"]
            short = [prop[s["customer_id"]]["probability"] for s in st["shortlist"]]
            pool = [prop[c]["probability"] for c in st["pool"]]
            lift_rows.append((prompt[:48] + "…", f"{statistics.mean(short):.2f}", f"{statistics.mean(pool):.2f}",
                              f"{statistics.mean(short) / statistics.mean(pool):.1f}x"))
        for d in st.get("drafts") or []:
            drafts_seen += 1
            ready += d["status"] in ("ready", "sent")
            unique_bodies.add(d["body"])
            named += d["first_name"] in d["body"]
            want = st["intent"].get("language", "auto")
            pref = st["features"][d["customer_id"]]["preferred_language"]
            lang_ok += d["language"] == (pref if want == "auto" else want)

    t = Table(title="2 · Targeting lift (mean P(convert): shortlist vs in-scope pool)")
    for col in ("request", "shortlist", "pool", "lift"):
        t.add_column(col)
    for row in lift_rows:
        t.add_row(*row)
    console.print(t)

    t = Table(title="3 · Outreach quality")
    for col in ("drafts", "passed guardrails", "unique messages", "addressed by name", "language correct"):
        t.add_column(col)
    t.add_row(str(drafts_seen), f"{ready / drafts_seen:.0%}", f"{len(unique_bodies) / drafts_seen:.0%}",
              f"{named / drafts_seen:.0%}", f"{lang_ok / drafts_seen:.0%}")
    console.print(t)

    timings = defaultdict(list)
    for s in session.trace():
        timings[(s["kind"], s["name"])].append(s["ms"])
    t = Table(title="4 · Latency by span (ms)")
    for col in ("kind", "name", "calls", "p50", "max"):
        t.add_column(col)
    for (kind, name), ms in sorted(timings.items(), key=lambda kv: -max(kv[1]))[:12]:
        t.add_row(kind, name, str(len(ms)), f"{statistics.median(ms):.0f}", f"{max(ms):.0f}")
    console.print(t)


if __name__ == "__main__":
    main()
