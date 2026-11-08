"""Scripted end-to-end demo of the main use cases (used for the demo video).

    python scripts/demo.py            # all scenarios
    python scripts/demo.py --pause    # wait for Enter between turns (for screen recording)
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.rule import Rule

from crm_agent.interface.render import render_plan, render_trace, render_turn
from crm_agent.orchestration.session import AgentSession

SCENARIOS = [
    ("1 · Core ask: find, score, recommend and draft", [
        "Find high-value customers likely to convert for a personal loan this month and generate "
        "personalized WhatsApp messages.",
    ]),
    ("2 · Conversation memory: refine without starting over", [
        "Only salaried customers in Mumbai, top 5, in Hindi",
        "Why is #1 on the list?",
    ]),
    ("3 · Explainability: why was someone left out?", [
        "Why not {excluded}?",
    ]),
    ("4 · Human-in-the-loop and guardrails", [
        "edit 2: make it shorter",
        "edit 3 text: Hi, your loan approval is guaranteed with no documents! Reply STOP to opt out",
        "approve 1,2,3",
        "show the outbox",
    ]),
    ("5 · Extensibility: a different product, same agent", [
        "Which customers should I pitch a credit card to? Top 3, draft messages",
    ]),
]


def _interesting_rejection(state: dict) -> str | None:
    """A high-value customer in the current pool who failed credit policy: a good 'why not?'."""
    if not state.get("eligibility") or (state.get("turn_intent") or {}).get("action") != "refine":
        return None
    for cid in state.get("pool", []):
        if state["value"][cid]["value_tier"] == "high" and not state["eligibility"][cid]["eligible"]:
            return state["features"][cid]["name"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--trace", action="store_true", help="Print the span trace after each turn")
    args = ap.parse_args()
    console = Console(width=120)
    session = AgentSession("demo")
    excluded_name = None

    for title, turns in SCENARIOS:
        console.print(Rule(f"[bold]{title}"))
        for text in turns:
            if "{excluded}" in text:
                text = text.format(excluded=excluded_name or "C0042")
            if args.pause:
                console.input("[dim]Enter ▶[/]")
            console.print(f"\n[bold yellow]RM>[/] {text}")
            state = session.send(text)
            render_turn(console, state)
            excluded_name = excluded_name or _interesting_rejection(state)
            if "personal loan" in text.lower() and text.startswith("Find"):
                render_plan(console, state)
            if args.trace:
                render_trace(console, session.trace(), last=15)


if __name__ == "__main__":
    main()
