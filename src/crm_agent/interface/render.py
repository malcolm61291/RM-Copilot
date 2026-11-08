"""Rich terminal rendering shared by the CLI and the scripted demo."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from crm_agent.formatting import inr
from crm_agent.orchestration.nodes import summarise_offer

STATUS_STYLE = {"ready": "green", "blocked": "red", "needs_revision": "yellow", "sent": "cyan",
                "rejected": "dim", "approved": "cyan", "send_failed": "red"}


def render_turn(console: Console, state: dict, show_drafts: bool = True) -> None:
    turn = state.get("turn_intent") or {}
    console.print(Panel(state.get("response") or "(no response)", title="Agent", border_style="blue"))
    action = turn.get("action")
    if action in ("find_leads", "refine") and state.get("shortlist"):
        render_shortlist(console, state)
    if show_drafts and action in ("find_leads", "refine", "edit", "approve", "reject") and state.get("drafts"):
        render_drafts(console, state)
    if action in ("find_leads", "refine") and state.get("excluded"):
        t = Table(title="Not contacted", show_header=True, header_style="bold")
        for col in ("Customer", "Reason"):
            t.add_column(col)
        for x in state["excluded"]:
            t.add_row(f"{x['name']} ({x['customer_id']})", x["reason"])
        console.print(t)


def render_shortlist(console: Console, state: dict) -> None:
    feats, value, prop = state["features"], state["value"], state["propensity"]
    offers, alts = state.get("offers") or {}, state.get("alternatives") or {}
    t = Table(title="Shortlist", header_style="bold", show_lines=False)
    for col in ("#", "Customer", "City", "Segment", "Value", "P(convert)", "Top reason", "Recommended offer"):
        t.add_column(col, overflow="fold")
    for s in state["shortlist"]:
        cid = s["customer_id"]
        f, p = feats[cid], prop[cid]
        offer = summarise_offer(offers.get(cid)) if cid in offers else (
            f"[yellow]alt: {summarise_offer(alts[cid])}[/]" if cid in alts else "-")
        t.add_row(str(s["rank"]), f"{f['name']}\n[dim]{cid}[/]", f["city"], f["segment"],
                  f"{value[cid]['value_score']:.2f}", f"{p['probability']:.0%}",
                  (p["drivers"] or ["-"])[0], offer)
    console.print(t)


def render_drafts(console: Console, state: dict) -> None:
    for d in state["drafts"]:
        style = STATUS_STYLE.get(d["status"], "white")
        notes = ""
        if d["violations"]:
            notes = "\n\n[red]" + "\n".join(f"✗ {v['rule_id']}: {v['message']}" for v in d["violations"]) + "[/]"
        title = (f"#{d['no']} {d['name']} · {d['language']} · {d['method']} · "
                 f"[{style}]{d['status'].upper()}[/]" + (f" · {d['message_id']}" if d.get("message_id") else ""))
        console.print(Panel(d["body"] + notes, title=title, border_style=style, width=88))


def render_trace(console: Console, spans: list[dict], last: int = 40) -> None:
    t = Table(title="Trace (most recent spans)", header_style="bold")
    for col in ("kind", "name", "ms", "status", "detail"):
        t.add_column(col, overflow="fold")
    for s in spans[-last:]:
        detail = s.get("result") or s.get("attrs") or ""
        t.add_row(s["kind"], s["name"], str(s["ms"]), s.get("status", ""), str(detail)[:90])
    console.print(t)


def render_plan(console: Console, state: dict) -> None:
    t = Table(title="Plan for the last message", header_style="bold")
    for col in ("#", "step", "why", "ran"):
        t.add_column(col)
    executed = state.get("executed", [])
    for i, p in enumerate(state.get("plan") or [], 1):
        t.add_row(str(i), p["step"], p["why"], "✓" if p["step"] in executed else "")
    console.print(t)


__all__ = ["render_turn", "render_trace", "render_plan", "render_shortlist", "render_drafts", "inr"]
