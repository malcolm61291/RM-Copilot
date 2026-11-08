"""Conversational CLI for the Relationship Manager.

    python -m crm_agent.interface.cli

Commands: /plan  /trace  /state  /reset  /quit
"""
from __future__ import annotations

import argparse
import json

from rich.console import Console

from crm_agent.config import settings
from crm_agent.interface.render import render_plan, render_trace, render_turn
from crm_agent.orchestration.session import AgentSession


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--thread", default=None, help="Resume a conversation id")
    args = p.parse_args()

    console = Console()
    session = AgentSession(args.thread)
    console.print(f"[bold]RM Copilot[/] · {settings.bank_name} · LLM: {settings.llm_provider} · "
                  f"thread {session.thread_id}\nType a request, or 'help'. Commands: /plan /trace /state /reset /quit\n")
    while True:
        prompt = "[yellow]approval pending[/] RM> " if session.awaiting_approval else "RM> "
        try:
            text = console.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            session = AgentSession()
            console.print(f"New conversation {session.thread_id}")
            continue
        if text == "/trace":
            render_trace(console, session.trace())
            continue
        if text == "/plan":
            render_plan(console, session.state())
            continue
        if text == "/state":
            st = session.state()
            console.print_json(json.dumps({"intent": st.get("intent"), "funnel": st.get("funnel"),
                                           "drafts": [{k: d[k] for k in ("no", "name", "status")}
                                                      for d in st.get("drafts") or []]}, default=str))
            continue
        with console.status("Working…"):
            state = session.send(text)
        render_turn(console, state)


if __name__ == "__main__":
    main()
