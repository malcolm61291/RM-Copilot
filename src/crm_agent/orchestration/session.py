"""One RM conversation. Hides the invoke-vs-resume detail from every interface (CLI, UI, tests)."""
from __future__ import annotations

import uuid

from langgraph.types import Command

from crm_agent.observability import tracer
from crm_agent.orchestration.graph import build_graph

_GRAPH = None


def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


class AgentSession:
    def __init__(self, thread_id: str | None = None):
        self.thread_id = thread_id or f"rm-{uuid.uuid4().hex[:8]}"
        self.config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 60}

    @property
    def awaiting_approval(self) -> bool:
        return bool(graph().get_state(self.config).interrupts)

    def send(self, message: str) -> dict:
        tracer.set_thread(self.thread_id)
        if self.awaiting_approval:
            graph().invoke(Command(resume=message), self.config)
        else:
            graph().invoke({"user_message": message, "thread_id": self.thread_id}, self.config)
        return self.state()

    def state(self) -> dict:
        return dict(graph().get_state(self.config).values)

    def trace(self) -> list[dict]:
        return tracer.spans(self.thread_id)
