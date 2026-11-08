"""LangGraph wiring.

    START -> understand -> [planned steps ...] -> present -> approval_gate (interrupt) -> understand ...

`understand` produces a plan; a single router walks it. Steps can rewind the cursor
(compliance -> compose self-correction) or replace the plan (review actions). State is
checkpointed per thread, which gives conversation memory and resumable human approval.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from crm_agent.orchestration import nodes
from crm_agent.orchestration.state import AgentState

STEP_NODES = {
    "retrieve_customers": nodes.retrieve_customers,
    "compute_features": nodes.compute_features,
    "assess_value": nodes.assess_value,
    "check_eligibility": nodes.check_eligibility,
    "score_propensity": nodes.score_propensity,
    "rank_and_select": nodes.rank_and_select,
    "recommend_products": nodes.recommend_products,
    "compose_messages": nodes.compose_messages,
    "compliance_check": nodes.compliance_check,
    "handle_review": nodes.handle_review,
    "send_outreach": nodes.send_outreach,
    "explain": nodes.explain,
    "help": nodes.help_,
    "show_outbox": nodes.show_outbox,
    "present": nodes.present,
}


def route(state: dict) -> str:
    plan, cursor = state.get("plan") or [], state.get("cursor", 0)
    return plan[cursor]["step"] if cursor < len(plan) else END


def after_present(state: dict) -> str:
    turn = state.get("turn_intent") or {}
    has_ready = any(d["status"] == nodes.READY for d in state.get("drafts") or [])
    if has_ready and turn.get("action") in ("find_leads", "refine", "approve", "reject", "edit"):
        return "approval_gate"
    return END


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("understand", nodes.understand)
    for name, fn in STEP_NODES.items():
        g.add_node(name, fn)
    g.add_node("approval_gate", nodes.approval_gate, destinations=("understand",))

    g.add_edge(START, "understand")
    targets = {name: name for name in STEP_NODES} | {END: END}
    g.add_conditional_edges("understand", route, targets)
    for name in STEP_NODES:
        if name != "present":
            g.add_conditional_edges(name, route, targets)
    g.add_conditional_edges("present", after_present, {"approval_gate": "approval_gate", END: END})
    return g.compile(checkpointer=checkpointer or InMemorySaver())
