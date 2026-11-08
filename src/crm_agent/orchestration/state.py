"""Graph state. Plain JSON-serialisable data only, so every step can be checkpointed,
resumed after an interrupt, replayed and audited."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    thread_id: str
    user_message: str
    history: Annotated[list[dict], operator.add]   # conversation memory (append-only)

    # Understanding and planning
    intent: dict              # the standing request (kept across refinements)
    turn_intent: dict         # what this particular message asked for
    prev_intent: dict
    plan: list[dict]          # [{step, why}]
    cursor: int               # index of the next plan step
    executed: list[str]       # steps run this turn (for the RM-facing "how I did this")

    # Analytical artefacts (reused across turns when still valid)
    pool: list[str]
    features: dict[str, dict]
    value: dict[str, dict]
    eligibility: dict[str, dict]
    propensity: dict[str, dict]
    scored_product: str
    min_propensity: float
    funnel: dict
    shortlist: list[dict]
    offers: dict[str, dict]
    alternatives: dict[str, dict]

    # Outreach
    drafts: list[dict]
    excluded: list[dict]
    pending_check: list[str]
    revision_ids: list[str]
    revision_round: int
    edit: dict | None

    # Output
    response: str
    response_kind: str
