"""Planner: decomposes an intent into an ordered, dependency-checked list of steps.

It re-plans incrementally: on a follow-up turn it works out which earlier results are still
valid (they live in graph state) and only re-runs the steps the change invalidates.
E.g. 'make them shorter' re-runs compose -> compliance only; 'only Mumbai' re-ranks without
recomputing features or propensity for the whole book.
"""
from __future__ import annotations

STEPS: dict[str, dict] = {
    "retrieve_customers": {"deps": [], "produces": "pool",
                           "why": "apply the RM's filters to the customer master"},
    "compute_features": {"deps": [], "produces": "features",
                         "why": "aggregate transactions, exposures and digital signals for the book"},
    "assess_value": {"deps": ["compute_features"], "produces": "value",
                     "why": "rank customers by relationship value to find high-value ones"},
    "check_eligibility": {"deps": ["compute_features"], "produces": "eligibility",
                          "why": "apply hard credit-policy rules for the product"},
    "score_propensity": {"deps": ["compute_features"], "produces": "propensity",
                         "why": "estimate likelihood to convert, with reason codes"},
    "rank_and_select": {"deps": ["retrieve_customers", "assess_value", "check_eligibility", "score_propensity"],
                        "produces": "shortlist", "why": "combine value and propensity into a shortlist"},
    "recommend_products": {"deps": ["rank_and_select"], "produces": "offers",
                           "why": "choose the product variant and size each offer"},
    "compose_messages": {"deps": ["recommend_products"], "produces": "drafts",
                         "why": "draft personalised WhatsApp messages (consent-checked)"},
    "compliance_check": {"deps": ["compose_messages"], "produces": "drafts_checked",
                         "why": "run guardrails; send failing drafts back for revision"},
}
PIPELINE = list(STEPS)

# Which intent fields invalidate which steps.
FIELD_TRIGGERS = {
    "filters": ["retrieve_customers"],
    "product_type": ["check_eligibility", "score_propensity"],
    "top_n": ["rank_and_select"],
    "min_value_tier": ["rank_and_select"],
    "language": ["compose_messages"],
    "tone": ["compose_messages"],
    "generate_messages": ["compose_messages"],
}


def _downstream(dirty: set[str]) -> set[str]:
    closure = set(dirty)
    changed = True
    while changed:
        changed = False
        for step, spec in STEPS.items():
            if step not in closure and any(d in closure for d in spec["deps"]):
                closure.add(step)
                changed = True
    return closure


def make_plan(intent: dict, previous: dict | None, available: set[str]) -> list[dict]:
    """Return [{step, why}] ending in 'present'. `available` = artefacts already in state."""
    action = intent["action"]
    if action in ("explain", "help", "show_outbox"):
        return [{"step": action, "why": f"answer the RM's {action.replace('_', ' ')} request"}]

    if action == "refine" and previous:
        dirty = {s for field, steps in FIELD_TRIGGERS.items()
                 if intent.get(field) != previous.get(field) for s in steps}
        dirty |= {s for s, spec in STEPS.items() if spec["produces"] not in available}
        if intent.get("generate_messages") and "drafts" not in available:
            dirty.add("compose_messages")
        steps = _downstream(dirty)
    else:
        steps = set(PIPELINE)
        if "features" in available:          # the book does not change within a session
            steps.discard("compute_features")
            steps.discard("assess_value")

    if not intent.get("generate_messages"):
        steps -= {"compose_messages", "compliance_check"}
    plan = [{"step": s, "why": STEPS[s]["why"]} for s in PIPELINE if s in steps]
    plan.append({"step": "present", "why": "summarise results for the RM"})
    _validate(plan, available)
    return plan


def _validate(plan: list[dict], available: set[str]) -> None:
    seen = {s for s, spec in STEPS.items() if spec["produces"] in available}
    for item in plan:
        step = item["step"]
        for dep in STEPS.get(step, {}).get("deps", []):
            if dep not in seen:
                raise ValueError(f"Invalid plan: '{step}' needs '{dep}' first")
        seen.add(step)
