"""Explains, in RM language, why a customer was or was not shortlisted.

Facts come from the stored scoring artefacts (so the explanation is faithful to the actual
decision); the LLM, if configured, only rephrases them.
"""
from __future__ import annotations

import json

from crm_agent.formatting import inr, plain_dashes
from crm_agent.llm.provider import LLMError, get_llm
from crm_agent.scoring.value import meets_tier


def decision_record(cid: str, state: dict) -> dict:
    f = state["features"][cid]
    value = state["value"][cid]
    elig = state["eligibility"][cid]
    prop = state["propensity"][cid]
    shortlist = {c["customer_id"]: c for c in state.get("shortlist", [])}
    in_pool = cid in set(state.get("pool", []))
    intent = state.get("intent") or {}

    if cid in shortlist:
        outcome = f"Shortlisted at rank #{shortlist[cid]['rank']} (priority score {shortlist[cid]['priority_score']})"
    elif not in_pool:
        outcome = "Not in scope: excluded by the current filters " + json.dumps(intent.get("filters", {}))
    elif not meets_tier(value["value_tier"], intent.get("min_value_tier", "high")):
        outcome = (f"Not shortlisted: value tier is '{value['value_tier']}' (score {value['value_score']}, "
                   f"high-value cut-off {value['value_cutoff']})")
    elif not elig["eligible"]:
        outcome = "Not shortlisted: failed credit policy " + ", ".join(elig["failed_rules"])
    elif prop["probability"] < state.get("min_propensity", 0.5):
        outcome = f"Not shortlisted: conversion likelihood {prop['probability']:.0%} is below the threshold"
    else:
        outcome = "Qualified but ranked below the top-N cut-off"

    draft = next((d for d in state.get("drafts", []) if d["customer_id"] == cid), None)
    excluded = next((x for x in state.get("excluded", []) if x["customer_id"] == cid), None)
    return {
        "customer": f"{f['name']} ({cid})", "outcome": outcome,
        "value": f"{value['value_tier']} (score {value['value_score']}); {f['segment']} segment, "
                 f"relationship value ₹{inr(f['relationship_value'])}, {f['tenure_years']:.1f} yrs with bank",
        "eligibility": [f"{'PASS' if c['passed'] else 'FAIL'} {c['rule_id']}: {c['detail']}" for c in elig["checks"]],
        "propensity": f"{prop['probability']:.0%} ({prop['method']})",
        "drivers": prop["drivers"], "watch_outs": prop["watch_outs"],
        "offer": (state.get("offers") or {}).get(cid),
        "contact": excluded["reason"] if excluded else ("draft ready" if draft else None),
    }


def explain(cid: str, state: dict) -> str:
    rec = decision_record(cid, state)
    llm = get_llm()
    if llm is not None:
        try:
            return plain_dashes(llm.complete(
                "Explain to a bank Relationship Manager why this customer was or was not shortlisted. "
                "Use only the facts given. 5 to 8 short bullet lines, plain language, no new numbers, "
                "and no em or en dashes.",
                json.dumps(rec, ensure_ascii=False, default=str), max_tokens=400, temperature=0.2))
        except LLMError:
            pass
    return format_record(rec)


def format_record(rec: dict) -> str:
    lines = [f"**{rec['customer']}**: {rec['outcome']}",
             f"• Value: {rec['value']}",
             f"• Conversion likelihood: {rec['propensity']}"]
    if rec["drivers"]:
        lines.append("• Why they may convert now: " + "; ".join(rec["drivers"]))
    if rec["watch_outs"]:
        lines.append("• Watch-outs: " + "; ".join(rec["watch_outs"]))
    lines.append("• Credit policy: " + " | ".join(rec["eligibility"]))
    if rec["offer"]:
        o = rec["offer"]
        detail = f"up to ₹{inr(o['amount'])}" + (f" @ {o['rate']}% for {o['tenure_months']}m" if o.get("rate") else "")
        lines.append(f"• Recommended: {o['product_name']} {detail} ({o['rationale']})")
    if rec["contact"]:
        lines.append(f"• Outreach: {rec['contact']}")
    return "\n".join(lines)
