"""Graph nodes. Each node is a thin step: read state -> call tools/agents -> return a state delta.
Business logic lives in tools/ and scoring/; language work lives in agents/."""
from __future__ import annotations

import functools
import re
from typing import Callable

from langgraph.types import Command, interrupt

from crm_agent.agents import explainer, planner
from crm_agent.agents.composer import compose
from crm_agent.agents.intent import understand as understand_intent
from crm_agent.config import settings
from crm_agent.formatting import inr, plain_dashes
from crm_agent.observability.tracer import span
from crm_agent.tools import ToolError, call_tool
from crm_agent.tools.crm_tools import repo

READY, NEEDS_REVISION, BLOCKED = "ready", "needs_revision", "blocked"
APPROVED, REJECTED, SENT, SEND_FAILED = "approved", "rejected", "sent", "send_failed"
FINAL = (SENT, REJECTED, SEND_FAILED)  # the RM has already decided on these drafts


def node(name: str) -> Callable:
    """Trace the node and advance the plan cursor."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            with span("node", name, cursor=state.get("cursor")) as s:
                delta = fn(state) or {}
                if "cursor" not in delta:
                    delta["cursor"] = state.get("cursor", 0) + 1
                delta["executed"] = state.get("executed", []) + [name]
                s["result"] = sorted(k for k in delta if k not in ("cursor", "executed"))
                return delta
        return wrapper
    return decorator


def _thread(state: dict) -> str:
    return state.get("thread_id", "default")


# --------------------------------------------------------------------------- understand + plan
REUSABLE_RESULTS = ("pool", "features", "value", "eligibility", "propensity", "shortlist", "offers", "drafts")


def _available(state: dict) -> set[str]:
    """Results from earlier turns that the planner may reuse instead of recomputing."""
    return {key for key in REUSABLE_RESULTS if state.get(key) is not None}


def understand(state: dict) -> dict:
    with span("node", "understand") as s:
        message = state["user_message"]
        previous = state.get("intent")
        intent = understand_intent(message, previous, state.get("history")).model_dump()
        if intent["action"] in ("approve", "reject", "edit"):
            plan = [{"step": "handle_review", "why": "apply the RM's review decision"}]
            intent_to_keep = previous or intent
        else:
            available = _available(state)
            if previous and intent["product_type"] != state.get("scored_product"):
                available -= {"eligibility", "propensity"}
            plan = planner.make_plan(intent, previous, available)
            intent_to_keep = intent if intent["action"] in ("find_leads", "refine") else (previous or intent)
        s["result"] = {"action": intent["action"], "parser": intent["parser"], "plan": [p["step"] for p in plan]}
        repo().audit(_thread(state), "agent", "plan_created",
                     {"message": message, "intent": intent, "plan": plan})
        return {
            "intent": intent_to_keep, "prev_intent": previous or {}, "plan": plan, "cursor": 0,
            "executed": [], "response": "", "response_kind": "", "revision_round": 0, "revision_ids": [],
            "pending_check": [], "edit": None, "turn_intent": intent,
            "history": [{"role": "rm", "content": message}],
        }


# --------------------------------------------------------------------------- analytical steps
@node("retrieve_customers")
def retrieve_customers(state: dict) -> dict:
    rows = call_tool("find_customers", filters=state["intent"].get("filters", {}))
    return {"pool": [r["customer_id"] for r in rows]}


@node("compute_features")
def compute_features(state: dict) -> dict:
    return {"features": call_tool("get_customer_features")}


@node("assess_value")
def assess_value(state: dict) -> dict:
    return {"value": call_tool("score_customer_value", features=state["features"])}


@node("check_eligibility")
def check_eligibility(state: dict) -> dict:
    product = state["intent"]["product_type"]
    return {"eligibility": call_tool("check_eligibility", features=state["features"], product_type=product),
            "scored_product": product}


@node("score_propensity")
def score_propensity(state: dict) -> dict:
    product = state["intent"]["product_type"]
    return {"propensity": call_tool("score_propensity", features=state["features"], product_type=product),
            "min_propensity": settings.propensity_threshold}


@node("rank_and_select")
def rank_and_select(state: dict) -> dict:
    intent = state["intent"]
    result = call_tool("rank_candidates", pool=state["pool"], value=state["value"],
                       eligibility=state["eligibility"], propensity=state["propensity"],
                       top_n=intent["top_n"], min_value_tier=intent["min_value_tier"],
                       min_propensity=settings.propensity_threshold)
    return {"shortlist": result["shortlist"], "funnel": result["funnel"]}


@node("recommend_products")
def recommend_products(state: dict) -> dict:
    offers, alternatives = {}, {}
    for item in state["shortlist"]:
        cid = item["customer_id"]
        rec = call_tool("recommend_product", features=state["features"][cid],
                        product_type=state["intent"]["product_type"])
        if rec["offer"]:
            offers[cid] = rec["offer"]
        elif rec["alternative"]:
            alternatives[cid] = rec["alternative"]
    return {"offers": offers, "alternatives": alternatives}


# --------------------------------------------------------------------------- outreach steps
def _language(intent: dict, f: dict) -> str:
    return f["preferred_language"] if intent.get("language", "auto") == "auto" else intent["language"]


@node("compose_messages")
def compose_messages(state: dict) -> dict:
    intent, feats = state["intent"], state["features"]
    revision_ids = state.get("revision_ids") or []
    edit = state.get("edit") or {}
    drafts = [dict(d) for d in state.get("drafts") or []] if revision_ids else []
    excluded = list(state.get("excluded") or []) if revision_ids else []
    targets = revision_ids or [s["customer_id"] for s in state["shortlist"]]
    pending = []

    for cid in targets:
        f = feats[cid]
        existing = next((d for d in drafts if d["customer_id"] == cid), None)
        if existing is None:
            consent = call_tool("check_contact_consent", customer=f, channel=intent["channel"])
            if not consent["allowed"]:
                excluded.append({"customer_id": cid, "name": f["name"], "reason": consent["reason"],
                                 "rule_id": consent["rule_id"]})
                continue
            offer = (state.get("offers") or {}).get(cid)
            if offer is None:
                alt = (state.get("alternatives") or {}).get(cid)
                excluded.append({"customer_id": cid, "name": f["name"], "rule_id": "NO_VIABLE_OFFER",
                                 "reason": "No viable offer for the requested product"
                                           + (f"; consider {alt['product_name']}" if alt else "")})
                continue
        else:
            offer = existing["offer"]

        language = existing["language"] if existing else _language(intent, f)
        if edit.get("customer_id") == cid and edit.get("instruction"):
            ins = edit["instruction"].lower()
            language = "hi" if "hindi" in ins else "en" if "english" in ins else language
        if edit.get("customer_id") == cid and edit.get("text"):
            body, method = plain_dashes(edit["text"]).strip(), "rm_edit"
        else:
            feedback = existing["violations"] if existing and existing["status"] == NEEDS_REVISION else None
            instruction = edit.get("instruction") if edit.get("customer_id") == cid else None
            body, method = compose(f, offer, language, intent.get("tone", "warm"), instruction, feedback)

        if existing:
            existing.update(body=body, method=method, status="checking", attempts=existing["attempts"] + 1)
        else:
            drafts.append({"no": len(drafts) + 1, "customer_id": cid, "name": f["name"],
                           "first_name": f["first_name"], "language": language, "offer": offer,
                           "body": body, "method": method, "status": "checking", "violations": [],
                           "attempts": 1, "message_id": None})
        pending.append(cid)
    return {"drafts": drafts, "excluded": excluded, "pending_check": pending, "revision_ids": []}


@node("compliance_check")
def compliance_check(state: dict) -> dict:
    drafts = [dict(d) for d in state["drafts"]]
    pending = set(state.get("pending_check") or [])
    revise = []
    round_ = state.get("revision_round", 0)
    for d in drafts:
        if d["customer_id"] not in pending:
            continue
        result = call_tool("validate_message", body=d["body"], offer=d["offer"], first_name=d["first_name"])
        d["violations"] = result["violations"]
        if result["passed"]:
            d["status"] = READY
            continue
        repo().audit(_thread(state), "guardrail", "draft_failed_checks",
                     {"customer_id": d["customer_id"], "violations": result["violations"], "method": d["method"]})
        if d["method"] == "rm_edit":
            d["status"] = BLOCKED            # never silently rewrite a human's words
        elif round_ < settings.max_revision_rounds:
            d["status"] = NEEDS_REVISION
            revise.append(d["customer_id"])
        else:
            d["status"] = BLOCKED
    delta = {"drafts": drafts, "pending_check": []}
    if revise:
        # Self-correction loop: jump back to compose with the violations as feedback.
        compose_idx = next(i for i, p in enumerate(state["plan"]) if p["step"] == "compose_messages")
        delta.update(revision_ids=revise, revision_round=round_ + 1, cursor=compose_idx)
    return delta


# --------------------------------------------------------------------------- human in the loop
def approval_gate(state: dict) -> Command:
    """Pause the graph until the RM reviews the drafts (LangGraph interrupt = durable HITL)."""
    ready = [d["no"] for d in state["drafts"] if d["status"] == READY]
    answer = interrupt({"type": "approval_required", "ready_drafts": ready,
                        "hint": "approve 1,3 | approve all | reject 2 | edit 2: make it shorter | "
                                "edit 2 text: <your message> | or ask anything else"})
    return Command(goto="understand", update={"user_message": str(answer)})


@node("handle_review")
def handle_review(state: dict) -> dict:
    turn = state.get("turn_intent") or {}
    action, numbers = turn.get("action"), turn.get("draft_numbers") or []
    drafts = [dict(d) for d in state.get("drafts") or []]
    if not drafts:
        return {"plan": [{"step": "present", "why": "nothing to review"}], "cursor": 0,
                "response": "There are no drafts to review yet. Ask me to find customers first."}
    chosen = [d for d in drafts if not numbers or d["no"] in numbers]
    unknown = sorted(set(numbers) - {d["no"] for d in drafts})
    notes = [f"No draft #{n}." for n in unknown]
    thread = _thread(state)
    report_only = [{"step": "present", "why": "report"}]

    if action == "approve":
        for d in chosen:
            if d["status"] == READY:
                d["status"] = APPROVED
                repo().audit(thread, settings.rm_id, "draft_approved", {"customer_id": d["customer_id"]})
            elif d["status"] in (BLOCKED, NEEDS_REVISION):
                notes.append(f"#{d['no']} {d['name']} is blocked by compliance and cannot be approved.")
            elif numbers and d["status"] in FINAL:
                notes.append(f"#{d['no']} {d['name']} was already {d['status']}.")
        plan = [{"step": "send_outreach", "why": "queue approved messages"}, {"step": "present", "why": "report"}]
        return {"drafts": drafts, "plan": plan, "cursor": 0, "response": " ".join(notes)}

    if action == "reject":
        rejected = []
        for d in chosen:
            if d["status"] in (READY, BLOCKED, NEEDS_REVISION):
                d["status"] = REJECTED
                rejected.append(f"#{d['no']}")
                repo().audit(thread, settings.rm_id, "draft_rejected", {"customer_id": d["customer_id"]})
            elif numbers and d["status"] in FINAL:
                notes.append(f"#{d['no']} {d['name']} was already {d['status']}, so it can't be rejected.")
        if rejected:
            notes.insert(0, f"Rejected {', '.join(rejected)}.")
        return {"drafts": drafts, "plan": report_only, "cursor": 0, "response": " ".join(notes)}

    # edit
    if not chosen or unknown:
        return {"plan": report_only, "cursor": 0, "response": "Which draft? e.g. `edit 2: make it more formal`."}
    target = chosen[0]
    if target["status"] in (APPROVED, SENT):
        # Re-composing would reset the status to ready and let the same customer be messaged twice.
        return {"plan": report_only, "cursor": 0,
                "response": f"#{target['no']} {target['name']} has already been sent, so it can't be edited."}
    edit = {"customer_id": target["customer_id"], "instruction": turn.get("instruction"),
            "text": turn.get("edit_text")}
    repo().audit(thread, settings.rm_id, "draft_edit_requested", edit)
    plan = [{"step": "compose_messages", "why": "rewrite the draft as instructed"},
            {"step": "compliance_check", "why": "re-check the edited draft"},
            {"step": "present", "why": "show the updated draft"}]
    return {"edit": edit, "revision_ids": [target["customer_id"]], "revision_round": 0, "plan": plan, "cursor": 0}


@node("send_outreach")
def send_outreach(state: dict) -> dict:
    drafts = [dict(d) for d in state["drafts"]]
    feats, thread = state["features"], _thread(state)
    queued = duplicates = failed = 0
    for d in drafts:
        if d["status"] != APPROVED:
            continue
        try:
            res = call_tool("send_whatsapp_message", customer=feats[d["customer_id"]], body=d["body"],
                            offer=d["offer"], approved_by=settings.rm_id, thread_id=thread)
        except ToolError as exc:
            d["status"], d["violations"] = SEND_FAILED, [{"rule_id": "SEND_BLOCKED", "message": str(exc)}]
            failed += 1
            continue
        d["status"], d["message_id"] = SENT, res["message_id"]
        if res["status"] == "duplicate_skipped":
            duplicates += 1
        else:
            queued += 1

    msg = f"Queued {queued} message(s) on WhatsApp."
    if duplicates:
        msg += f" {duplicates} skipped because the identical message was already sent to that customer."
    if failed:
        msg += f" {failed} blocked at send."
    return {"drafts": drafts, "response": f"{state.get('response') or ''} {msg}".strip()}


# --------------------------------------------------------------------------- answers
def _resolve_target(target: str | None, state: dict) -> str | None:
    if not target:
        return None
    shortlist = state.get("shortlist") or []
    if m := re.fullmatch(r"#?(\d{1,2})", target.strip()):
        n = int(m.group(1))
        return next((s["customer_id"] for s in shortlist if s["rank"] == n), None)
    if re.fullmatch(r"C\d{4}", target.strip(), re.IGNORECASE):
        return target.strip().upper()
    words = [w for w in re.findall(r"[A-Za-z']+", target.lower()) if len(w) > 1]
    feats = state.get("features") or {}
    ranked_ids = [s["customer_id"] for s in shortlist] + list(feats)
    for cid in ranked_ids:
        name = feats[cid]["name"].lower()
        if words and all(w in name for w in words):
            return cid
    for cid in ranked_ids:                       # fall back to any-word match, shortlist first
        if any(w in feats[cid]["name"].lower().split() for w in words):
            return cid
    return None


@node("explain")
def explain(state: dict) -> dict:
    turn = state.get("turn_intent") or {}
    delta: dict = {}
    if not state.get("propensity"):
        product = (state.get("intent") or {}).get("product_type", "personal_loan")
        feats = call_tool("get_customer_features")
        delta = {"features": feats,
                 "value": call_tool("score_customer_value", features=feats),
                 "eligibility": call_tool("check_eligibility", features=feats, product_type=product),
                 "propensity": call_tool("score_propensity", features=feats, product_type=product),
                 "scored_product": product, "min_propensity": settings.propensity_threshold}
    merged = {**state, **delta}
    cid = _resolve_target(turn.get("target"), merged)
    if cid is None or cid not in merged["features"]:
        delta["response"] = (f"I couldn't find a customer matching '{turn.get('target')}'. "
                             "Try a customer id (e.g. C0123), a full name, or a shortlist number (#2).")
    else:
        delta["response"] = explainer.explain(cid, merged)
    delta["response_kind"] = "explain"
    return delta


@node("help")
def help_(state: dict) -> dict:
    return {"response_kind": "help", "response": (
        "I help you find and engage the right customers. Try:\n"
        "• Find high-value customers likely to convert for a personal loan this month and draft WhatsApp messages\n"
        "• Only salaried customers in Mumbai, top 5, in Hindi\n"
        "• Why is #2 on the list? / Why not C0042?\n"
        "• approve 1,3 | reject 2 | edit 2: make it more formal | edit 2 text: <your own message>\n"
        "• Which customers should I pitch a credit card to?\n"
        "• Show the outbox")}


@node("show_outbox")
def show_outbox(state: dict) -> dict:
    rows = repo().outbox(_thread(state))
    if not rows:
        return {"response": "Nothing queued in this session yet.", "response_kind": "outbox"}
    lines = [f"• {r['message_id']} → {r['customer_id']} ({r['status']}, approved by {r['approved_by']})" for r in rows]
    return {"response": "Queued messages:\n" + "\n".join(lines), "response_kind": "outbox"}


@node("present")
def present(state: dict) -> dict:
    turn = state.get("turn_intent") or {}
    parts = [state.get("response", "").strip()] if state.get("response") else []
    if turn.get("action") in ("find_leads", "refine"):
        intent, funnel = state["intent"], state.get("funnel", {})
        parts.insert(0, f"Understood: {turn.get('understood_as') or intent.get('understood_as')} "
                        f"[parser: {turn.get('parser')}]")
        parts.append(
            "Funnel: {in_scope} in scope → {value_ok} {tier}-value → {eligible} credit-eligible → "
            "{likely_to_convert} likely to convert → {selected} shortlisted".format(
                tier=intent["min_value_tier"], **{k: funnel.get(k, 0) for k in
                                                  ("in_scope", "value_ok", "eligible", "likely_to_convert",
                                                   "selected")}))
        if not state.get("shortlist"):
            parts.append("No one met all the criteria. Try `include medium value` or widen the filters.")
    drafts = state.get("drafts") or []
    edited_id = (state.get("edit") or {}).get("customer_id")
    if turn.get("action") == "edit" and edited_id:
        d = next(d for d in drafts if d["customer_id"] == edited_id)
        outcome = {READY: "it passed compliance and is ready for approval",
                   BLOCKED: "it failed compliance: " + "; ".join(v["message"] for v in d["violations"])}
        parts.insert(0, f"Updated draft #{d['no']} for {d['name']}; "
                        f"{outcome.get(d['status'], 'status: ' + d['status'])}.")
    counts = {s: sum(d["status"] == s for d in drafts) for s in (READY, BLOCKED, SENT, REJECTED)}
    if drafts and turn.get("action") not in ("explain",):
        parts.append(f"Drafts: {counts[READY]} ready for your approval, {counts[BLOCKED]} blocked by compliance, "
                     f"{counts[SENT]} sent, {counts[REJECTED]} rejected.")
    if state.get("excluded") and turn.get("action") in ("find_leads", "refine"):
        parts.append(f"{len(state['excluded'])} shortlisted customer(s) not contacted (consent/DND/no offer).")
    executed = [s for s in state.get("executed", []) if s not in ("present",)]
    if executed:
        parts.append("Steps run this turn: " + " → ".join(executed))
    text = plain_dashes("\n".join(p for p in parts if p))
    return {"response": text, "response_kind": state.get("response_kind") or "leads",
            "history": [{"role": "agent", "content": text}], "cursor": len(state["plan"])}


def summarise_offer(o: dict) -> str:
    if not o:
        return "-"
    if o.get("rate") is None:
        return f"{o['product_name']} (limit ₹{inr(o['amount'])})"
    return f"{o['product_name']} ₹{inr(o['amount'])} @ {o['rate']}% · EMI ₹{inr(o['emi'])}"
