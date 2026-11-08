"""Intent understanding: turn an RM's chat message into a validated, structured `Intent`.

The LLM (when configured) does the language understanding; Pydantic validates the result;
a rule-based parser is the fallback so the system degrades gracefully, never silently.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from crm_agent.config import settings
from crm_agent.llm.provider import LLMError, get_llm, parse_json

Action = Literal["find_leads", "refine", "explain", "approve", "reject", "edit", "show_outbox", "help"]
ProductType = Literal["personal_loan", "credit_card", "loan_against_fd"]

CITIES = ["mumbai", "pune", "bengaluru", "bangalore", "delhi", "hyderabad", "chennai"]


class Intent(BaseModel):
    action: Action
    product_type: ProductType = "personal_loan"
    filters: dict = Field(default_factory=dict)
    top_n: int = Field(settings.default_top_n, ge=1, le=50)
    min_value_tier: Literal["high", "medium", "standard"] = "high"
    timeframe_days: int = 30
    generate_messages: bool = True
    channel: Literal["whatsapp"] = "whatsapp"
    language: Literal["auto", "en", "hi"] = "auto"
    tone: Literal["warm", "formal", "concise"] = "warm"
    target: str | None = None            # customer for explain
    draft_numbers: list[int] = Field(default_factory=list)  # [] with approve/reject = all
    instruction: str | None = None       # free-text edit instruction
    edit_text: str | None = None         # RM-written replacement message
    understood_as: str = ""              # one-line paraphrase shown to the RM
    parser: str = "rules"


SYSTEM = f"""You convert a bank Relationship Manager's chat message into a JSON intent for a CRM agent.
Fields: action (find_leads | refine | explain | approve | reject | edit | show_outbox | help),
product_type (personal_loan | credit_card | loan_against_fd), filters (object with optional keys city [list],
segment [mass|affluent|priority], occupation_type [salaried|self_employed], min_age, max_age),
top_n (int), min_value_tier (high|medium|standard), timeframe_days (int), generate_messages (bool),
language (auto|en|hi), tone (warm|formal|concise), target (customer id or name, for explain),
draft_numbers (list of ints, for approve/reject/edit; empty list means all), instruction (edit instruction),
edit_text (exact replacement text if the RM wrote the message), understood_as (one short sentence).
Rules: 'refine' when the message modifies the previous request (e.g. 'only Mumbai', 'make them shorter',
'top 5'); copy unchanged fields from the previous intent. 'high-value' means min_value_tier=high.
'this month' means timeframe_days=30. Default top_n is {settings.default_top_n}. Never invent customers."""


def understand(message: str, previous: dict | None, history: list[dict] | None = None) -> Intent:
    llm = get_llm()
    if llm is not None:
        try:
            user = json.dumps({"message": message, "previous_intent": previous,
                               "recent_turns": (history or [])[-4:]}, ensure_ascii=False)
            data = parse_json(llm.complete(SYSTEM, user, json_mode=True, max_tokens=500, temperature=0))
            data["parser"] = f"llm:{llm.name}"
            return Intent(**data)
        except (LLMError, ValidationError, TypeError) as exc:
            fallback = parse_rules(message, previous)
            fallback.parser = f"rules (LLM fallback: {type(exc).__name__})"
            return fallback
    return parse_rules(message, previous)


# ---------------------------------------------------------------------------------------
# Rule-based parser (offline mode and LLM fallback)
# ---------------------------------------------------------------------------------------
def _numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\b(\d{1,2})\b", text)]


def parse_rules(message: str, previous: dict | None) -> Intent:
    text = message.strip()
    t = text.lower()

    if re.match(r"^\s*(approve|send|go ahead)\b", t):
        nums = [] if re.search(r"\ball\b", t) else _numbers(t)
        return Intent(action="approve", draft_numbers=nums, understood_as=f"Approve drafts {nums or 'all'}")
    if re.match(r"^\s*(reject|drop|remove|skip|discard)\b", t):
        nums = [] if re.search(r"\ball\b", t) else _numbers(t)
        return Intent(action="reject", draft_numbers=nums, understood_as=f"Reject drafts {nums or 'all'}")
    m = re.match(r"^\s*edit\s+#?(\d+)\s*(text)?\s*[:\-]?\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        num, is_text, rest = int(m.group(1)), bool(m.group(2)), m.group(3).strip()
        return Intent(action="edit", draft_numbers=[num], edit_text=rest if is_text else None,
                      instruction=None if is_text else rest, understood_as=f"Edit draft {num}")
    if re.search(r"\b(outbox|what (did|have) (we|i) sen[dt]|sent messages)\b", t):
        return Intent(action="show_outbox", understood_as="Show queued messages")
    if re.search(r"\b(why|explain|reason)\b", t):
        target = None
        if cid := re.search(r"\bC\d{4}\b", text, re.IGNORECASE):
            target = cid.group(0).upper()
        else:
            cleaned = re.sub(r"(?i)\b(why|explain|reason|is|was|wasn't|isn't|not|did|didn't|on|in|the|list|"
                             r"shortlist|picked|selected|included|excluded|make|it|for|me|customer|number|#|"
                             r"\?|ranked|so|high|low|get)\b", " ", text)
            cleaned = re.sub(r"[^A-Za-z' ]", " ", cleaned).strip()
            nums = _numbers(t)
            target = cleaned if cleaned else (f"#{nums[0]}" if nums else None)
        return Intent(action="explain", target=target, understood_as=f"Explain the decision for {target}")
    if re.search(r"\b(help|what can you do)\b", t) and len(t) < 40:
        return Intent(action="help")

    product = None
    if re.search(r"credit card|\bcc\b", t):
        product = "credit_card"
    elif re.search(r"(loan against|against (my |their )?(fd|fixed deposit)|\blafd\b)", t):
        product = "loan_against_fd"
    elif re.search(r"personal loan|\bpl\b|\bloan\b", t):
        product = "personal_loan"

    mentioned: dict = {}
    filters: dict = {}
    cities = [("Bengaluru" if c == "bangalore" else c.title()) for c in CITIES if re.search(rf"\b{c}\b", t)]
    if cities:
        filters["city"] = sorted(set(cities))
    if re.search(r"self[- ]employed|business ?owners?", t):
        filters["occupation_type"] = "self_employed"
    elif re.search(r"\bsalaried\b", t):
        filters["occupation_type"] = "salaried"
    for seg in ("priority", "affluent", "mass"):
        if re.search(rf"\b{seg}\b", t):
            filters["segment"] = seg
    if m := re.search(r"(?:under|below|younger than)\s+(\d{2})", t):
        filters["max_age"] = int(m.group(1))
    if m := re.search(r"(?:over|above|older than)\s+(\d{2})", t):
        filters["min_age"] = int(m.group(1))
    if re.search(r"\b(all cities|any city|everyone|remove (the )?filters?|clear filters?)\b", t):
        mentioned["filters"] = {}
    elif filters:
        mentioned["filters"] = filters
    if m := re.search(r"\btop\s*(\d{1,2})\b|\b(\d{1,2})\s+(?:customers|leads|people|names)\b", t):
        mentioned["top_n"] = int(m.group(1) or m.group(2))
    if re.search(r"\bhigh[- ]value\b|\bhnis?\b|\bvaluable\b", t):
        mentioned["min_value_tier"] = "high"
    elif re.search(r"\b(any value|all customers|including medium|medium value)\b", t):
        mentioned["min_value_tier"] = "medium"
    if re.search(r"\bhindi\b", t):
        mentioned["language"] = "hi"
    elif re.search(r"\benglish\b", t):
        mentioned["language"] = "en"
    elif re.search(r"preferred language|their language", t):
        mentioned["language"] = "auto"
    if re.search(r"\b(shorter|short|concise|brief|crisp)\b", t):
        mentioned["tone"] = "concise"
    elif re.search(r"\bformal\b", t):
        mentioned["tone"] = "formal"
    elif re.search(r"\b(warm|friendly|casual)\b", t):
        mentioned["tone"] = "warm"
    wants_msgs = bool(re.search(r"\b(message|messages|whatsapp|outreach|draft|pitch|write)\b", t))
    if wants_msgs:
        mentioned["generate_messages"] = True
    if re.search(r"\b(no messages|without messages|just the list|only the list)\b", t):
        mentioned["generate_messages"] = False

    is_new_request = product is not None or re.search(r"\b(find|identify|show me|list|who should|which customers)\b", t)
    if previous and (not is_new_request or (product is None and mentioned)):
        merged = {**previous, **mentioned, "action": "refine", "parser": "rules"}
        if "filters" in mentioned and mentioned["filters"]:
            merged["filters"] = {**previous.get("filters", {}), **mentioned["filters"]}
        merged["understood_as"] = "Refine previous request: " + (", ".join(f"{k}={v}" for k, v in mentioned.items())
                                                                  or "re-run")
        merged.pop("target", None)
        return Intent(**{k: v for k, v in merged.items() if k in Intent.model_fields})
    if is_new_request or mentioned:
        intent = Intent(action="find_leads", product_type=product or "personal_loan",
                        generate_messages=wants_msgs, **{k: v for k, v in mentioned.items()
                                                         if k != "generate_messages"})
        intent.understood_as = (f"Find {intent.min_value_tier}-value customers likely to take a "
                                f"{intent.product_type.replace('_', ' ')} in the next {intent.timeframe_days} days"
                                + (f", filters {intent.filters}" if intent.filters else "")
                                + (", and draft WhatsApp messages" if intent.generate_messages else ""))
        return intent
    return Intent(action="help")
