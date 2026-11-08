"""Outbound-message guardrails. Every draft (LLM, template or RM-edited) passes through here
before it can be approved, and again inside the send tool (defence in depth).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from crm_agent.config import settings

BANNED_CLAIMS = [
    r"guarantee", r"100\s*%\s*approv", r"assured approval", r"no documents?", r"no questions asked",
    r"lowest rate", r"risk[- ]free", r"instant approval", r"approved already", r"limited time only",
]
SENSITIVE = [r"medical", r"hospital", r"illness", r"surgery", r"treatment", r"debt trap", r"struggl",
             r"overdue", r"default", r"cibil", r"credit score", r"bureau",
             r"इलाज", r"अस्पताल"]
PII = [r"\b\d{9,18}\b", r"\bA\d{6}\b", r"\bbalance\b", r"account (?:no|number)", r"\bsalary of\b"]
DISCLAIMER = [r"T\s*&\s*C", r"terms and conditions", r"नियम व शर्तें"]
OPT_OUT = [r"\bSTOP\b"]

AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|l\b|k\b|cr|crore)?", re.IGNORECASE)
RATE_RE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*%")
UNIT = {"lakh": 1e5, "lac": 1e5, "l": 1e5, "k": 1e3, "cr": 1e7, "crore": 1e7}


@dataclass
class Violation:
    rule_id: str
    message: str


def _any(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _amounts(text: str) -> list[float]:
    out = []
    for num, unit in AMOUNT_RE.findall(text):
        value = float(num.replace(",", ""))
        out.append(value * UNIT.get(unit.lower(), 1) if unit else value)
    return out


def check_consent(customer: dict, channel: str = "whatsapp") -> Violation | None:
    if channel == "whatsapp" and not customer.get("whatsapp_opt_in"):
        return Violation("CONSENT_WHATSAPP", "Customer has not opted in to WhatsApp communication")
    if customer.get("dnd"):
        return Violation("DND_REGISTERED", "Customer is on the Do-Not-Disturb registry")
    return None


def check_message(body: str, ctx: dict) -> list[Violation]:
    """ctx: offer (amount/emi/rate), first_name, max_chars."""
    v: list[Violation] = []
    offer = ctx.get("offer") or {}
    if hit := _any(BANNED_CLAIMS, body):
        v.append(Violation("MISLEADING_CLAIM", f"Prohibited claim: '{hit}'"))
    if hit := _any(SENSITIVE, body):
        v.append(Violation("SENSITIVE_INFERENCE", f"References sensitive/credit information: '{hit}'"))
    if hit := _any(PII, body):
        v.append(Violation("PII_EXPOSURE", f"Possible personal/financial data in message: '{hit}'"))

    allowed = [x for x in (offer.get("amount"), offer.get("emi")) if x]
    for amt in _amounts(body):
        if not any(abs(amt - a) <= max(1000, 0.01 * a) for a in allowed):
            v.append(Violation("UNGROUNDED_AMOUNT", f"Amount ₹{amt:,.0f} is not part of the computed offer"))
    if offer.get("rate"):
        for rate in (float(r) for r in RATE_RE.findall(body)):
            if rate < offer["rate"] - 1e-6:
                v.append(Violation("RATE_UNDERSTATED", f"Quoted {rate}% is below the offer rate {offer['rate']}%"))
    if (_amounts(body) or RATE_RE.search(body)) and not _any(DISCLAIMER, body):
        v.append(Violation("MISSING_DISCLAIMER", "Pricing mentioned without a T&C disclaimer"))
    if not _any(OPT_OUT, body):
        v.append(Violation("MISSING_OPT_OUT", "Message must tell the customer how to opt out (STOP)"))
    if len(body) > ctx.get("max_chars", settings.whatsapp_max_chars):
        v.append(Violation("TOO_LONG", f"{len(body)} chars exceeds {ctx.get('max_chars', settings.whatsapp_max_chars)}"))
    first = ctx.get("first_name")
    if first and first.lower() not in body.lower():
        v.append(Violation("NOT_PERSONALISED", "Message does not address the customer by name"))
    return v


def as_dicts(violations: list[Violation]) -> list[dict]:
    return [asdict(x) for x in violations]
