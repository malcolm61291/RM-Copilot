"""Outreach composer: writes a personalised WhatsApp message for one customer.

Design rules:
- Data minimisation: the composer (and the LLM) only sees an allow-listed fact sheet, never raw
  transactions, balances or bureau scores.
- Personalisation by *theme*, not surveillance: a recent travel spend becomes "plans for the
  season", never "we saw your ₹1.2L flight booking". Medical spend is never used.
- All numbers come from the computed offer; the guardrails reject anything else.
"""
from __future__ import annotations

import json

from crm_agent.config import settings
from crm_agent.formatting import inr, plain_dashes
from crm_agent.llm.provider import LLMError, get_llm

THEMES = {
    "travel": {"en": "planning something special this season",
               "hi": "इस सीज़न कुछ ख़ास प्लान कर रहे हैं"},
    "education": {"en": "investing in learning or upskilling",
                  "hi": "पढ़ाई या नई स्किल में निवेश की सोच रहे हैं"},
    "home_improvement": {"en": "thinking about giving your home an upgrade",
                         "hi": "घर को नया रूप देने की सोच रहे हैं"},
    "wedding": {"en": "preparing for a big celebration in the family",
                "hi": "परिवार में किसी बड़े जश्न की तैयारी कर रहे हैं"},
}
SENSITIVE_THEMES = {"medical"}
SEGMENT_LABEL = {"priority": "Priority Banking", "affluent": "Preferred", "mass": "valued"}


def fact_sheet(f: dict, offer: dict, language: str) -> dict:
    """The ONLY customer information that reaches the message writer."""
    theme = None
    if f.get("recent_big_ticket_category") and f.get("big_ticket_ratio", 0) >= 1.5:
        if f["recent_big_ticket_category"] not in SENSITIVE_THEMES:
            theme = f["recent_big_ticket_category"]
    return {
        "first_name": f["first_name"],
        "language": language,
        "occupation_type": f["occupation_type"],
        "segment": f["segment"],
        "years_with_bank": int(f["tenure_years"]),
        "life_theme": theme,
        "offer": {k: offer.get(k) for k in ("product_name", "product_type", "amount", "rate",
                                             "tenure_months", "emi", "pitch")},
        "rm_name": settings.rm_name,
        "bank_name": settings.bank_name,
    }


SYSTEM = """You write WhatsApp messages from a bank Relationship Manager to an existing customer.
Hard rules (a compliance checker will reject violations):
- Address the customer by first name. Write in the requested language (en = English, hi = Hindi in
  Devanagari; product names may stay in English). Under 600 characters. Plain text, at most one emoji.
- Use ONLY numbers present in the fact sheet's offer, formatted in Indian style with ₹ (e.g. ₹4,50,000).
  Never state a rate lower than the offer rate. Say rates 'start from' the offer rate.
- No promises of approval ('guaranteed', 'instant approval', 'no documents'), no urgency pressure.
- Never mention balances, salary, transactions, credit/CIBIL score, health or debts.
- If life_theme is set you may gently reference it as a possibility, never as something you observed.
- End with: '*T&C apply. Subject to credit assessment. Reply STOP to opt out.' (translated for Hindi,
  keeping the word STOP).
- Tone: {tone}. Sign off with the RM's first name and bank.
- Do not use em dashes or en dashes; use commas or full stops instead.
Return only the message text."""


def compose(f: dict, offer: dict, language: str, tone: str = "warm",
            instruction: str | None = None, feedback: list[dict] | None = None) -> tuple[str, str]:
    """Returns (body, method). Uses the LLM if configured, else the template composer."""
    facts = fact_sheet(f, offer, language)
    llm = get_llm()
    if llm is not None:
        try:
            user = {"fact_sheet": facts}
            if instruction:
                user["rm_instruction"] = instruction
            if feedback:
                user["fix_these_compliance_issues"] = feedback
            body = llm.complete(SYSTEM.format(tone=tone), json.dumps(user, ensure_ascii=False),
                                max_tokens=600, temperature=0.6)
            body = plain_dashes(body).strip()
            if body:
                return body, f"llm:{llm.name}"
        except LLMError:
            pass
    return template_compose(facts, tone, instruction), "template"


def _apply_instruction(tone: str, language: str, instruction: str | None) -> tuple[str, str]:
    if not instruction:
        return tone, language
    i = instruction.lower()
    if any(w in i for w in ("short", "concise", "brief", "crisp")):
        tone = "concise"
    elif "formal" in i:
        tone = "formal"
    elif any(w in i for w in ("warm", "friendly")):
        tone = "warm"
    if "hindi" in i:
        language = "hi"
    elif "english" in i:
        language = "en"
    return tone, language


def template_compose(facts: dict, tone: str = "warm", instruction: str | None = None) -> str:
    tone, lang = _apply_instruction(tone, facts["language"], instruction)
    o, name, rm, bank = facts["offer"], facts["first_name"], facts["rm_name"], facts["bank_name"]
    theme = THEMES.get(facts["life_theme"] or "", {}).get(lang)
    is_card = o["product_type"] == "credit_card"
    amount = f"₹{inr(o['amount'])}"

    if lang == "hi":
        greet = {"formal": f"आदरणीय {name} जी,", "concise": f"{name} जी,"}.get(tone, f"नमस्ते {name} जी,")
        intro = f"मैं {rm}, {bank} में आपका रिलेशनशिप मैनेजर।"
        hook = (f"अगर आप {theme}, तो यह सही समय हो सकता है।" if theme
                else f"{bank} के साथ आपके {facts['years_with_bank']} साल के रिश्ते के लिए धन्यवाद।"
                if facts["years_with_bank"] >= 2 else "आपकी अगली बड़ी योजना में हम आपके साथ हैं।")
        if is_card:
            offer = f"आप हमारे {o['product_name']} के लिए आवेदन कर सकते हैं, लिमिट {amount} तक।"
        else:
            offer = (f"आपके लिए {amount} तक का {o['product_name']} ऑफ़र है। ब्याज दर {o['rate']}% प्रति वर्ष* से "
                     f"शुरू, EMI ₹{inr(o['emi'])}/माह से ({o['tenure_months']} महीने)।")
        cta = ("कृपया बताएँ कि क्या आप विस्तृत जानकारी या अपनी सुविधा के समय एक कॉल चाहेंगे।" if tone == "formal"
               else "क्या मैं पूरी जानकारी भेजूँ या आपकी सुविधा के समय कॉल करूँ?")
        legal = "*नियम व शर्तें लागू। क्रेडिट मूल्यांकन के अधीन। ऑप्ट-आउट के लिए STOP लिखें।"
        sign = f"सादर,\n{rm}, {bank}"
        lines = [f"{greet} {intro}", hook, offer, cta, sign, legal] if tone != "concise" else \
                [f"{greet} {offer}", cta, sign, legal]
        return "\n\n".join(lines)

    greet = {"formal": f"Dear {name},", "concise": f"Hi {name},"}.get(tone, f"Hi {name} 👋")
    intro = f"{rm} here, your Relationship Manager at {bank}."
    if theme:
        hook = f"If you're {theme}, this could be good timing."
    elif facts["years_with_bank"] >= 2:
        hook = f"Thank you for {facts['years_with_bank']} years with {bank}."
    else:
        hook = "We'd love to support your next big plan."
    seg = SEGMENT_LABEL.get(facts["segment"], "valued")
    if is_card:
        offer = (f"As a {seg} customer, you can apply for our {o['product_name']} with a limit of up to "
                 f"{amount}. {o['pitch']}")
    else:
        offer = (f"As a {seg} customer, you have a {o['product_name']} offer of up to {amount}, with rates "
                 f"starting from {o['rate']}% p.a.* and EMIs from ₹{inr(o['emi'])}/month over "
                 f"{o['tenure_months']} months.")
    benefit = o["pitch"] if not is_card else ""
    cta = ("Shall I share the details, or call you at a time that suits you?" if tone != "formal"
           else "Please let me know if you would like the details or a call at your convenience.")
    sign = f"Regards,\n{rm}, {bank}"
    legal = "*T&C apply. Subject to credit assessment. Reply STOP to opt out."
    if tone == "concise":
        offer_short = (f"You're eligible to apply for our {o['product_name']} (limit up to {amount})." if is_card
                       else f"You have a {o['product_name']} offer of up to {amount}, rates from {o['rate']}% p.a.*")
        return "\n\n".join([f"{greet} {offer_short}", cta, sign, legal])
    return "\n\n".join(x for x in [f"{greet} {intro}", hook, offer, benefit, cta, sign, legal] if x)
