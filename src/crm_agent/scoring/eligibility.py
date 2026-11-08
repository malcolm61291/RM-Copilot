"""Hard credit-policy gates. Deterministic, explainable, and owned by the bank, not the LLM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from crm_agent.formatting import inr


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    check: Callable[[dict], bool]
    detail: Callable[[dict], str]


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


COMMON = [
    Rule("KYC_VERIFIED", "KYC must be verified",
         lambda f: f["kyc_status"] == "verified", lambda f: f"KYC status: {f['kyc_status']}"),
]

POLICIES: dict[str, list[Rule]] = {
    "personal_loan": COMMON + [
        Rule("AGE_21_58", "Age between 21 and 58 at application",
             lambda f: 21 <= f["age"] <= 58, lambda f: f"age {f['age']}"),
        Rule("CREDIT_SCORE_700", "Bureau score of at least 700",
             lambda f: f["credit_score"] >= 700, lambda f: f"score {f['credit_score']}"),
        Rule("MIN_INCOME_25K", "Average monthly income of at least ₹25,000",
             lambda f: f["monthly_income"] >= 25000, lambda f: f"income ₹{inr(f['monthly_income'])}/month"),
        Rule("INCOME_STABILITY", "Income credited in at least 5 of the last 6 months",
             lambda f: f["income_months"] >= 5, lambda f: f"{min(f['income_months'], 6)}/6 months"),
        Rule("DTI_HEADROOM", "Existing EMIs at most 45% of income (room for a new EMI)",
             lambda f: f["dti"] <= 0.45, lambda f: f"current DTI {_pct(f['dti'])}"),
    ],
    "credit_card": COMMON + [
        Rule("AGE_21_65", "Age between 21 and 65",
             lambda f: 21 <= f["age"] <= 65, lambda f: f"age {f['age']}"),
        Rule("CREDIT_SCORE_720", "Bureau score of at least 720",
             lambda f: f["credit_score"] >= 720, lambda f: f"score {f['credit_score']}"),
        Rule("MIN_INCOME_30K", "Average monthly income of at least ₹30,000",
             lambda f: f["monthly_income"] >= 30000, lambda f: f"income ₹{inr(f['monthly_income'])}/month"),
        Rule("NO_EXISTING_CARD", "Does not already hold our credit card",
             lambda f: not f["has_credit_card"], lambda f: "holds card" if f["has_credit_card"] else "no card"),
    ],
    "loan_against_fd": COMMON + [
        Rule("HAS_FD", "Holds a fixed deposit with us",
             lambda f: f["fd_balance"] > 0, lambda f: f"FD ₹{inr(f['fd_balance'])}"),
    ],
}


def check_eligibility(features: dict, product_type: str) -> dict:
    rules = POLICIES.get(product_type)
    if rules is None:
        raise ValueError(f"No credit policy defined for product type '{product_type}'")
    results = [{"rule_id": r.rule_id, "description": r.description,
                "passed": bool(r.check(features)), "detail": r.detail(features)} for r in rules]
    failed = [r["rule_id"] for r in results if not r["passed"]]
    return {"eligible": not failed, "failed_rules": failed, "checks": results}
