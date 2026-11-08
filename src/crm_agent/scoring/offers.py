"""Product recommendation and offer sizing. Pure functions over features + product catalogue."""
from __future__ import annotations

import math

from crm_agent.config import settings

RATE_BANDS = [(780, 0.0), (750, 0.5), (720, 1.25), (0, 2.0)]  # credit score -> spread over rate_from


def emi(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 1200
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def _max_principal_for_emi(max_emi: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 1200
    return max_emi * (1 - (1 + r) ** -months) / r


def _rate_for(score: int, rate_from: float) -> float:
    spread = next(s for floor, s in RATE_BANDS if score >= floor)
    return round(rate_from + spread, 2)


def _requirement_met(product: dict, f: dict) -> bool:
    req = product["requires"]
    return {
        "salaried": f["occupation_type"] == "salaried",
        "active_pl": bool(f["has_active_pl"]),
        "fixed_deposit": f["fd_balance"] > 0,
        "none": True,
    }.get(req, False)


def size_loan(product: dict, f: dict) -> dict | None:
    """Largest amount that respects product cap, income multiple and post-loan DTI limit."""
    rate = _rate_for(f["credit_score"], product["rate_from"])
    tenure = product["max_tenure_months"]
    if product["product_type"] == "loan_against_fd":
        amount = min(0.9 * f["fd_balance"], product["max_amount"])
        constraint = "90% of FD value"
    else:
        headroom = settings.max_dti_after_loan * f["monthly_income"] - f["monthly_emi"]
        if headroom <= 0:
            return None
        caps = {
            "affordability (post-loan EMIs <= 50% of income)": _max_principal_for_emi(headroom, rate, tenure),
            "income multiple": settings.income_multiplier[f["occupation_type"]] * f["monthly_income"],
            "product maximum": product["max_amount"],
        }
        constraint, amount = min(caps.items(), key=lambda kv: kv[1])
    amount = math.floor(amount / 10000) * 10000
    if amount < 50000:
        return None
    return {"amount": amount, "rate": rate, "tenure_months": tenure,
            "emi": round(emi(amount, rate, tenure) / 100) * 100, "binding_constraint": constraint}


def recommend(f: dict, product_type: str, catalogue: list[dict]) -> dict | None:
    """Pick the best-fitting variant in catalogue order (most attractive first)."""
    for product in catalogue:
        if product["product_type"] != product_type:
            continue
        if f["monthly_income"] < product["min_income"] or f["credit_score"] < product["min_credit_score"]:
            continue
        if not _requirement_met(product, f):
            continue
        if product_type == "credit_card":
            limit = math.floor(min(2.5 * f["monthly_income"], product["max_amount"]) / 10000) * 10000
            return {"product_id": product["product_id"], "product_name": product["name"],
                    "product_type": product_type, "amount": limit, "rate": None, "tenure_months": None,
                    "emi": None, "pitch": product["pitch"],
                    "rationale": f"card limit sized at 2.5x monthly income (₹{limit:,.0f})"}
        sized = size_loan(product, f)
        if sized is None:
            continue
        return {"product_id": product["product_id"], "product_name": product["name"],
                "product_type": product_type, "pitch": product["pitch"], **sized,
                "rationale": f"{product['name']}: amount limited by {sized['binding_constraint']}; "
                             f"rate {sized['rate']}% from credit-score band"}
    return None


def alternative(f: dict, primary_type: str, catalogue: list[dict]) -> dict | None:
    """Cross-sell fallback when the requested product does not fit."""
    for alt_type in ("loan_against_fd", "credit_card", "personal_loan"):
        if alt_type == primary_type:
            continue
        rec = recommend(f, alt_type, catalogue)
        if rec:
            return rec
    return None
