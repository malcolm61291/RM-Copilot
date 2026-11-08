"""Tools that score, rank and recommend. Deterministic: numbers come from code, not the LLM."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from crm_agent.config import settings
from crm_agent.guardrails.rules import as_dicts, check_message
from crm_agent.scoring import eligibility as elig
from crm_agent.scoring import offers
from crm_agent.scoring.propensity import get_scorer
from crm_agent.scoring.value import meets_tier, score_value
from crm_agent.tools.crm_tools import repo
from crm_agent.tools.registry import tool


class BookArgs(BaseModel):
    features: dict[str, dict] = Field(..., description="customer_id -> features for the WHOLE book")


@tool("score_customer_value", "Percentile-based value score and tier (high/medium/standard) across the book.",
      BookArgs)
def score_customer_value(features: dict[str, dict]) -> dict[str, dict]:
    df = score_value(pd.DataFrame(list(features.values())))
    return {r["customer_id"]: r for r in df.to_dict(orient="records")}


class EligibilityArgs(BaseModel):
    features: dict[str, dict]
    product_type: str = "personal_loan"


@tool("check_eligibility", "Apply the bank's hard credit-policy rules for a product.", EligibilityArgs)
def check_eligibility(features: dict[str, dict], product_type: str = "personal_loan") -> dict[str, dict]:
    return {cid: elig.check_eligibility(f, product_type) for cid, f in features.items()}


class PropensityArgs(BaseModel):
    features: dict[str, dict]
    product_type: str = "personal_loan"


@tool("score_propensity", "Estimate probability of conversion with reason codes (ML or heuristic per product).",
      PropensityArgs)
def score_propensity(features: dict[str, dict], product_type: str = "personal_loan") -> dict[str, dict]:
    return get_scorer(product_type).score(list(features.values()))


class RankArgs(BaseModel):
    pool: list[str] = Field(..., description="customer_ids that passed the RM's filters")
    value: dict[str, dict]
    eligibility: dict[str, dict]
    propensity: dict[str, dict]
    top_n: int = settings.default_top_n
    min_value_tier: str = "high"
    min_propensity: float = settings.propensity_threshold


@tool("rank_candidates", "Combine value, eligibility and propensity into a ranked shortlist with a funnel.",
      RankArgs)
def rank_candidates(pool, value, eligibility, propensity, top_n, min_value_tier, min_propensity) -> dict:
    # Each stage narrows the previous one; the funnel counts are shown to the RM as-is.
    valuable = [c for c in pool if meets_tier(value[c]["value_tier"], min_value_tier)]
    eligible = [c for c in valuable if eligibility[c]["eligible"]]
    likely = [c for c in eligible if propensity[c]["probability"] >= min_propensity]

    priority = {c: (settings.priority_weight_propensity * propensity[c]["probability"]
                    + settings.priority_weight_value * value[c]["value_score"]) for c in likely}
    selected = sorted(likely, key=priority.get, reverse=True)[:top_n]

    funnel = {"in_scope": len(pool), "value_ok": len(valuable), "eligible": len(eligible),
              "likely_to_convert": len(likely), "selected": len(selected)}
    shortlist = [{"rank": i, "customer_id": c, "priority_score": round(priority[c], 3)}
                 for i, c in enumerate(selected, start=1)]
    return {"funnel": funnel, "shortlist": shortlist}


class RecommendArgs(BaseModel):
    features: dict
    product_type: str = "personal_loan"


@tool("recommend_product", "Choose the best product variant and size the offer (amount, rate, EMI).",
      RecommendArgs)
def recommend_product(features: dict, product_type: str = "personal_loan") -> dict:
    catalogue = repo().products()
    primary = offers.recommend(features, product_type, catalogue)
    if primary:
        return {"offer": primary, "alternative": None}
    return {"offer": None, "alternative": offers.alternative(features, product_type, catalogue)}


class ValidateArgs(BaseModel):
    body: str
    offer: dict
    first_name: str


@tool("validate_message", "Run compliance guardrails on an outbound message.", ValidateArgs)
def validate_message(body: str, offer: dict, first_name: str) -> dict:
    v = check_message(body, {"offer": offer, "first_name": first_name,
                             "max_chars": settings.whatsapp_max_chars})
    return {"passed": not v, "violations": as_dicts(v)}
