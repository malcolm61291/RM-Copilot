"""Tools that score, rank and recommend. Deterministic: numbers come from code, not the LLM."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from crm_agent.config import settings
from crm_agent.guardrails.rules import as_dicts, check_message
from crm_agent.scoring import eligibility as elig
from crm_agent.scoring import offers
from crm_agent.scoring.propensity import get_scorer
from crm_agent.scoring.value import score_value
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
    tiers = {"standard": 0, "medium": 1, "high": 2}
    floor = tiers.get(min_value_tier, 2)
    funnel = {"in_scope": len(pool)}
    stage = [c for c in pool if tiers[value[c]["value_tier"]] >= floor]
    funnel["value_ok"] = len(stage)
    stage = [c for c in stage if eligibility[c]["eligible"]]
    funnel["eligible"] = len(stage)
    stage = [c for c in stage if propensity[c]["probability"] >= min_propensity]
    funnel["likely_to_convert"] = len(stage)
    ranked = sorted(stage, key=lambda c: (settings.priority_weight_propensity * propensity[c]["probability"]
                                          + settings.priority_weight_value * value[c]["value_score"]),
                    reverse=True)
    selected = ranked[:top_n]
    funnel["selected"] = len(selected)
    return {
        "funnel": funnel,
        "shortlist": [{
            "rank": i + 1, "customer_id": c,
            "priority_score": round(settings.priority_weight_propensity * propensity[c]["probability"]
                                    + settings.priority_weight_value * value[c]["value_score"], 3),
        } for i, c in enumerate(selected)],
    }


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
