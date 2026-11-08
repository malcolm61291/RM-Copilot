"""Derived features shared by value scoring, eligibility and propensity models."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_RANK = {"mass": 0, "affluent": 1, "priority": 2}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios and normalised signals to the raw feature frame from the repository."""
    out = df.copy()
    income = out["monthly_income"].clip(lower=1)
    out["relationship_value"] = out["savings_balance"] + out["fd_balance"]
    out["dti"] = (out["monthly_emi"] / income).round(4)
    out["income_stability"] = (out["income_months"].clip(upper=6) / 6).round(3)
    prev = out["monthly_income_prev"].replace(0, np.nan)
    out["income_growth"] = ((out["monthly_income"] - prev) / prev).fillna(0).clip(-1, 1).round(4)
    out["big_ticket_ratio"] = (out["big_ticket_45d"] / (out["big_ticket_baseline_45d"] + 0.05 * income)).round(3)
    out["big_ticket_signal"] = np.log1p(out["big_ticket_ratio"]).round(4)
    out["is_salaried"] = (out["occupation_type"] == "salaried").astype(int)
    out["segment_rank"] = out["segment"].map(SEGMENT_RANK).fillna(0).astype(int)
    out["tenure_years"] = (out["tenure_months"] / 12).round(2)
    out["log_income"] = np.log(income).round(4)
    out["first_name"] = out["name"].str.split().str[0]
    return out


def to_records(df: pd.DataFrame) -> dict[str, dict]:
    """customer_id -> plain-JSON feature dict (safe to keep in graph state / checkpoints)."""
    clean = df.replace({np.nan: None})
    return {row["customer_id"]: {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in clean.to_dict(orient="records")}
