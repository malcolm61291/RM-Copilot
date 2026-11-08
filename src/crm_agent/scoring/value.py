"""Customer value scoring: who is 'high-value' relative to the RM's book."""
from __future__ import annotations

import pandas as pd

from crm_agent.config import settings

WEIGHTS = {"relationship_value": 0.45, "monthly_income": 0.35, "tenure_months": 0.20}
SEGMENT_BONUS = {"priority": 0.15, "affluent": 0.05, "mass": 0.0}


def score_value(book: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank value score over the whole book (not just the filtered subset),
    so 'high-value' means the same thing whatever filter the RM applies."""
    scores = sum(book[col].rank(pct=True) * w for col, w in WEIGHTS.items())
    scores = (scores + book["segment"].map(SEGMENT_BONUS).fillna(0)).clip(upper=1.0)
    cutoff = scores.quantile(settings.high_value_quantile)
    tier = pd.Series("standard", index=book.index)
    tier[scores >= scores.quantile(0.40)] = "medium"
    tier[(scores >= cutoff) | (book["segment"] == "priority")] = "high"
    return pd.DataFrame({
        "customer_id": book["customer_id"],
        "value_score": scores.round(3),
        "value_tier": tier,
        "value_cutoff": round(float(cutoff), 3),
    })
