"""Conversion-propensity scorers behind one interface, so each product can use the
right technique: a trained ML model where we have labelled campaign history, and a
transparent heuristic scorecard where we do not (cold start).

Every scorer returns a probability *and* reason codes, because an RM (and a regulator)
needs to know why a customer was picked.

Train:  python -m crm_agent.scoring.propensity
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crm_agent.config import settings
from crm_agent.formatting import inr
from crm_agent.scoring.features import enrich

ML_FEATURES = [
    "loan_intent_events_30d", "big_ticket_signal", "credit_score", "dti", "income_stability",
    "income_growth", "log_income", "cc_utilisation", "is_salaried", "segment_rank",
    "has_active_pl", "tenure_years", "age",
]

REASON_TEXT = {
    "loan_intent_events_30d": "browsed loan pages / EMI calculator {loan_intent_events_30d:.0f}x in the last 30 days",
    "big_ticket_signal": "big-ticket spending ({recent_big_ticket_category}) {big_ticket_level} their usual level in the last 45 days",
    "credit_score": "strong credit score ({credit_score:.0f})",
    "dti": "low existing EMI burden ({dti_pct:.0f}% of income)",
    "income_stability": "regular income ({income_months_capped:.0f}/6 months)",
    "income_growth": "income up {income_growth_pct:.0f}% vs the previous quarter",
    "log_income": "income of ₹{income_inr}/month",
    "cc_utilisation": "credit-card utilisation at {cc_util_pct:.0f}%",
    "is_salaried": "salaried profile",
    "segment_rank": "{segment} relationship",
    "has_active_pl": "already has an active personal loan",
    "tenure_years": "{tenure_years:.1f} years with the bank",
    "age": "age {age:.0f}",
}


NEGATIVE_TEXT = {
    "credit_score": "credit score on the lower side ({credit_score:.0f})",
    "dti": "existing EMIs already take {dti_pct:.0f}% of income",
    "income_stability": "irregular income ({income_months_capped:.0f}/6 months credited)",
    "income_growth": "income down {income_drop_pct:.0f}% vs the previous quarter",
    "cc_utilisation": "high credit-card utilisation ({cc_util_pct:.0f}%)",
    "has_active_pl": "already has an active personal loan",
}
WATCH_OUT_FEATURES = {"dti", "cc_utilisation", "has_active_pl", "income_stability", "income_growth", "credit_score"}
DRIVER_EXCLUDE = {"age", "has_active_pl"}


def _fmt(feature: str, f: dict, negative: bool = False) -> str:
    ratio = f.get("big_ticket_ratio") or 0
    ctx = {
        **f,
        "income_inr": inr(f["monthly_income"]),
        "big_ticket_level": "more than 10x" if ratio > 10 else f"{ratio:.1f}x",
        "recent_big_ticket_category": (f.get("recent_big_ticket_category") or "mixed").replace("_", " "),
        "dti_pct": f["dti"] * 100,
        "cc_util_pct": f["cc_utilisation"] * 100,
        "income_growth_pct": f["income_growth"] * 100,
        "income_drop_pct": abs(min(f["income_growth"], 0)) * 100,
        "income_months_capped": min(f["income_months"], 6),
    }
    template = NEGATIVE_TEXT.get(feature, REASON_TEXT.get(feature)) if negative else REASON_TEXT.get(feature)
    try:
        return template.format(**ctx)
    except (KeyError, ValueError, AttributeError):
        return feature


class PropensityScorer(ABC):
    method: str

    @abstractmethod
    def score(self, features: list[dict]) -> dict[str, dict]:
        """customer_id -> {probability, drivers, watch_outs, method}"""


class MLPropensityScorer(PropensityScorer):
    """Logistic regression on historical campaign outcomes. Chosen over a GBM for this
    use case because per-customer contributions (coef x standardised value) are exact,
    which gives faithful reason codes without a separate explainer."""

    method = "ml_logistic_regression"

    def __init__(self, product_type: str = "personal_loan"):
        path = settings.model_dir / f"propensity_{product_type}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not trained: {path}. Run `make train`.")
        bundle = joblib.load(path)
        self.pipeline: Pipeline = bundle["pipeline"]
        self.features: list[str] = bundle["features"]
        self.metrics: dict = bundle["metrics"]

    def score(self, features: list[dict]) -> dict[str, dict]:
        if not features:
            return {}
        X = pd.DataFrame(features)[self.features].astype(float)
        probs = self.pipeline.predict_proba(X)[:, 1]
        scaler: StandardScaler = self.pipeline.named_steps["scale"]
        coefs = self.pipeline.named_steps["model"].coef_[0]
        contributions = scaler.transform(X) * coefs
        out = {}
        for f, p, contrib in zip(features, probs, contributions):
            order = np.argsort(contrib)
            drivers = [_fmt(self.features[i], f) for i in order[::-1]
                       if contrib[i] > 0.15 and self.features[i] not in DRIVER_EXCLUDE][:3]
            watch = [_fmt(self.features[i], f, negative=True) for i in order
                     if contrib[i] < -0.25 and self.features[i] in WATCH_OUT_FEATURES][:2]
            out[f["customer_id"]] = {"probability": round(float(p), 3), "drivers": drivers,
                                     "watch_outs": watch, "method": self.method}
        return out


class HeuristicScorer(PropensityScorer):
    """Transparent points-based scorecard for products without enough labelled history."""

    method = "heuristic_scorecard"

    def __init__(self, weights: dict[str, float], bias: float):
        self.weights, self.bias = weights, bias

    def score(self, features: list[dict]) -> dict[str, dict]:
        out = {}
        for f in features:
            signals = {
                "loan_intent_events_30d": min(f["loan_intent_events_30d"], 4) / 4,
                "big_ticket_signal": min(f["big_ticket_signal"], 2.5) / 2.5,
                "credit_score": (f["credit_score"] - 650) / 200,
                "log_income": (f["log_income"] - 10) / 2.5,
                "segment_rank": f["segment_rank"] / 2,
                "tenure_years": min(f["tenure_years"], 10) / 10,
                "fd_balance": 1.0 if f.get("fd_balance", 0) > 0 else 0.0,
                "cc_utilisation": f["cc_utilisation"],
            }
            contrib = {k: self.weights.get(k, 0) * v for k, v in signals.items()}
            z = self.bias + sum(contrib.values())
            p = 1 / (1 + math.exp(-z))
            ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
            drivers = [_fmt(k, f) if k in REASON_TEXT else "holds a fixed deposit" for k, v in ranked[:3] if v > 0.2]
            out[f["customer_id"]] = {"probability": round(p, 3), "drivers": drivers,
                                     "watch_outs": [], "method": self.method}
        return out


def get_scorer(product_type: str) -> PropensityScorer:
    """Scorer registry: add a product by registering a scorer here."""
    if product_type == "personal_loan":
        return MLPropensityScorer("personal_loan")
    if product_type == "credit_card":
        return HeuristicScorer({"credit_score": 1.2, "log_income": 1.0, "segment_rank": 0.8,
                                "big_ticket_signal": 1.0, "tenure_years": 0.5}, bias=-2.7)
    if product_type == "loan_against_fd":
        return HeuristicScorer({"fd_balance": 1.5, "big_ticket_signal": 1.5, "loan_intent_events_30d": 1.2,
                                "cc_utilisation": 0.8}, bias=-2.6)
    raise ValueError(f"No propensity scorer registered for '{product_type}'")


def train(product_type: str = "personal_loan", seed: int = 7) -> dict:
    from crm_agent.data.repository import Repository

    df = enrich(Repository().training_frame(product_type))
    X, y = df[ML_FEATURES].astype(float), df["converted"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=1000))])
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    top_k = max(1, int(0.2 * len(p)))
    metrics = {
        "rows": int(len(df)), "base_rate": round(float(y.mean()), 3),
        "auc": round(float(roc_auc_score(y_te, p)), 3),
        "avg_precision": round(float(average_precision_score(y_te, p)), 3),
        "precision_at_top20pct": round(float(y_te.to_numpy()[np.argsort(-p)[:top_k]].mean()), 3),
    }
    pipe.fit(X, y)  # refit on all labelled data after evaluation
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    path = settings.model_dir / f"propensity_{product_type}.joblib"
    joblib.dump({"pipeline": pipe, "features": ML_FEATURES, "metrics": metrics}, path)
    coefs = dict(zip(ML_FEATURES, np.round(pipe.named_steps["model"].coef_[0], 3).tolist()))
    (settings.model_dir / f"propensity_{product_type}_card.json").write_text(
        json.dumps({"metrics": metrics, "coefficients": coefs}, indent=2))
    return {"path": str(path), "metrics": metrics, "coefficients": coefs}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
