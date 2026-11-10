import json
import os
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_DATA = ROOT / "data" / "test"
os.environ["CRM_DB_PATH"] = str(TEST_DATA / "crm.db")
os.environ["CRM_MODEL_DIR"] = str(TEST_DATA / "models")
os.environ["CRM_RUNS_DIR"] = str(TEST_DATA / "runs")
os.environ["LLM_PROVIDER"] = "offline"


@pytest.fixture(scope="session", autouse=True)
def dataset():
    from crm_agent.config import settings
    from crm_agent.data.generate import generate
    from crm_agent.scoring.propensity import train

    generate(settings.db_path, n_customers=1200, seed=42, as_of=date(2026, 11, 8))
    train()
    yield


class FakeLLM:
    """Deterministic stand-in for a real provider. The first draft for each customer contains a
    compliance violation, so tests exercise the guardrail -> revise loop end to end."""

    name = "fake:test"

    def __init__(self):
        self.calls = []

    def complete(self, system, user, *, json_mode=False, max_tokens=700, temperature=0.3):
        self.calls.append(system[:40])
        if json_mode:
            msg = json.loads(user)["message"].lower()
            if "credit card" in msg:
                return json.dumps({"action": "find_leads", "product_type": "credit_card", "top_n": 3,
                                   "generate_messages": True, "understood_as": "credit card leads"})
            return json.dumps({"action": "find_leads", "product_type": "personal_loan", "top_n": 4,
                               "min_value_tier": "high", "generate_messages": True,
                               "understood_as": "PL leads with WhatsApp drafts"})
        payload = json.loads(user)
        if "fact_sheet" not in payload:
            return "- explanation from fake LLM"
        f, o = payload["fact_sheet"], payload["fact_sheet"]["offer"]
        from crm_agent.formatting import inr

        body = (f"Hi {f['first_name']}, {o['product_name']} of up to ₹{inr(o['amount'])} from {o['rate']}% p.a.* "
                f"Regards, {f['rm_name']}, {f['bank_name']}. *T&C apply. Reply STOP to opt out.")
        if "fix_these_compliance_issues" not in payload:
            body = "Guaranteed approval! " + body
        return body


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    import crm_agent.agents.composer as composer
    import crm_agent.agents.explainer as explainer
    import crm_agent.agents.intent as intent

    for mod in (composer, explainer, intent):
        monkeypatch.setattr(mod, "get_llm", lambda: llm)
    return llm
