import pytest

from crm_agent.agents.composer import fact_sheet, template_compose
from crm_agent.guardrails.rules import check_message
from crm_agent.scoring.offers import emi, recommend
from crm_agent.tools import ToolError, call_tool, catalogue


@pytest.fixture(scope="module")
def book():
    return call_tool("get_customer_features")


def test_features_cover_whole_book(book):
    assert len(book) == 1200
    sample = next(iter(book.values()))
    for key in ("monthly_income", "dti", "big_ticket_ratio", "loan_intent_events_30d", "relationship_value"):
        assert key in sample


def test_value_tiers_are_relative_to_book(book):
    value = call_tool("score_customer_value", features=book)
    share_high = sum(v["value_tier"] == "high" for v in value.values()) / len(value)
    assert 0.25 <= share_high <= 0.40


def test_eligibility_is_explainable(book):
    elig = call_tool("check_eligibility", features=book, product_type="personal_loan")
    failed = next(e for e in elig.values() if not e["eligible"])
    assert failed["failed_rules"] and all("detail" in c for c in failed["checks"])


def test_propensity_has_reason_codes(book):
    prop = call_tool("score_propensity", features=book, product_type="personal_loan")
    top = max(prop.values(), key=lambda p: p["probability"])
    assert 0 <= top["probability"] <= 1 and top["drivers"]


def test_offer_respects_affordability(book):
    from crm_agent.tools.crm_tools import repo

    catalogue_ = repo().products()
    for f in list(book.values())[:300]:
        o = recommend(f, "personal_loan", catalogue_)
        if o:
            total_emi = f["monthly_emi"] + emi(o["amount"], o["rate"], o["tenure_months"])
            assert total_emi <= 0.5 * f["monthly_income"] + 200


def test_every_template_message_is_compliant(book):
    from crm_agent.tools.crm_tools import repo

    catalogue_ = repo().products()
    checked = 0
    for f in list(book.values())[:200]:
        for product in ("personal_loan", "credit_card"):
            o = recommend(f, product, catalogue_)
            if not o:
                continue
            for lang in ("en", "hi"):
                for tone in ("warm", "formal", "concise"):
                    body = template_compose(fact_sheet(f, o, lang), tone)
                    assert check_message(body, {"offer": o, "first_name": f["first_name"]}) == [], body
                    checked += 1
    assert checked > 100


def test_sensitive_theme_never_reaches_composer(book):
    medical = next(f for f in book.values() if f["recent_big_ticket_category"] == "medical"
                   and f["big_ticket_ratio"] >= 1.5)
    assert fact_sheet(medical, {"amount": 1}, "en")["life_theme"] is None


def test_tool_input_validation():
    with pytest.raises(ToolError):
        call_tool("check_eligibility", product_type="personal_loan")  # missing features
    with pytest.raises(ToolError):
        call_tool("does_not_exist")


def test_send_tool_blocks_non_consented(book):
    cust = next(f for f in book.values() if not f["whatsapp_opt_in"])
    with pytest.raises(ToolError, match="Blocked at send"):
        call_tool("send_whatsapp_message", customer=cust, body="Hi. Reply STOP", offer={},
                  approved_by="RM001", thread_id="t")


def test_catalogue_exports_schemas():
    names = {t["name"] for t in catalogue()}
    assert {"find_customers", "score_propensity", "send_whatsapp_message"} <= names
    assert next(t for t in catalogue() if t["name"] == "send_whatsapp_message")["side_effect"]
