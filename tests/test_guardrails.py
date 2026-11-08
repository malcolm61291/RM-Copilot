from crm_agent.guardrails.rules import check_consent, check_message

OFFER = {"amount": 450000, "emi": 9800, "rate": 11.25}
CTX = {"offer": OFFER, "first_name": "Priya"}
GOOD = ("Hi Priya, you have a Personal Loan offer of up to ₹4,50,000 at rates from 11.25% p.a.* "
        "with EMIs from ₹9,800/month. *T&C apply. Reply STOP to opt out.")


def ids(body, ctx=CTX):
    return {v.rule_id for v in check_message(body, ctx)}


def test_compliant_message_passes():
    assert ids(GOOD) == set()


def test_misleading_claims_blocked():
    assert "MISLEADING_CLAIM" in ids(GOOD + " Guaranteed approval, no documents needed!")


def test_hallucinated_amount_blocked():
    assert "UNGROUNDED_AMOUNT" in ids(GOOD.replace("₹4,50,000", "₹10,00,000"))


def test_lakh_notation_is_parsed():
    assert "UNGROUNDED_AMOUNT" not in ids(GOOD.replace("₹4,50,000", "₹4.5 lakh"))


def test_understated_rate_blocked():
    assert "RATE_UNDERSTATED" in ids(GOOD.replace("11.25%", "9.99%"))


def test_disclaimer_and_opt_out_required():
    found = ids(GOOD.replace("*T&C apply. ", "").replace("Reply STOP to opt out.", ""))
    assert {"MISSING_DISCLAIMER", "MISSING_OPT_OUT"} <= found


def test_sensitive_and_pii_blocked():
    assert "SENSITIVE_INFERENCE" in ids(GOOD + " We noticed your hospital bills.")
    assert "PII_EXPOSURE" in ids(GOOD + " Your balance is low.")
    assert "SENSITIVE_INFERENCE" in ids(GOOD + " Your CIBIL score is great.")


def test_must_address_customer():
    assert "NOT_PERSONALISED" in ids(GOOD.replace("Priya", "there"))


def test_consent_rules():
    assert check_consent({"whatsapp_opt_in": 1, "dnd": 0}) is None
    assert check_consent({"whatsapp_opt_in": 0, "dnd": 0}).rule_id == "CONSENT_WHATSAPP"
    assert check_consent({"whatsapp_opt_in": 1, "dnd": 1}).rule_id == "DND_REGISTERED"
