"""End-to-end conversation tests through the LangGraph agent."""
from crm_agent.agents.intent import parse_rules
from crm_agent.agents.planner import make_plan
from crm_agent.orchestration.session import AgentSession
from crm_agent.tools.crm_tools import repo

ASK = ("Find high-value customers likely to convert for a personal loan this month and generate "
       "personalized WhatsApp messages.")


def steps(plan):
    return [p["step"] for p in plan]


def test_intent_parsing_rules():
    i = parse_rules(ASK, None)
    assert i.action == "find_leads" and i.product_type == "personal_loan"
    assert i.generate_messages and i.min_value_tier == "high"
    r = parse_rules("only salaried customers in Mumbai, top 5, in Hindi", i.model_dump())
    assert r.action == "refine" and r.filters == {"city": ["Mumbai"], "occupation_type": "salaried"}
    assert r.top_n == 5 and r.language == "hi" and r.product_type == "personal_loan"
    assert parse_rules("approve 1, 3", None).draft_numbers == [1, 3]
    assert parse_rules("edit 2: more formal", None).instruction == "more formal"
    assert parse_rules("why not C0042?", None).target == "C0042"


def test_planner_replans_incrementally():
    first = parse_rules(ASK, None).model_dump()
    full = make_plan(first, None, set())
    assert steps(full)[0] == "retrieve_customers" and steps(full)[-1] == "present"
    everything = {"pool", "features", "value", "eligibility", "propensity", "shortlist", "offers", "drafts"}
    shorter = parse_rules("make them shorter", first).model_dump()
    assert steps(make_plan(shorter, first, everything)) == ["compose_messages", "compliance_check", "present"]
    mumbai = parse_rules("only Mumbai", first).model_dump()
    assert "compute_features" not in steps(make_plan(mumbai, first, everything))
    assert "retrieve_customers" in steps(make_plan(mumbai, first, everything))


def test_core_use_case_end_to_end():
    s = AgentSession("t-core")
    st = s.send(ASK)
    assert st["funnel"]["selected"] == len(st["shortlist"]) > 0
    assert st["funnel"]["in_scope"] >= st["funnel"]["value_ok"] >= st["funnel"]["eligible"]
    assert s.awaiting_approval                      # human approval required before sending
    ready = [d for d in st["drafts"] if d["status"] == "ready"]
    assert ready and all(d["first_name"] in d["body"] for d in ready)
    assert len({d["body"] for d in ready}) == len(ready)   # personalised, not one template blast
    excluded_ids = {x["customer_id"] for x in st["excluded"]}
    assert not excluded_ids & {d["customer_id"] for d in st["drafts"]}
    assert repo().outbox("t-core") == []            # nothing sent without approval


def test_refine_explain_edit_approve():
    s = AgentSession("t-flow")
    s.send(ASK)
    st = s.send("only salaried customers in Mumbai, top 5, in Hindi")
    assert "compute_features" not in st["executed"]           # reused from the first turn
    assert all(d["language"] == "hi" for d in st["drafts"])
    assert all(st["features"][c["customer_id"]]["city"] == "Mumbai" for c in st["shortlist"])

    st = s.send("why is #1 on the list?")
    assert "Shortlisted at rank #1" in st["response"]

    st = s.send("edit 2 text: Hi, approval is guaranteed! Reply STOP to opt out")
    d2 = next(d for d in st["drafts"] if d["no"] == 2)
    assert d2["status"] == "blocked" and d2["method"] == "rm_edit"

    st = s.send("approve all")
    statuses = {d["no"]: d["status"] for d in st["drafts"]}
    assert statuses[2] == "blocked" and list(statuses.values()).count("sent") >= 1
    assert len(repo().outbox("t-flow")) == list(statuses.values()).count("sent")


def test_other_product_reuses_features():
    s = AgentSession("t-cc")
    s.send(ASK)
    st = s.send("Which customers should I pitch a credit card to? top 3, draft messages")
    assert st["intent"]["product_type"] == "credit_card"
    assert "compute_features" not in st["executed"] and "check_eligibility" in st["executed"]
    assert all(st["offers"][c["customer_id"]]["product_type"] == "credit_card"
               for c in st["shortlist"] if c["customer_id"] in st["offers"])


def test_llm_path_with_self_correction(fake_llm):
    s = AgentSession("t-llm")
    st = s.send(ASK)
    assert st["turn_intent"]["parser"].startswith("llm:")
    ready = [d for d in st["drafts"] if d["status"] == "ready"]
    assert ready and all(d["method"].startswith("llm:") for d in ready)
    assert all(d["attempts"] == 2 for d in ready)            # first draft failed, revised draft passed
    assert all("guarantee" not in d["body"].lower() for d in ready)
    assert st["executed"].count("compliance_check") == 2
