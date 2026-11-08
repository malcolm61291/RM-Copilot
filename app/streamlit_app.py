"""Streamlit chat UI for the RM Copilot.   streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from crm_agent.config import settings  # noqa: E402
from crm_agent.orchestration.nodes import summarise_offer  # noqa: E402
from crm_agent.orchestration.session import AgentSession  # noqa: E402

st.set_page_config(page_title="RM Copilot", page_icon="💬", layout="wide")

if "session" not in st.session_state:
    st.session_state.session = AgentSession()
    st.session_state.chat = []
session: AgentSession = st.session_state.session

with st.sidebar:
    st.header("RM Copilot")
    st.caption(f"{settings.bank_name} · RM {settings.rm_name} · LLM: {settings.llm_provider}")
    st.markdown("**Try**\n- Find high-value customers likely to convert for a personal loan this month "
                "and generate personalized WhatsApp messages\n- Only salaried customers in Mumbai, top 5, in Hindi\n"
                "- Why is #1 on the list?\n- edit 2: make it more formal\n- approve 1,3\n"
                "- Which customers should I pitch a credit card to?")
    if st.button("New conversation"):
        st.session_state.session = AgentSession()
        st.session_state.chat = []
        st.rerun()
    if session.awaiting_approval:
        st.warning("Drafts are waiting for your approval.")

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask about your customers…")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Working…"):
        state = session.send(prompt)
    st.session_state.chat.append(("assistant", state.get("response", "")))
    with st.chat_message("assistant"):
        st.markdown(state.get("response", "").replace("\n", "  \n"))

state = session.state()
turn = state.get("turn_intent") or {}
if state.get("shortlist"):
    feats, value, prop, offers = state["features"], state["value"], state["propensity"], state.get("offers") or {}
    st.subheader("Shortlist")
    st.dataframe(pd.DataFrame([{
        "#": s["rank"], "Customer": feats[s["customer_id"]]["name"], "ID": s["customer_id"],
        "City": feats[s["customer_id"]]["city"], "Segment": feats[s["customer_id"]]["segment"],
        "Value": value[s["customer_id"]]["value_score"],
        "P(convert)": f"{prop[s['customer_id']]['probability']:.0%}",
        "Top reason": (prop[s["customer_id"]]["drivers"] or ["-"])[0],
        "Offer": summarise_offer(offers.get(s["customer_id"])),
    } for s in state["shortlist"]]), hide_index=True, use_container_width=True)

if state.get("drafts"):
    st.subheader("WhatsApp drafts")
    cols = st.columns(2)
    for i, d in enumerate(state["drafts"]):
        with cols[i % 2].container(border=True):
            st.markdown(f"**#{d['no']} {d['name']}** · `{d['status']}` · {d['language']} · {d['method']}")
            st.text(d["body"])
            for v in d["violations"]:
                st.error(f"{v['rule_id']}: {v['message']}")

with st.expander("Plan and trace (observability)"):
    if state.get("plan"):
        st.table(pd.DataFrame(state["plan"]))
    spans = session.trace()[-30:]
    if spans:
        st.dataframe(pd.DataFrame([{"kind": s["kind"], "name": s["name"], "ms": s["ms"],
                                    "status": s.get("status"), "detail": str(s.get("result", ""))[:80]}
                                   for s in spans]), hide_index=True, use_container_width=True)
