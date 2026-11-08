"""Streamlit workspace for the RM Copilot.   streamlit run app/streamlit_app.py

Layout: conversation on the left, a tabbed workspace on the right (leads, drafts, customer 360,
activity). Every button simply sends the same text command the RM could type, so the agent's
behaviour (and audit trail) is identical whichever way the RM works.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from crm_agent.config import settings  # noqa: E402
from crm_agent.formatting import inr  # noqa: E402
from crm_agent.orchestration.nodes import summarise_offer  # noqa: E402
from crm_agent.orchestration.session import AgentSession  # noqa: E402

st.set_page_config(page_title="RM Copilot", page_icon=":material/account_balance:", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- constants
PRODUCTS = {"personal_loan": "Personal loan", "credit_card": "Credit card", "loan_against_fd": "Loan against FD"}
LANGS = {"en": "English", "hi": "Hindi"}
STATUS = {  # status -> (label, badge colour, icon)
    "ready": ("Ready for approval", "blue", ":material/pending_actions:"),
    "approved": ("Approved", "violet", ":material/task_alt:"),
    "sent": ("Queued on WhatsApp", "green", ":material/send:"),
    "blocked": ("Blocked by compliance", "red", ":material/gpp_bad:"),
    "needs_revision": ("Needs revision", "orange", ":material/edit_note:"),
    "rejected": ("Rejected", "gray", ":material/block:"),
    "send_failed": ("Send failed", "red", ":material/error:"),
    "checking": ("Checking", "gray", ":material/hourglass_top:"),
}
# Material Symbols "account_balance" (Apache-2.0), used as the product mark in the header.
LOGO_SVG = ('<svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor" aria-hidden="true">'
            '<path d="M4 10v7h3v-7H4zm6 0v7h3v-7h-3zM2 22h19v-3H2v3zm14-12v7h3v-7h-3zm-4.5-9L2 6v2h19V6l-9.5-5z"/>'
            '</svg>')
EXAMPLES = [
    (":material/person_search: Find personal-loan leads", "Find high-value customers likely to convert for a personal loan this month "
                                    "and generate personalised WhatsApp messages"),
    (":material/filter_alt: Salaried in Mumbai, in Hindi", "Only salaried customers in Mumbai, top 5, in Hindi"),
    (":material/credit_card: Credit-card prospects", "Which customers should I pitch a credit card to?"),
    (":material/outbox: Show outbox", "show outbox"),
]

CSS = """
<style>
[data-testid="stAppDeployButton"], footer {display: none;}
.block-container {padding-top: 1.6rem; padding-bottom: 6rem; max-width: 1500px;}
.hero {display:flex; justify-content:space-between; align-items:center; padding:12px 18px; border-radius:10px;
       background:#fff; border:1px solid #E3E8EF; border-top:3px solid #1F4E79; margin-bottom:14px;}
.brand {display:flex; align-items:center; gap:12px;}
.mark {width:40px; height:40px; border-radius:8px; background:#1F4E79; color:#fff;
       display:flex; align-items:center; justify-content:center;}
.hero h1 {font-size:1.25rem; margin:0; color:#1B2430; padding:0; font-weight:700; letter-spacing:-.01em;}
.hero p {margin:1px 0 0; color:#5B6B7F; font-size:.85rem;}
.pill {display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:999px; font-size:.78rem;
       font-weight:600; background:#EEF2F7; color:#33475B; margin-left:6px; white-space:nowrap;}
.pill .dot {width:7px; height:7px; border-radius:50%; background:#2E8B57;}
.pill.warn {background:#FFF4E0; color:#8A5300;} .pill.warn .dot {background:#E08E0B;}
.funnel {display:flex; gap:6px; align-items:stretch; margin:4px 0 14px;}
.stage {flex:1; background:#fff; border:1px solid #E3E8EF; border-radius:10px; padding:10px 12px;}
.stage .n {font-size:1.35rem; font-weight:700; color:#1F4E79;}
.stage .l {font-size:.75rem; color:#5B6B7F; text-transform:uppercase; letter-spacing:.03em;}
.stage.last {background:#1F4E79;} .stage.last .n, .stage.last .l {color:#fff;}
.arrow {align-self:center; color:#9AA8B8;}
.bubble {background:#E7F7DC; border-radius:4px 12px 12px 12px; padding:12px 14px; white-space:pre-wrap;
         font-size:.9rem; line-height:1.45; color:#1B2430; border:1px solid #D2EBC2;}
.muted {color:#5B6B7F; font-size:.82rem;}
.kv {display:flex; justify-content:space-between; border-bottom:1px dashed #E3E8EF; padding:4px 0; font-size:.9rem;}
.kv span:first-child {color:#5B6B7F;}
</style>
"""


# --------------------------------------------------------------------------- session helpers
def reset() -> None:
    st.session_state.session = AgentSession()
    st.session_state.chat = []
    st.session_state.pending = None


def queue(command: str) -> None:
    """Button callback: run `command` through the agent on this rerun."""
    st.session_state.pending = command


if "session" not in st.session_state:
    reset()
session: AgentSession = st.session_state.session


def split_response(text: str) -> tuple[str, list[str]]:
    """Separate the RM-facing answer from the agent's bookkeeping lines (shown as captions)."""
    body, meta = [], []
    for line in (text or "").splitlines():
        if line.startswith(("Steps run this turn:", "Understood:")):
            meta.append(line)
        else:
            body.append(line)
    return "  \n".join(body).strip(), meta


def render_message(role: str, text: str) -> None:
    with st.chat_message(role, avatar=":material/person:" if role == "user" else ":material/account_balance:"):
        if role == "user":
            st.markdown(text)
            return
        body, meta = split_response(text)
        st.markdown(body or "_(no response)_")
        for m in meta:
            st.caption(m)


# --------------------------------------------------------------------------- run any pending command
prompt = st.chat_input("Ask about your customers, or e.g. “approve 1,3”, “edit 2: make it more formal”…")
command = prompt or st.session_state.pop("pending", None)
st.session_state.pending = None
if command:
    st.session_state.chat.append(("user", command))
    with st.spinner("Copilot is working…"):
        try:
            result = session.send(command)
            st.session_state.chat.append(("assistant", result.get("response", "")))
        except Exception as exc:  # keep the UI alive and tell the RM what happened
            st.session_state.chat.append(("assistant", f":red[:material/error:] Something went wrong: `{exc}`"))

state = session.state()
drafts = state.get("drafts") or []
shortlist = state.get("shortlist") or []
awaiting = session.awaiting_approval

# --------------------------------------------------------------------------- chrome
st.markdown(CSS, unsafe_allow_html=True)
mode = "Offline (rules + templates)" if settings.llm_provider == "offline" else f"LLM: {settings.llm_provider}"
status_pill = ('<span class="pill warn"><span class="dot"></span>Awaiting your approval</span>' if awaiting and drafts
               else '<span class="pill"><span class="dot"></span>Ready</span>')
st.markdown(
    f'<div class="hero"><div class="brand"><div class="mark">{LOGO_SVG}</div><div><h1>RM Copilot</h1>'
    f'<p>{html.escape(settings.bank_name)} · Relationship Manager {html.escape(settings.rm_name)}</p></div></div><div><span class="pill">{html.escape(mode)}</span>{status_pill}</div></div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {settings.bank_name}")
    st.caption(f"Signed in as **{settings.rm_name}** · session `{session.thread_id}`")
    st.button("New conversation", icon=":material/add:", on_click=reset, width="stretch", type="primary")
    st.divider()
    st.markdown("**Quick start**")
    for label, text in EXAMPLES:
        st.button(label, on_click=queue, args=(text,), width="stretch", key=f"ex-{label}")
    st.divider()
    with st.expander("Commands you can type"):
        st.markdown("- `Why is #1 on the list?`\n- `approve 1,3` · `approve all`\n- `reject 2`\n"
                    "- `edit 2: make it more formal`\n- `edit 2 text: <your own message>`\n"
                    "- `include medium value` · `top 5` · `in Hindi`")
    with st.expander("How it works"):
        st.markdown("Scoring, credit policy, offer sizing and compliance checks are deterministic. "
                    "The language model (if enabled) only reads requests and writes messages.\n\n"
                    "**Nothing is sent until you approve it.**")

# --------------------------------------------------------------------------- layout
left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown("#### :material/forum: Conversation")
    chat_box = st.container(height=640, border=True)
    with chat_box:
        if not st.session_state.chat:
            with st.chat_message("assistant", avatar=":material/account_balance:"):
                st.markdown(f"Hi {settings.rm_name}, I can find customers to target, explain why they qualify, "
                            "and draft compliant WhatsApp messages for your approval.\n\n"
                            "Pick a quick start on the left, or type a request below.")
        for role, text in st.session_state.chat:
            render_message(role, text)


def funnel_html(funnel: dict, tier: str) -> str:
    stages = [("in_scope", "In scope"), ("value_ok", f"{tier.title()} value"), ("eligible", "Credit-eligible"),
              ("likely_to_convert", "Likely to convert"), ("selected", "Shortlisted")]
    cells = []
    for i, (k, label) in enumerate(stages):
        cls = "stage last" if i == len(stages) - 1 else "stage"
        cells.append(f'<div class="{cls}"><div class="n">{funnel.get(k, 0):,}</div><div class="l">{label}</div></div>')
    return '<div class="funnel">' + '<div class="arrow">&rsaquo;</div>'.join(cells) + "</div>"


def leads_tab() -> None:
    if not shortlist:
        st.info("No leads yet. Ask the copilot to find customers, e.g. *“Find high-value customers likely to "
                "convert for a personal loan this month.”*")
        return
    intent = state.get("intent") or {}
    product = PRODUCTS.get(state.get("scored_product") or intent.get("product_type"), "—")
    filters = intent.get("filters") or {}
    chips = [f"**Product:** {product}", f"**Top:** {intent.get('top_n', '—')}",
             f"**Window:** {intent.get('timeframe_days', '—')} days"]
    chips += [f"**{k.replace('_', ' ').title()}:** {', '.join(map(str, v)) if isinstance(v, list) else v}"
              for k, v in filters.items() if v]
    st.markdown(" · ".join(chips))
    st.markdown(funnel_html(state.get("funnel") or {}, intent.get("min_value_tier", "high")), unsafe_allow_html=True)

    feats, value, prop = state["features"], state["value"], state["propensity"]
    offers, alts = state.get("offers") or {}, state.get("alternatives") or {}
    status_by_cid = {d["customer_id"]: STATUS.get(d["status"], (d["status"], ""))[0] for d in drafts}
    excluded = {x["customer_id"]: x["reason"] for x in state.get("excluded") or []}
    rows = []
    for s in shortlist:
        cid = s["customer_id"]
        f, p = feats[cid], prop[cid]
        offer = summarise_offer(offers[cid]) if cid in offers else (
            f"Alternative: {summarise_offer(alts[cid])}" if cid in alts else "—")
        rows.append({
            "#": s["rank"], "Customer": f["name"], "ID": cid, "City": f["city"],
            "Segment": str(f["segment"]).title(), "Value": round(value[cid]["value_score"] * 100),
            "Conversion": round(p["probability"] * 100), "Why now": (p["drivers"] or ["—"])[0],
            "Recommended offer": offer,
            "Outreach": status_by_cid.get(cid) or ("Not contacted" if cid in excluded else "—"),
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "#": st.column_config.NumberColumn(width=40),
            "Value": st.column_config.ProgressColumn("Value", min_value=0, max_value=100, format="%d",
                                                     help="Relationship value score, ranked across the whole book"),
            "Conversion": st.column_config.ProgressColumn("Conversion", min_value=0, max_value=100, format="%d%%",
                                                          help="Estimated probability of converting"),
            "Why now": st.column_config.TextColumn(width="large"),
            "Recommended offer": st.column_config.TextColumn(width="large"),
        })
    if excluded:
        with st.expander(f"{len(excluded)} shortlisted customer(s) not contacted", icon=":material/do_not_disturb_on:"):
            st.dataframe(pd.DataFrame([{"Customer": feats[c]["name"] if c in feats else c, "ID": c, "Reason": r}
                                       for c, r in excluded.items()]), hide_index=True, width="stretch")


def draft_card(d: dict) -> None:
    label, colour, icon = STATUS.get(d["status"], (d["status"], "gray", None))
    with st.container(border=True):
        top, badge = st.columns([3, 2])
        top.markdown(f"**#{d['no']} · {d['name']}**  \n"
                     f"<span class='muted'>{d['customer_id']} · {LANGS.get(d['language'], d['language'])} · "
                     f"{'AI-written' if d['method'] != 'template' else 'Template'}</span>", unsafe_allow_html=True)
        with badge:
            st.badge(label, color=colour, icon=icon)
        st.markdown(f'<div class="bubble">{html.escape(d["body"])}</div>', unsafe_allow_html=True)
        for v in d.get("violations") or []:
            st.error(f"**{v['rule_id']}** — {v['message']}", icon=":material/gpp_bad:")
        if d.get("message_id"):
            st.caption(f"Message ID `{d['message_id']}`")

        n = d["no"]
        if d["status"] in ("ready", "blocked", "needs_revision"):
            a, r, e = st.columns(3)
            a.button("Approve", icon=":material/check:", key=f"ap-{n}", on_click=queue, args=(f"approve {n}",), type="primary",
                     width="stretch", disabled=d["status"] != "ready",
                     help=None if d["status"] == "ready" else "Blocked drafts can't be approved — edit it first")
            r.button("Reject", icon=":material/close:", key=f"rj-{n}", on_click=queue, args=(f"reject {n}",), width="stretch")
            with e.popover("Edit", icon=":material/edit:", width="stretch"):
                instr = st.text_input("Ask the copilot to rewrite", key=f"ei-{n}",
                                      placeholder="e.g. make it more formal, shorter, mention the EMI")
                st.button("Rewrite", key=f"eb-{n}", disabled=not instr, width="stretch",
                          on_click=queue, args=(f"edit {n}: {instr}",))
                st.divider()
                own = st.text_area("…or write it yourself", value=d["body"], key=f"et-{n}", height=180)
                st.button("Use my text", key=f"es-{n}", width="stretch",
                          on_click=queue, args=(f"edit {n} text: {own}",),
                          help="Your text is still checked for compliance before it can be approved")


def drafts_tab() -> None:
    if not drafts:
        st.info("No drafts yet. Ask for leads *and* WhatsApp messages to generate drafts for approval.")
        return
    counts = {k: sum(d["status"] == k for d in drafts) for k in STATUS}
    c = st.columns(4)
    c[0].metric("Ready", counts["ready"])
    c[1].metric("Blocked", counts["blocked"] + counts["needs_revision"] + counts["send_failed"])
    c[2].metric("Queued", counts["sent"])
    c[3].metric("Rejected", counts["rejected"])

    show = st.segmented_control("Show", ["Needs action", "All", "Done"], default="Needs action",
                                key="draft-filter", label_visibility="collapsed")
    if counts["ready"]:
        st.button(f"Approve all {counts['ready']} ready drafts", icon=":material/done_all:", type="primary", on_click=queue,
                  args=("approve all",))
    active = ("ready", "blocked", "needs_revision", "approved", "send_failed")
    subset = [d for d in drafts if show == "All" or (show == "Done") != (d["status"] in active)]
    if not subset:
        st.success("Nothing here — all caught up.")
    cols = st.columns(2)
    for i, d in enumerate(subset):
        with cols[i % 2]:
            draft_card(d)


def kv(rows: list[tuple[str, str]]) -> None:
    st.markdown("".join(f'<div class="kv"><span>{html.escape(k)}</span><span><b>{html.escape(str(v))}</b>'
                        f'</span></div>' for k, v in rows), unsafe_allow_html=True)


def customer_tab() -> None:
    if not shortlist:
        st.info("Customer profiles appear here once you have a shortlist.")
        return
    feats = state["features"]
    options = {s["customer_id"]: f"#{s['rank']} · {feats[s['customer_id']]['name']} ({s['customer_id']})"
               for s in shortlist}
    cid = st.selectbox("Customer", list(options), format_func=options.get, label_visibility="collapsed")
    f, v, p = feats[cid], state["value"][cid], state["propensity"][cid]
    elig = (state.get("eligibility") or {}).get(cid) or {}
    offer = (state.get("offers") or {}).get(cid) or (state.get("alternatives") or {}).get(cid)
    rank = next(s["rank"] for s in shortlist if s["customer_id"] == cid)

    m = st.columns(4)
    m[0].metric("Conversion", f"{p['probability']:.0%}")
    m[1].metric("Value tier", str(v.get("value_tier", "—")).title(), f"score {v['value_score']:.2f}",
                delta_color="off")
    m[2].metric("Credit score", f.get("credit_score", "—"))
    m[3].metric("Monthly income", f"₹{inr(f.get('monthly_income') or 0)}")

    a, b = st.columns(2)
    with a:
        st.markdown("**Profile**")
        kv([("Age", f.get("age")), ("City", f.get("city")), ("Segment", str(f.get("segment")).title()),
            ("Occupation", str(f.get("occupation_type", "—")).replace("_", " ").title()),
            ("With bank", f"{(f.get('tenure_months') or 0) / 12:.1f} yrs"),
            ("Language", LANGS.get(f.get("preferred_language"), f.get("preferred_language"))),
            ("WhatsApp consent", "Yes" if f.get("whatsapp_opt_in") else "No"),
            ("Existing EMIs", f"₹{inr(f.get('monthly_emi') or 0)}/mo")])
    with b:
        st.markdown("**Recommended offer**")
        if offer:
            kv([("Product", offer["product_name"]), ("Amount", f"₹{inr(offer['amount'])}"),
                ("Rate", f"{offer['rate']}%" if offer.get("rate") is not None else "—"),
                ("EMI", f"₹{inr(offer['emi'])}/mo" if offer.get("emi") else "—"),
                ("Tenure", f"{offer['tenure_months']} months" if offer.get("tenure_months") else "—"),
                ("Limited by", offer.get("binding_constraint", "—"))])
        else:
            st.caption("No suitable offer.")

    st.markdown("**Why they may convert now**")
    for d in p.get("drivers") or []:
        st.markdown(f":green[:material/check_circle:] {d}")
    for w in p.get("watch_outs") or []:
        st.markdown(f":orange[:material/warning:] {w}")
    if elig.get("checks"):
        with st.expander("Credit policy checks", expanded=not elig.get("eligible", True)):
            st.dataframe(pd.DataFrame([{"Result": "Pass" if c["passed"] else "Fail", "Rule": c["description"],
                                        "Detail": c["detail"]} for c in elig["checks"]]),
                         hide_index=True, width="stretch")
    st.button("Ask the copilot to explain this decision", icon=":material/psychology_alt:", on_click=queue,
              args=(f"Why is #{rank} on the list?",))


def activity_tab() -> None:
    plan, executed = state.get("plan") or [], state.get("executed") or []
    if plan:
        st.markdown("**Plan for the last message**")
        st.dataframe(pd.DataFrame([{"#": i, "Step": p["step"].replace("_", " "), "Why": p["why"],
                                    "Ran": p["step"] in executed}
                                   for i, p in enumerate(plan, 1)]), hide_index=True, width="stretch")
    spans = session.trace()[-40:]
    if spans:
        st.markdown("**Trace** (most recent spans)")
        st.dataframe(pd.DataFrame([{"Kind": s["kind"], "Name": s["name"], "ms": s["ms"],
                                    "Status": s.get("status"), "Detail": str(s.get("result") or "")[:100]}
                                   for s in reversed(spans)]), hide_index=True, width="stretch", height=380)
    if not plan and not spans:
        st.info("The agent's plan and an execution trace appear here after your first request.")


with right:
    ready = sum(d["status"] == "ready" for d in drafts)
    t_leads, t_drafts, t_cust, t_act = st.tabs([
        f":material/groups: Leads ({len(shortlist)})" if shortlist else ":material/groups: Leads",
        f":material/chat: Drafts · {ready} to review" if ready else ":material/chat: Drafts",
        ":material/badge: Customer 360", ":material/timeline: Activity"])
    with t_leads:
        leads_tab()
    with t_drafts:
        drafts_tab()
    with t_cust:
        customer_tab()
    with t_act:
        activity_tab()
