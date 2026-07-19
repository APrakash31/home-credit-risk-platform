import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from datetime import date

MAX_MESSAGES_PER_SESSION = 8
MAX_MESSAGES_PER_DAY = 200


@st.cache_resource
def _global_counter():
    return {"date": date.today(), "count": 0}


def check_rate_limit():
    """Returns (allowed, message)."""
    counter = _global_counter()
    if counter["date"] != date.today():
        counter["date"] = date.today()
        counter["count"] = 0

    if counter["count"] >= MAX_MESSAGES_PER_DAY:
        return False, "Daily demo limit reached. Please try again tomorrow."

    used = st.session_state.get("agent_calls", 0)
    if used >= MAX_MESSAGES_PER_SESSION:
        return False, (
            f"Session limit of {MAX_MESSAGES_PER_SESSION} questions reached. "
            "Refresh the page to start a new session."
        )

    return True, ""


def record_call():
    st.session_state["agent_calls"] = st.session_state.get("agent_calls", 0) + 1
    _global_counter()["count"] += 1

st.set_page_config(page_title="Credit Risk Platform", page_icon="●", layout="wide")

FEATURES = ROOT / "data" / "processed" / "features.parquet"


@st.cache_data
def load_data():
    full = ROOT / "data" / "processed" / "features.parquet"
    demo = ROOT / "app" / "demo_data" / "features_sample.parquet"
    path = full if full.exists() else demo
    return pd.read_parquet(path)


@st.cache_resource
def get_explainer():
    from src.explain.reason_codes import ApplicantExplainer
    return ApplicantExplainer()


@st.cache_resource
def get_agent():
    from src.agent.agent import CreditAgent
    return CreditAgent()


def clean_cols(df):
    df = df.copy()
    df.columns = [c.replace(" ", "_").replace(":", "_").replace(",", "_") for c in df.columns]
    return df


st.title("Credit Risk Decisioning Platform")
st.caption(
    "Model-assisted risk assessment with SHAP explainability and a tool-using AI analyst. "
    "All decisions require human underwriter review."
)

tab1, tab2, tab3 = st.tabs(["Portfolio overview", "Applicant assessment", "AI analyst"])

# ---------------------------------------------------------------- Portfolio
with tab1:
    df = load_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{len(df):,}")
    c2.metric("Default rate", f"{df['TARGET'].mean():.2%}")
    c3.metric("Model AUC", "0.786")
    c4.metric("Features", f"{df.shape[1]:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Default rate by employment length")
        d = df.copy()
        d["band"] = pd.cut(
            d["YEARS_EMPLOYED"],
            bins=[0, 2, 5, 10, 20, 50],
            labels=["0-2y", "2-5y", "5-10y", "10-20y", "20y+"],
        )
        agg = d.groupby("band", observed=True)["TARGET"].agg(["mean", "size"]).reset_index()
        agg["mean"] = (agg["mean"] * 100).round(2)
        fig = px.bar(agg, x="band", y="mean", labels={"mean": "Default rate (%)", "band": ""})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Default rate by external score")
        d = df.dropna(subset=["EXT_SOURCE_2"]).copy()
        d["band"] = pd.qcut(d["EXT_SOURCE_2"], q=5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
        agg = d.groupby("band", observed=True)["TARGET"].mean().reset_index()
        agg["TARGET"] = (agg["TARGET"] * 100).round(2)
        fig = px.bar(agg, x="band", y="TARGET", labels={"TARGET": "Default rate (%)", "band": ""})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global feature importance")
    imp_path = ROOT / "reports" / "shap_global_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).head(15).sort_values("mean_abs_shap")
        fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Applicant
with tab2:
    df = load_data()
    st.subheader("Individual applicant assessment")

    ids = df["SK_ID_CURR"].head(200).tolist()
    applicant_id = st.selectbox("Select applicant ID", ids)

    if st.button("Assess applicant", type="primary"):
        row = df[df["SK_ID_CURR"] == applicant_id]
        X = clean_cols(row.drop(columns=[c for c in ["TARGET", "SK_ID_CURR"] if c in row.columns]))

        with st.spinner("Scoring and explaining..."):
            result = get_explainer().explain(X)

        prob = result["default_probability"]
        band = result["risk_band"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Default probability", f"{prob:.2%}")
        c2.metric("Risk band", band.upper())
        c3.metric("Policy outcome", "Auto-approve" if band == "low" else "Requires review")

        if band != "low":
            st.warning("This application requires human underwriter review under credit policy section 1.")

        st.divider()

        left, right = st.columns(2)
        with left:
            st.markdown("**Factors increasing risk**")
            for d in result["top_risk_drivers"]:
                st.markdown(f"- {d['reason_code']}")
        with right:
            st.markdown("**Factors reducing risk**")
            for d in result["top_protective_factors"]:
                st.markdown(f"- {d['reason_code']}")

        with st.expander("Technical detail (internal use only)"):
            st.json(result)

# ---------------------------------------------------------------- Agent
with tab3:
    st.subheader("AI credit analyst")
    st.caption(
        "Ask about applicants, the portfolio, or credit policy. The agent retrieves policy, "
        "queries data, scores applicants, and explains decisions using its tools."
    )

    examples = [
        "Assess applicant 100002 and give me a recommendation.",
        "What is the default rate by employment length?",
        "Applicant 100003 was declined — what reasons should we communicate?",
        "What is our policy on thin-file applicants?",
    ]
    st.markdown("**Try:** " + " · ".join(f"`{e}`" for e in examples))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                with st.expander("Tools used"):
                    for t in m["trace"]:
                        st.markdown(f"**{t['tool']}** — `{t['arguments']}`")

    used = st.session_state.get("agent_calls", 0)
    st.caption(f"Demo usage: {used}/{MAX_MESSAGES_PER_SESSION} questions this session")

    if prompt := st.chat_input("Ask the credit analyst..."):
        allowed, limit_msg = check_rate_limit()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not allowed:
                st.warning(limit_msg)
                answer, trace = limit_msg, []
            else:
                with st.spinner("Analysing..."):
                    try:
                        record_call()
                        result = get_agent().run(prompt, verbose=False)
                        answer = result["answer"]
                        trace = result.get("trace", [])
                    except Exception as e:
                        answer = f"Error: {e}"
                        trace = []
                st.markdown(answer)
                if trace:
                    with st.expander("Tools used"):
                        for t in trace:
                            st.markdown(f"**{t['tool']}** — `{t['arguments']}`")

        st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})