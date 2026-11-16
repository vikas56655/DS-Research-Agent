"""Interactive Data Science Research Agent — Streamlit entry point."""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from research.ai import interpret
from research.analysis import baseline, load_csv, profile, sample_data

load_dotenv()
st.set_page_config(page_title="Atlas | DS Research Agent", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top:2rem;max-width:1350px}
.hero {background:linear-gradient(120deg,#172441,#11192d);padding:28px 32px;
border:1px solid #33415e;border-radius:18px;margin-bottom:24px}
.eyebrow {letter-spacing:3px;color:#ff986f;font-size:12px;font-weight:700}
.hero h1 {font-size:2.6rem;margin:8px 0}
.hero p {color:#aebbd0;margin-bottom:0}
[data-testid="stMetric"] {background:#151d31;border:1px solid #293852;
padding:16px;border-radius:12px}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">ATLAS / RESEARCH WORKSPACE</div>'
            '<h1>Explore data. Find the signal.</h1>'
            '<p>Interactive profiling, statistical visualization, reproducible baselines '
            'and optional AI-assisted interpretation.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("01 / Data source")
    source = st.radio("Choose a dataset", ["Demo dataset", "Upload CSV"], label_visibility="collapsed")
    uploaded = st.file_uploader("Upload UTF-8 CSV (max 15 MB)", type=["csv"]) if source == "Upload CSV" else None
    st.divider()
    st.header("02 / AI interpretation")
    st.caption("Optional. Your API key stays in this session and is not written to disk.")
    key = st.text_input("OpenAI API key", type="password",
                        value=os.getenv("OPENAI_API_KEY", ""))
    st.caption("Only aggregate dataset summaries are sent to the API, never raw rows.")
    st.divider()
    st.caption("Synthetic demo data • No external data collection • No code execution")

if source == "Upload CSV" and uploaded is None:
    st.info("Upload a CSV to begin, or select the built-in demo dataset.")
    st.stop()
try:
    df = sample_data() if source == "Demo dataset" else load_csv(uploaded.getvalue())
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Bound large files for responsive charts and safer computation.
if len(df) > 100_000:
    st.warning("Dataset exceeds 100,000 rows; using a reproducible 100,000-row sample.")
    df = df.sample(100_000, random_state=42).reset_index(drop=True)
p = profile(df)
a, b, c, d = st.columns(4)
a.metric("Observations", f"{p['rows']:,}")
b.metric("Features", str(p["columns"]))
c.metric("Missing cells", f"{p['missing']:,}")
d.metric("Duplicate rows", f"{p['duplicates']:,}")

tabs = st.tabs(["Overview", "Distributions", "Relationships", "Model lab", "Research brief"])

with tabs[0]:
    st.subheader("Dataset at a glance")
    st.caption("Preview shows at most 100 records; uploaded datasets are processed in memory.")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        missing = p["missing_by_column"].head(20)
        fig = px.bar(x=missing.values, y=missing.index, orientation="h",
                     labels={"x": "Missing values", "y": "Feature"},
                     title="01 / Missingness by feature")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        types = df.dtypes.astype(str).value_counts()
        fig = px.pie(names=types.index, values=types.values, hole=0.6,
                     title="02 / Column types")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    numeric = p["numeric_columns"]
    categories = p["categorical_columns"]
    left, right = st.columns(2)
    with left:
        if numeric:
            field = st.selectbox("Numeric feature", numeric, key="hist_field")
            fig = px.histogram(df, x=field, nbins=35, marginal="box",
                               title="03 / Distribution & outliers")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric features available.")
    with right:
        if categories:
            field = st.selectbox("Categorical feature", categories, key="cat_field")
            counts = df[field].fillna("(missing)").astype(str).value_counts().head(15)
            fig = px.bar(x=counts.index, y=counts.values,
                         labels={"x": field, "y": "Count"},
                         title="04 / Top categories")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical features available.")

with tabs[2]:
    numeric = p["numeric_columns"]
    if len(numeric) >= 2:
        left, right = st.columns(2)
        with left:
            corr = df[numeric].corr(numeric_only=True)
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="05 / Pearson correlation matrix", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            x = st.selectbox("X-axis", numeric, index=0)
            y = st.selectbox("Y-axis", numeric, index=1)
            fig = px.scatter(df.sample(min(2500, len(df)), random_state=42),
                             x=x, y=y, opacity=0.65,
                             title="06 / Feature relationship")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation measures linear association, not causation.")
    else:
        st.info("At least two numeric columns are needed for relationship charts.")

with tabs[3]:
    st.subheader("Train a transparent baseline")
    st.caption("Random 75/25 holdout; no hyperparameter tuning. "
               "Exploratory results are not deployment-ready estimates.")
    target = st.selectbox("Target variable", list(df.columns))
    task = st.radio("Task", ["classification", "regression"], horizontal=True)
    if st.button("Run baseline", type="primary"):
        try:
            result = baseline(df, target, task)
            st.session_state["baseline_result"] = result
            st.session_state["baseline_signature"] = (source, uploaded.name if uploaded else None,
                                                       len(df), target, task)
        except ValueError as exc:
            st.error(str(exc))
    signature = (source, uploaded.name if uploaded else None, len(df), target, task)
    result = st.session_state.get("baseline_result") if st.session_state.get("baseline_signature") == signature else None
    if result:
        metrics = st.columns(len(result["metrics"]))
        for col, (name, value) in zip(metrics, result["metrics"].items()):
            col.metric(name, f"{value:.3f}")
        st.caption(f"Train: {result['train_rows']} rows | Test: {result['test_rows']} rows")
        if result["dropped_high_cardinality"]:
            st.warning("Excluded high-cardinality predictors: " + ", ".join(result["dropped_high_cardinality"]))
        if task == "regression":
            fig = px.scatter(x=result["actual"], y=result["predicted"],
                             labels={"x": "Actual", "y": "Predicted"},
                             title="07 / Predicted vs actual")
            st.plotly_chart(fig, use_container_width=True)
        else:
            table = pd.crosstab(pd.Series(result["actual"], name="Actual"),
                                pd.Series(result["predicted"], name="Predicted"))
            fig = px.imshow(table, text_auto=True, color_continuous_scale="Blues",
                            title="07 / Confusion matrix")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Warning: random splits can leak temporal or grouped information. "
                   "Check target leakage, class balance and validation design.")

with tabs[4]:
    st.subheader("Research brief")
    st.caption("Download a deterministic Markdown summary or request optional AI commentary.")
    summary = {
        "rows": p["rows"], "columns": p["columns"], "missing_cells": p["missing"],
        "duplicate_rows": p["duplicates"], "numeric_features": p["numeric_columns"][:30],
        "categorical_features": p["categorical_columns"][:30],
        "missing_by_column": p["missing_by_column"].head(30).to_dict(),
    }
    report = ("# Dataset research brief\n\n"
              f"- Observations: {p['rows']:,}\n- Features: {p['columns']}\n"
              f"- Missing cells: {p['missing']:,}\n- Duplicate rows: {p['duplicates']:,}\n\n"
              "## Numeric summary\n\n" + df.describe(include="number").to_markdown()
              if p["numeric_columns"] else
              f"# Dataset research brief\n\nRows: {p['rows']:,}\nColumns: {p['columns']}\n")
    report += "\n\n## Limitations\n\nDescriptive statistics alone cannot establish causality.\n"
    st.download_button("Download Markdown brief", report, file_name="research_brief.md",
                       mime="text/markdown")
    question = st.text_area("Research question", value="What patterns and data quality concerns stand out?")
    if st.button("Generate AI interpretation"):
        if not key:
            st.warning("Enter an OpenAI API key in the sidebar first.")
        else:
            try:
                with st.spinner("Interpreting aggregate statistics..."):
                    answer = interpret(summary, question, key)
                st.markdown(answer)
            except Exception as exc:
                st.error(f"AI request failed ({type(exc).__name__}). Check your API key, network and model access.")

st.caption("Atlas Research Agent · Open-source portfolio demo · Synthetic sample data")
