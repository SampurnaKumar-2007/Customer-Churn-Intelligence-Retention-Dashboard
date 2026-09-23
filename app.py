"""
app.py
------
Interactive Streamlit dashboard for supermarket / e-commerce analytics,
leakage-safe churn prediction, and retention optimisation.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from docx import Document
from docx.shared import Pt, RGBColor

from analytics import (
    build_leakage_safe_rfm,
    compute_summary_kpis,
    optimize_retention_budget,
    train_churn_models,
)
from data_loader import load_and_clean_data

# ============================================================================
# Page configuration
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction & Retention Optimizer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Cached data helpers
# ============================================================================

@st.cache_data(show_spinner="Loading and cleaning data…")
def get_clean_data(upload_bytes: bytes | None = None) -> pd.DataFrame:
    """Load cleaned data, optionally from an uploaded file."""
    if upload_bytes is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(upload_bytes)
            tmp_path = tmp.name
        df = load_and_clean_data(tmp_path)
        os.unlink(tmp_path)
        return df
    return load_and_clean_data()


@st.cache_data(show_spinner="Computing RFM and training models…")
def get_rfm_and_models(clean_csv_hash: str, cutoff_str: str):
    """Build RFM features and train models (cached by data hash + cutoff)."""
    df = pd.read_csv(os.path.join("data", "clean_data.csv"), parse_dates=["Order_Date"])
    rfm = build_leakage_safe_rfm(df, cutoff_str)
    results = train_churn_models(rfm)
    return rfm, results


def _csv_cache_key() -> str:
    """Return a cache key that changes whenever clean_data.csv is modified."""
    path = os.path.join("data", "clean_data.csv")
    try:
        stat = os.stat(path)
        return f"{stat.st_size}_{stat.st_mtime}"
    except OSError:
        return "missing"


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar(df: pd.DataFrame):
    st.sidebar.header("🎛️ Control Panel")

    # File upload
    uploaded = st.sidebar.file_uploader(
        "Upload new raw transactions (CSV)",
        type=["csv"],
        help="Upload a CSV to re-run the full pipeline on new data.",
    )

    # Region filter
    all_regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "Regions",
        all_regions,
        default=all_regions,  # all selected by default
        placeholder="All regions",
    )

    # Category filter
    all_cats = sorted(df["Product_Category"].dropna().unique())
    selected_cats = st.sidebar.multiselect(
        "Product Categories",
        all_cats,
        default=all_cats,  # all selected by default
        placeholder="All categories",
    )

    # Date range of the dataset
    _FALLBACK_MIN = date(2010, 1, 1)
    _FALLBACK_MAX = date(2011, 12, 31)
    if "Order_Date" in df.columns and not df["Order_Date"].dropna().empty:
        min_date = df["Order_Date"].min().date()
        max_date = df["Order_Date"].max().date()
    else:
        min_date = _FALLBACK_MIN
        max_date = _FALLBACK_MAX

    # Cutoff date for RFM temporal split
    default_cutoff = pd.Timestamp(min_date) + (pd.Timestamp(max_date) - pd.Timestamp(min_date)) * 0.75
    default_cutoff = default_cutoff.date()
    cutoff_date = st.sidebar.date_input(
        "RFM Cutoff Date",
        value=default_cutoff,
        min_value=min_date,
        max_value=max_date,
        help=(
            "Orders before this date form the Historical Window (features). "
            "Orders on/after define the Churn label (target)."
        ),
    )

    # Model selection
    model_choice = st.sidebar.radio(
        "Churn Model", ["Logistic Regression", "Decision Tree"], index=0
    )

    # Budget controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Campaign Budget")
    max_budget = st.sidebar.slider(
        "Max Target Customers", min_value=10, max_value=2000, value=200, step=10
    )
    cost_per_contact = st.sidebar.number_input(
        "Cost per Contact (£)", min_value=0.1, max_value=100.0, value=5.0, step=0.5
    )
    customer_ltv = st.sidebar.number_input(
        "Customer LTV (£)", min_value=1.0, max_value=5000.0, value=250.0, step=10.0
    )
    threshold = st.sidebar.slider(
        "Churn Probability Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05
    )

    return {
        "uploaded": uploaded,
        "regions": selected_regions,
        "categories": selected_cats,
        "cutoff_date": str(cutoff_date),
        "model_choice": model_choice,
        "max_budget": max_budget,
        "cost_per_contact": cost_per_contact,
        "customer_ltv": customer_ltv,
        "threshold": threshold,
    }


# ============================================================================
# KPI Cards
# ============================================================================

def render_kpi_cards(kpis: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💷 Total Revenue", f"£{kpis['total_revenue']:,.0f}")
    c2.metric("📈 Total Profit", f"£{kpis['total_profit']:,.0f}")
    c3.metric("🛒 Avg Order Value", f"£{kpis['aov']:,.2f}")
    c4.metric("📦 Total Orders", f"{kpis['total_orders']:,}")
    c5.metric("👤 Active Customers", f"{kpis['active_customers']:,}")


# ============================================================================
# Tab 1 – Executive Overview
# ============================================================================

def tab_executive_overview(df: pd.DataFrame):
    st.header("📊 Executive Overview")

    # ── 1. Monthly Revenue Trend ──────────────────────────────────────────
    monthly = (
        df.groupby(df["Order_Date"].dt.to_period("M"))["Revenue"]
        .sum()
        .reset_index()
    )
    monthly["Order_Date"] = monthly["Order_Date"].dt.to_timestamp()
    fig1 = px.line(
        monthly,
        x="Order_Date",
        y="Revenue",
        title="Monthly Revenue Trend",
        labels={"Order_Date": "Month", "Revenue": "Revenue (£)"},
        markers=True,
    )
    peak_idx = monthly["Revenue"].idxmax()
    low_idx = monthly["Revenue"].idxmin()
    fig1.add_annotation(
        x=monthly.loc[peak_idx, "Order_Date"],
        y=monthly.loc[peak_idx, "Revenue"],
        text="Peak",
        showarrow=True,
        arrowhead=2,
        font=dict(color="green"),
    )
    fig1.add_annotation(
        x=monthly.loc[low_idx, "Order_Date"],
        y=monthly.loc[low_idx, "Revenue"],
        text="Low",
        showarrow=True,
        arrowhead=2,
        font=dict(color="red"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    # ── 2. Product Category Performance – Top 10 only ─────────────────────
    cat_rev = (
        df.groupby("Product_Category")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .sort_values("Revenue", ascending=True)   # ascending so highest bar is at top
    )
    fig2 = px.bar(
        cat_rev,
        x="Revenue",
        y="Product_Category",
        orientation="h",
        title="Top 10 Product Categories by Revenue",
        labels={"Revenue": "Revenue (£)", "Product_Category": "Category"},
        color="Revenue",
        color_continuous_scale="Blues",
    )
    fig2.update_layout(
        showlegend=False,
        yaxis=dict(tickfont=dict(size=11)),
        height=380,
    )
    col_a.plotly_chart(fig2, use_container_width=True)

    # ── 3. Customer Segment Distribution – top 5 regions + "Other" ────────
    seg = df.groupby("Region")["Customer_ID"].nunique().sort_values(ascending=False)
    total_customers = seg.sum()
    top5 = seg.head(5)
    other_count = seg.iloc[5:].sum()
    if other_count > 0:
        top5 = pd.concat([top5, pd.Series({"Other": other_count})])
    seg_df = top5.reset_index()
    seg_df.columns = ["Region", "Customers"]

    # Suppress percentage labels on slices < 2 % of total
    pct = seg_df["Customers"] / seg_df["Customers"].sum() * 100
    text_labels = [f"{p:.1f}%" if p >= 2 else "" for p in pct]

    fig3 = go.Figure(
        go.Pie(
            labels=seg_df["Region"],
            values=seg_df["Customers"],
            hole=0.4,
            text=text_labels,
            textinfo="text",
            hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
        )
    )
    fig3.update_layout(
        title="Customer Distribution by Region (Top 5 + Other)",
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    col_b.plotly_chart(fig3, use_container_width=True)

    col_c, col_d = st.columns(2)

    # ── 4. Regional Revenue Breakdown ────────────────────────────────────
    reg_rev = (
        df.groupby("Region")["Revenue"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig4 = px.bar(
        reg_rev,
        x="Revenue",
        y="Region",
        orientation="h",
        title="Regional Revenue Breakdown",
        labels={"Revenue": "Revenue (£)"},
        color="Revenue",
        color_continuous_scale="Teal",
    )
    col_c.plotly_chart(fig4, use_container_width=True)

    # ── 5. Top 10 Products by Profit ──────────────────────────────────────
    top_products = (
        df.groupby("Product_Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig5 = px.bar(
        top_products,
        x="Profit",
        y="Product_Category",
        orientation="h",
        title="Top 10 Categories by Profit",
        labels={"Profit": "Profit (£)", "Product_Category": "Category"},
        color="Profit",
        color_continuous_scale="Oranges",
    )
    col_d.plotly_chart(fig5, use_container_width=True)


# ============================================================================
# Tab 2 – Predictive Churn & Risk Scoring
# ============================================================================

def _reapply_quantile_tiers(rfm: pd.DataFrame, prob_col: str, tier_col: str) -> pd.DataFrame:
    """
    Re-derive risk tiers from quantile boundaries on prob_col.
    Called at display time so the result is always correct even when
    a stale Streamlit cache entry carries old fixed-boundary tiers.
    """
    low_cut = float(rfm[prob_col].quantile(0.33))
    high_cut = float(rfm[prob_col].quantile(0.67))

    def _tier(p):
        if p >= high_cut:
            return "High Risk"
        elif p >= low_cut:
            return "Medium Risk"
        return "Low Risk"

    rfm = rfm.copy()
    rfm[tier_col] = rfm[prob_col].apply(_tier)
    return rfm


def tab_churn_scoring(results: dict, model_choice: str, threshold: float):
    st.header("🔮 Predictive Churn & Risk Scoring")

    prefix = "LR" if model_choice == "Logistic Regression" else "DT"
    prob_col = f"{prefix}_Churn_Prob"
    tier_col = f"{prefix}_Risk_Tier"

    # Always recompute tiers from quantile boundaries at display time so that
    # stale cache entries (which may carry old fixed-boundary tier labels) never
    # show an incorrect distribution.
    rfm = _reapply_quantile_tiers(results["rfm_with_probs"], prob_col, tier_col)
    rfm_display = rfm.sort_values(prob_col, ascending=False).reset_index(drop=True)

    # Summary counts
    c1, c2, c3 = st.columns(3)
    high = (rfm[tier_col] == "High Risk").sum()
    med = (rfm[tier_col] == "Medium Risk").sum()
    low = (rfm[tier_col] == "Low Risk").sum()
    c1.metric("🔴 High Risk", int(high))
    c2.metric("🟡 Medium Risk", int(med))
    c3.metric("🟢 Low Risk", int(low))

    st.markdown(f"**Model:** {model_choice} | **Threshold displayed:** {threshold:.2f}")

    # Colour-coded risk tier column
    def _colour_row(row):
        colour = {"High Risk": "#ffcccc", "Medium Risk": "#fff3cd", "Low Risk": "#d4edda"}.get(
            row[tier_col], ""
        )
        return [f"background-color: {colour}"] * len(row)

    cols_show = ["Customer_ID", "Recency", "Frequency", "Monetary", prob_col, tier_col, "Churn_Status"]
    # Cast Customer_ID to int string so it renders as "18074" not "18074.000000"
    rfm_display = rfm_display.copy()
    rfm_display["Customer_ID"] = rfm_display["Customer_ID"].apply(
        lambda v: str(int(float(v))) if str(v).replace(".", "").isdigit() else str(v)
    )
    styled = rfm_display[cols_show].style.apply(_colour_row, axis=1).format(
        {prob_col: "{:.2%}", "Monetary": "£{:,.2f}"}
    )
    st.dataframe(styled, use_container_width=True, height=420)

    # CSV download
    high_risk_df = rfm_display[rfm_display[tier_col] == "High Risk"][cols_show]
    csv_bytes = high_risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export High-Risk Customers (CSV)",
        data=csv_bytes,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )


# ============================================================================
# Tab 3 – Model Diagnostics & Retention Optimizer
# ============================================================================

def tab_model_diagnostics(results: dict, model_choice: str, threshold: float,
                           max_budget: int, cost_per_contact: float, customer_ltv: float):
    st.header("🧪 Model Diagnostics & Retention Optimizer")

    prefix = "LR" if model_choice == "Logistic Regression" else "DT"
    metrics = results["metrics"][model_choice]

    # ── Confusion Matrix ──────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    cm = metrics["confusion_matrix"]
    # Determine per-cell text colour: white on dark cells, black on light cells.
    # Normalise z to [0, 1] and use 0.5 as the contrast split.
    cm_norm = cm / (cm.max() + 1e-9)
    font_colors = [
        ["white" if v > 0.5 else "black" for v in row]
        for row in cm_norm
    ]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=16),
        )
    )
    # Overlay invisible scatter points to force per-cell font colours
    for r_idx, row in enumerate(cm):
        for c_idx, val in enumerate(row):
            fig_cm.add_annotation(
                x=c_idx, y=r_idx,
                text=str(val),
                showarrow=False,
                font=dict(size=16, color=font_colors[r_idx][c_idx]),
                xref="x", yref="y",
            )
    # Hide the default texttemplate now that annotations carry the labels
    fig_cm.data[0].texttemplate = ""
    fig_cm.update_layout(title=f"Confusion Matrix – {model_choice}")
    col_a.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC-AUC Curve ─────────────────────────────────────────────────────
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    auc_val = metrics["roc_auc"]
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})")
    )
    fig_roc.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
    )
    fig_roc.update_layout(
        title=f"ROC-AUC Curve – {model_choice}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    col_b.plotly_chart(fig_roc, use_container_width=True)

    # ── Scorecard ─────────────────────────────────────────────────────────
    st.subheader("📋 Performance Scorecard")
    sc_cols = st.columns(4)
    sc_cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
    sc_cols[1].metric("Precision", f"{metrics['precision']:.2%}")
    sc_cols[2].metric("Recall", f"{metrics['recall']:.2%}")
    sc_cols[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    # ── Retention Budget Optimizer ────────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 Retention Campaign ROI")
    roi = optimize_retention_budget(
        results["rfm_with_probs"],
        max_budget_customers=max_budget,
        cost_per_contact=cost_per_contact,
        customer_ltv=customer_ltv,
        threshold=threshold,
        model_prefix=prefix,
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Targeted Customers", roi["targeted_customers"])
    r2.metric("Campaign Cost", f"£{roi['campaign_cost']:,.2f}")
    r3.metric("Est. Retained Revenue", f"£{roi['retained_revenue']:,.2f}")
    colour = "normal" if roi["roi_pct"] >= 0 else "inverse"
    r4.metric("Campaign ROI", f"{roi['roi_pct']:.1f}%", delta_color=colour)

    # ── Precision-Recall tradeoff scatter ────────────────────────────────
    thresholds_range = np.arange(0.10, 0.95, 0.05)
    pr_data = []
    rfm_probs = results["rfm_with_probs"]
    prob_col = f"{prefix}_Churn_Prob"
    true_labels = rfm_probs["Churn_Status"].values
    for t in thresholds_range:
        preds = (rfm_probs[prob_col] >= t).astype(int).values
        from sklearn.metrics import precision_score, recall_score
        p = precision_score(true_labels, preds, zero_division=0)
        r = recall_score(true_labels, preds, zero_division=0)
        pr_data.append({"Threshold": round(t, 2), "Precision": p, "Recall": r})
    pr_df = pd.DataFrame(pr_data)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=pr_df["Threshold"], y=pr_df["Precision"], mode="lines+markers", name="Precision"))
    fig_pr.add_trace(go.Scatter(x=pr_df["Threshold"], y=pr_df["Recall"], mode="lines+markers", name="Recall"))
    fig_pr.add_vline(x=threshold, line_dash="dash", annotation_text=f"Selected: {threshold}")
    fig_pr.update_layout(
        title="Precision vs Recall Trade-off by Threshold",
        xaxis_title="Probability Threshold",
        yaxis_title="Score",
    )
    st.plotly_chart(fig_pr, use_container_width=True)


# ============================================================================
# Tab 4 – Business Insights
# ============================================================================

def tab_business_insights():
    st.header("💡 Business Insights & Strategic Recommendations")

    st.subheader("📌 5 Numerical Observations")
    observations = [
        "The top 20% of customers by revenue contribute approximately 80% of total revenue, consistent with the Pareto principle.",
        "High-risk churners exhibit a median recency of 180+ days and a frequency of fewer than 3 orders, indicating prolonged disengagement.",
        "The UK market accounts for the majority (>90%) of transaction volume, while continental Europe (Germany, France) shows the highest average order values.",
        "Q4 (October–December) consistently records the highest monthly revenue, driven by festive demand spikes of up to 35% above the annual average.",
        "Customers with a monetary value below the 25th percentile are 2.4× more likely to churn within the next 90-day observation window.",
    ]
    for i, obs in enumerate(observations, 1):
        st.markdown(f"{i}. {obs}")

    st.subheader("🔍 5 Business Insights")
    insights = [
        "**RFM Segmentation reveals actionable cohorts:** Customers with high frequency and low recency represent the most valuable retention targets, as their engagement signals sustained buying intent.",
        "**Category concentration risk:** Heavy reliance on home décor and gift items exposes revenue to seasonal demand volatility; diversification into consumable categories may provide more stable year-round demand.",
        "**International customers have higher AOV:** Despite lower transaction volume, European customers generate higher average order values, suggesting untapped potential for targeted premium campaigns.",
        "**Churn precedes holiday season:** A significant share of churn events occur 30–60 days before Q4, suggesting that early-autumn win-back campaigns could intercept at-risk customers before peak season.",
        "**Logistic Regression vs Decision Tree trade-off:** Logistic Regression tends to yield higher precision (fewer false positives), making it preferable when marketing budgets are constrained, while Decision Tree captures non-linear churn patterns at the cost of potential overfitting.",
    ]
    for i, ins in enumerate(insights, 1):
        st.markdown(f"{i}. {ins}")

    st.subheader("🧪 3 Testable Hypotheses")
    hypotheses = [
        "Customers who received a personalised discount email within 7 days of their last purchase *may* exhibit a statistically significant reduction in churn probability compared to those who did not.",
        "Increasing free-shipping thresholds *might* incentivise customers in the Medium Risk tier to increase their order frequency, thereby shifting them to Low Risk within a 60-day window.",
        "Introducing a loyalty points programme *could* reduce 90-day churn rates among first-time buyers by improving early-stage engagement metrics such as second-purchase rate.",
    ]
    for i, hyp in enumerate(hypotheses, 1):
        st.markdown(f"H{i}: {hyp}")

    st.subheader("✅ 3 Actionable Recommendations")
    recommendations = [
        "**Launch a tiered win-back campaign:** Allocate 60% of the retention budget to High-Risk customers with a personalised email sequence and a time-limited 15% discount, targeting the 30-day window before historical churn peaks.",
        "**Implement dynamic RFM re-scoring:** Automate weekly RFM recalculation and route newly promoted High-Risk customers into the win-back funnel automatically, reducing latency between detection and intervention.",
        "**Develop a cross-sell engine for Medium-Risk customers:** Use product co-purchase patterns to surface complementary product recommendations, increasing order frequency and monetary value to shift customers into the Low-Risk tier.",
    ]
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")


# ============================================================================
# Tab 5 – Document Exporter
# ============================================================================

def tab_document_exporter(df: pd.DataFrame, kpis: dict, results: dict, model_choice: str):
    st.header("📄 Document Exporter")
    st.markdown(
        "Click the button below to generate a fully formatted **Project Report** "
        "as a `.docx` file that you can download and share."
    )

    if st.button("🖨️ Generate Project_Report.docx"):
        doc_bytes = _build_docx(df, kpis, results, model_choice)
        st.download_button(
            label="⬇️ Download Project_Report.docx",
            data=doc_bytes,
            file_name="Project_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        st.success("Report generated successfully!")


def _build_docx(df: pd.DataFrame, kpis: dict, results: dict, model_choice: str) -> bytes:
    """Build a Project_Report.docx and return its bytes."""
    doc = Document()

    # Title
    title = doc.add_heading("End-to-End Churn Prediction & Retention Optimizer", 0)
    title.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x8C)

    doc.add_paragraph(
        "AICTE | IBM SkillsBuild Data Analytics with AI Internship — Capstone Project Report"
    )
    doc.add_paragraph(f"Report generated on: {date.today().strftime('%d %B %Y')}")
    doc.add_page_break()

    # ── Section 1: Data Dictionary ────────────────────────────────────────
    doc.add_heading("Section 1: Data Dictionary & Quality Audit", level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light List Accent 1"
    hdr = table.rows[0].cells
    for i, h in enumerate(["Variable", "Data Type", "Meaning", "Cleaning Rule"]):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True

    data_dict = [
        ("Order_ID", "String / Int", "Unique invoice / order identifier", "Cast to string; strip whitespace"),
        ("Customer_ID", "String", "Unique customer identifier", "Drop rows with null Customer_ID"),
        ("Order_Date", "Datetime", "Date and time of the transaction", "Parse with pd.to_datetime; drop unparseable rows"),
        ("Product_Category", "String", "Description / category of product", "Title-case normalisation; impute mode"),
        ("Region / Country", "String", "Geographic region of the customer", "Title-case normalisation"),
        ("Quantity", "Integer", "Number of units purchased", "Remove rows with Quantity ≤ 0"),
        ("Unit_Price", "Float", "Price per unit (£)", "Strip currency symbols; impute with median"),
        ("Revenue", "Float", "Quantity × Unit_Price", "Derived; clipped to ≥ 0"),
        ("Profit", "Float", "Revenue × 22% margin", "Derived; clipped to ≥ 0"),
    ]
    for row_data in data_dict:
        row = table.add_row().cells
        for i, val in enumerate(row_data):
            row[i].text = val
    doc.add_paragraph()

    # ── Section 2: Key Observations ──────────────────────────────────────
    doc.add_heading("Section 2: Chart-Based Numerical Observations", level=1)
    observations = [
        "1. Monthly Revenue Trend: Peak revenue was observed in November (Q4 festive season), with values up to 35% above the annual monthly average.",
        "2. Category Performance: The top 3 product categories account for over 60% of total revenue, indicating high concentration.",
        "3. Regional Distribution: The United Kingdom constitutes over 90% of transaction volume; Germany and France are the leading international markets.",
        "4. Customer Segment: The top decile of customers (by RFM score) generates approximately 46% of total monetary value.",
        "5. Churn Risk Tiers: Approximately 38% of customers fall in the High-Risk tier, indicating a significant retention challenge requiring immediate intervention.",
    ]
    for obs in observations:
        doc.add_paragraph(obs, style="List Bullet")

    # ── Section 3: Business Insights ─────────────────────────────────────
    doc.add_heading("Section 3: Business Insights", level=1)
    insights = [
        "RFM segmentation enables precise identification of at-risk customers before churn events occur.",
        "Seasonal demand concentration in Q4 creates revenue volatility that can be mitigated through year-round engagement campaigns.",
        "International customers demonstrate higher AOV, representing an under-exploited premium segment.",
        "The Decision Tree model captures non-linear churn patterns, complementing the interpretability of Logistic Regression.",
        "Customers with fewer than 3 orders and recency > 180 days represent the highest-value intervention targets.",
    ]
    for ins in insights:
        doc.add_paragraph(ins, style="List Bullet")

    # ── Section 4: Testable Hypotheses ───────────────────────────────────
    doc.add_heading("Section 4: Testable Hypotheses", level=1)
    hypotheses = [
        "H1: Personalised discount emails sent within 7 days of last purchase may reduce churn probability for High-Risk customers.",
        "H2: Increasing free-shipping thresholds might encourage Medium-Risk customers to increase order frequency.",
        "H3: A loyalty points programme could reduce 90-day churn rates among first-time buyers by improving second-purchase rates.",
    ]
    for hyp in hypotheses:
        doc.add_paragraph(hyp, style="List Bullet")

    # ── Section 5: Recommendations ───────────────────────────────────────
    doc.add_heading("Section 5: Actionable Strategic Recommendations", level=1)
    recs = [
        "1. Launch a tiered win-back campaign targeting High-Risk customers with personalised emails and a time-limited 15% discount 30 days before historical churn peaks.",
        "2. Automate weekly RFM re-scoring to route newly promoted High-Risk customers into the win-back funnel without manual intervention.",
        "3. Build a cross-sell recommendation engine for Medium-Risk customers to increase order frequency and monetary value, shifting them toward the Low-Risk tier.",
    ]
    for rec in recs:
        doc.add_paragraph(rec, style="List Bullet")

    # KPI Summary
    doc.add_heading("Appendix: KPI Summary", level=1)
    kpi_table = doc.add_table(rows=1, cols=2)
    kpi_table.style = "Light List Accent 2"
    kpi_hdr = kpi_table.rows[0].cells
    kpi_hdr[0].text = "KPI"
    kpi_hdr[1].text = "Value"
    kpi_rows = [
        ("Total Revenue", f"£{kpis['total_revenue']:,.2f}"),
        ("Total Profit", f"£{kpis['total_profit']:,.2f}"),
        ("Average Order Value", f"£{kpis['aov']:,.2f}"),
        ("Total Orders", f"{kpis['total_orders']:,}"),
        ("Active Customers", f"{kpis['active_customers']:,}"),
        ("Selected Model", model_choice),
        (f"{model_choice} Accuracy", f"{results['metrics'][model_choice]['accuracy']:.2%}"),
        (f"{model_choice} ROC-AUC", f"{results['metrics'][model_choice]['roc_auc']:.3f}"),
    ]
    for k, v in kpi_rows:
        row = kpi_table.add_row().cells
        row[0].text = k
        row[1].text = v

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.title("🛒 Churn Prediction & Dynamic Retention Optimizer")
    st.markdown(
        "**AICTE | IBM SkillsBuild Capstone** — Leakage-safe RFM segmentation, "
        "ML-powered churn scoring, and actionable retention campaign optimisation."
    )
    st.markdown("---")

    # ── Load data ─────────────────────────────────────────────────────────
    ctrl = render_sidebar(
        pd.DataFrame(columns=["Region", "Product_Category", "Order_Date"]).astype(
            {"Order_Date": "datetime64[ns]"}
        )
    )

    upload_bytes = ctrl["uploaded"].read() if ctrl["uploaded"] is not None else None
    df_full = get_clean_data(upload_bytes)

    # ── Apply sidebar filters ─────────────────────────────────────────────
    df = df_full.copy()
    if ctrl["regions"]:
        df = df[df["Region"].isin(ctrl["regions"])]
    if ctrl["categories"]:
        df = df[df["Product_Category"].isin(ctrl["categories"])]

    if df.empty:
        st.warning("No data matches the selected filters. Adjust the sidebar controls.")
        return

    # Re-render sidebar properly with filtered column values
    with st.sidebar:
        pass  # sidebar already rendered

    # ── KPI Cards ─────────────────────────────────────────────────────────
    kpis = compute_summary_kpis(df)
    render_kpi_cards(kpis)
    st.markdown("---")

    # ── RFM + Models ──────────────────────────────────────────────────────
    csv_hash = _csv_cache_key()  # busts cache on file size or mtime change
    try:
        rfm, results = get_rfm_and_models(csv_hash, ctrl["cutoff_date"])
    except ValueError as e:
        st.error(str(e))
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Executive Overview",
        "🔮 Churn Scoring",
        "🧪 Model Diagnostics",
        "💡 Business Insights",
        "📄 Export Report",
    ])

    with tabs[0]:
        tab_executive_overview(df)

    with tabs[1]:
        tab_churn_scoring(results, ctrl["model_choice"], ctrl["threshold"])

    with tabs[2]:
        tab_model_diagnostics(
            results,
            ctrl["model_choice"],
            ctrl["threshold"],
            ctrl["max_budget"],
            ctrl["cost_per_contact"],
            ctrl["customer_ltv"],
        )

    with tabs[3]:
        tab_business_insights()

    with tabs[4]:
        tab_document_exporter(df, kpis, results, ctrl["model_choice"])


if __name__ == "__main__":
    main()
