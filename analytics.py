"""
analytics.py
------------
Leakage-safe RFM segmentation, churn prediction models, and retention
budget optimiser.

Public API
----------
compute_summary_kpis(df)
build_leakage_safe_rfm(df, cutoff_date)
train_churn_models(rfm_df)
optimize_retention_budget(rfm_with_probs, max_budget_customers,
                          cost_per_contact, customer_ltv, threshold)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ============================================================================
# 1. Summary KPIs
# ============================================================================

def compute_summary_kpis(df: pd.DataFrame) -> dict:
    """
    Compute headline KPIs from the cleaned transaction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data with columns Revenue, Profit, Order_ID,
        Customer_ID.

    Returns
    -------
    dict with keys:
        total_revenue, total_profit, aov, total_orders, active_customers
    """
    total_revenue = float(df["Revenue"].sum())
    total_profit = float(df["Profit"].sum())
    total_orders = int(df["Order_ID"].nunique())
    active_customers = int(df["Customer_ID"].nunique())
    aov = total_revenue / total_orders if total_orders > 0 else 0.0

    return {
        "total_revenue": round(total_revenue, 2),
        "total_profit": round(total_profit, 2),
        "aov": round(aov, 2),
        "total_orders": total_orders,
        "active_customers": active_customers,
    }


# ============================================================================
# 2. Leakage-Safe RFM
# ============================================================================

def build_leakage_safe_rfm(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """
    Build RFM features using strict temporal isolation to prevent target leakage.

    Temporal split:
      Historical window : Order_Date <  cutoff_date  → used to compute R, F, M
      Observation window: Order_Date >= cutoff_date  → used to define Churn_Status

    Churn_Status = 1 if the customer made zero purchases in the observation
    window, otherwise 0.  The recency / frequency / monetary features are
    computed *exclusively* from the historical window, guaranteeing no
    information from the observation window contaminates the features.

    Parameters
    ----------
    df          : cleaned transaction DataFrame
    cutoff_date : temporal split point (string or Timestamp)

    Returns
    -------
    pd.DataFrame with columns:
        Customer_ID, Recency, Frequency, Monetary, Churn_Status
    """
    cutoff = pd.Timestamp(cutoff_date)

    # ── Temporal split ────────────────────────────────────────────────────
    hist = df[df["Order_Date"] < cutoff].copy()
    obs = df[df["Order_Date"] >= cutoff].copy()

    if hist.empty:
        raise ValueError(
            f"No historical data before cutoff {cutoff_date}. "
            "Choose an earlier cutoff date."
        )

    # ── Historical RFM features ───────────────────────────────────────────
    # Recency: days since the customer's last purchase *before* the cutoff
    last_purchase = hist.groupby("Customer_ID")["Order_Date"].max()
    recency = (cutoff - last_purchase).dt.days

    frequency = hist.groupby("Customer_ID")["Order_ID"].nunique()
    monetary = hist.groupby("Customer_ID")["Revenue"].sum()

    rfm = pd.DataFrame(
        {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
        }
    ).reset_index()
    rfm.rename(columns={"Customer_ID": "Customer_ID"}, inplace=True)

    # ── Observation-window churn target ───────────────────────────────────
    # Customers who appear in the observation window are NOT churned (0).
    active_in_obs = set(obs["Customer_ID"].unique())
    rfm["Churn_Status"] = rfm["Customer_ID"].apply(
        lambda cid: 0 if cid in active_in_obs else 1
    )

    # ── Sanity clipping ───────────────────────────────────────────────────
    rfm["Recency"] = rfm["Recency"].clip(lower=0)
    rfm["Frequency"] = rfm["Frequency"].clip(lower=1)
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0)

    return rfm.reset_index(drop=True)


# ============================================================================
# 3. Churn Model Training
# ============================================================================

def train_churn_models(rfm_df: pd.DataFrame) -> dict:
    """
    Train Logistic Regression and Decision Tree classifiers on RFM features.

    Parameters
    ----------
    rfm_df : pd.DataFrame returned by build_leakage_safe_rfm()

    Returns
    -------
    dict with keys:
        models        : {"Logistic Regression": <fitted model>, "Decision Tree": <fitted model>}
        scaler        : fitted StandardScaler (for Logistic Regression pipeline)
        metrics       : dict of per-model accuracy, precision, recall, roc_auc, confusion_matrix
        roc_curves    : dict of per-model (fpr, tpr, thresholds)
        rfm_with_probs: original rfm_df augmented with columns
                        LR_Churn_Prob, DT_Churn_Prob, LR_Risk_Tier, DT_Risk_Tier
        X_test, y_test: held-out evaluation split
    """
    features = ["Recency", "Frequency", "Monetary"]
    target = "Churn_Status"

    X = rfm_df[features].values
    y = rfm_df[target].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale for Logistic Regression ─────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Logistic Regression ───────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    # ── Decision Tree ─────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    # ── Evaluation helper ─────────────────────────────────────────────────
    def _eval(model, X_scaled, X_raw, y_true):
        if isinstance(model, LogisticRegression):
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]
        else:
            y_pred = model.predict(X_raw)
            y_prob = model.predict_proba(X_raw)[:, 1]

        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": thresholds,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    lr_metrics = _eval(lr, X_test_s, X_test, y_test)
    dt_metrics = _eval(dt, X_test_s, X_test, y_test)

    # ── Full-dataset probabilities (for the risk table) ───────────────────
    all_X_s = scaler.transform(X)
    rfm_out = rfm_df.copy()
    rfm_out["LR_Churn_Prob"] = np.round(lr.predict_proba(all_X_s)[:, 1], 4)
    rfm_out["DT_Churn_Prob"] = np.round(dt.predict_proba(X)[:, 1], 4)

    for prefix in ("LR", "DT"):
        col = f"{prefix}_Churn_Prob"
        tier_col = f"{prefix}_Risk_Tier"
        # Use data-driven quantile boundaries so tiers are always spread across
        # all three bins, even when LR probabilities cluster near the base rate.
        low_cut = float(np.percentile(rfm_out[col], 33))
        high_cut = float(np.percentile(rfm_out[col], 67))
        rfm_out[tier_col] = rfm_out[col].apply(
            lambda p, lo=low_cut, hi=high_cut: _assign_risk_tier_quantile(p, lo, hi)
        )

    return {
        "models": {"Logistic Regression": lr, "Decision Tree": dt},
        "scaler": scaler,
        "metrics": {
            "Logistic Regression": lr_metrics,
            "Decision Tree": dt_metrics,
        },
        "rfm_with_probs": rfm_out,
        "X_test": X_test,
        "y_test": y_test,
    }


def _assign_risk_tier(prob: float) -> str:
    """Map a churn probability to fixed-boundary risk tier (kept for backward compat)."""
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def _assign_risk_tier_quantile(prob: float, low_cut: float, high_cut: float) -> str:
    """
    Assign risk tier using data-driven quantile boundaries.

    Bottom third  (prob <  low_cut)  → Low Risk
    Middle third  (low_cut <= prob < high_cut) → Medium Risk
    Top third     (prob >= high_cut) → High Risk

    This prevents clustering when model probabilities are compressed near the
    dataset churn rate (common with Logistic Regression on RFM features).
    """
    if prob >= high_cut:
        return "High Risk"
    elif prob >= low_cut:
        return "Medium Risk"
    else:
        return "Low Risk"


# ============================================================================
# 4. Retention Budget Optimiser
# ============================================================================

def optimize_retention_budget(
    rfm_with_probs: pd.DataFrame,
    max_budget_customers: int,
    cost_per_contact: float,
    customer_ltv: float,
    threshold: float,
    model_prefix: str = "LR",
) -> dict:
    """
    Dynamic retention campaign ROI calculator.

    Parameters
    ----------
    rfm_with_probs      : DataFrame from train_churn_models()
    max_budget_customers: maximum number of customers to target
    cost_per_contact    : marketing spend per customer (currency units)
    customer_ltv        : expected lifetime value of a retained customer
    threshold           : minimum churn probability to include a customer
    model_prefix        : "LR" or "DT"

    Returns
    -------
    dict with keys:
        targeted_customers, campaign_cost, retained_revenue, roi_pct,
        threshold_used, targeted_df
    """
    prob_col = f"{model_prefix}_Churn_Prob"

    targeted = (
        rfm_with_probs[rfm_with_probs[prob_col] >= threshold]
        .sort_values(prob_col, ascending=False)
        .head(max_budget_customers)
        .copy()
    )

    n = len(targeted)
    campaign_cost = round(n * cost_per_contact, 2)
    # Assume avg churn probability represents the chance of preventing churn
    avg_prob = targeted[prob_col].mean() if n > 0 else 0.0
    retained_revenue = round(n * customer_ltv * avg_prob, 2)
    roi_pct = (
        round((retained_revenue - campaign_cost) / campaign_cost * 100, 2)
        if campaign_cost > 0
        else 0.0
    )

    return {
        "targeted_customers": n,
        "campaign_cost": campaign_cost,
        "retained_revenue": retained_revenue,
        "roi_pct": roi_pct,
        "threshold_used": threshold,
        "targeted_df": targeted,
    }
