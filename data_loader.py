"""
data_loader.py
--------------
Data ingestion and sanitisation engine.

The raw file is a UK Online Retail dataset with columns:
  Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
  Customer ID, Country

This module normalises those columns, derives Revenue/Profit, imputes
missing values, and writes data/clean_data.csv.
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime


RAW_PATH = os.path.join("data", "supermarket_sales_raw.csv")
CLEAN_PATH = os.path.join("data", "clean_data.csv")

# Approximate cost-of-goods ratio used to synthesise Profit
PROFIT_MARGIN = 0.22


# ---------------------------------------------------------------------------
# Synthetic dataset fallback
# ---------------------------------------------------------------------------
def _generate_synthetic_data() -> pd.DataFrame:
    """Return a 2 000-row realistic e-commerce transaction dataset."""
    rng = np.random.default_rng(42)
    n = 2_000
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    regions = ["North", "South", "East", "West", "Central"]

    order_dates = pd.to_datetime(
        rng.integers(
            int(pd.Timestamp("2022-01-01").timestamp()),
            int(pd.Timestamp("2023-12-31").timestamp()),
            n,
        ),
        unit="s",
    ).normalize()

    qty = rng.integers(1, 20, n)
    unit_price = np.round(rng.uniform(5, 500, n), 2)
    revenue = np.round(qty * unit_price, 2)
    profit = np.round(revenue * PROFIT_MARGIN, 2)

    df = pd.DataFrame(
        {
            "Order_ID": [f"ORD{100000 + i}" for i in range(n)],
            "Customer_ID": [f"CUST{rng.integers(1000, 3000)}" for _ in range(n)],
            "Order_Date": order_dates,
            "Product_Category": rng.choice(categories, n),
            "Region": rng.choice(regions, n),
            "Quantity": qty,
            "Unit_Price": unit_price,
            "Revenue": revenue,
            "Profit": profit,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Column-name normalisation helpers
# ---------------------------------------------------------------------------
def _strip_currency(series: pd.Series) -> pd.Series:
    """Remove currency symbols and convert to float."""
    return (
        series.astype(str)
        .str.replace(r"[₹$£€INR,\s]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def _title_case_series(series: pd.Series) -> pd.Series:
    """Trim whitespace and apply title-case normalisation."""
    return series.astype(str).str.strip().str.title()


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def load_and_clean_data(file_path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load raw transaction data, sanitise it, and return a clean DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV (defaults to data/supermarket_sales_raw.csv).

    Returns
    -------
    pd.DataFrame
        Cleaned, enriched DataFrame also written to data/clean_data.csv.
    """
    # ── 1. Load or synthesise ──────────────────────────────────────────────
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
    else:
        print("[data_loader] Raw file not found – generating synthetic dataset.")
        df = _generate_synthetic_data()
        os.makedirs("data", exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        df.to_csv(CLEAN_PATH, index=False)
        return df

    # ── 2. Rename raw UK Online-Retail columns → canonical names ──────────
    rename_map = {
        "Invoice": "Order_ID",
        "StockCode": "Product_Code",
        "Description": "Product_Category",
        "Quantity": "Quantity",
        "InvoiceDate": "Order_Date",
        "Price": "Unit_Price",
        "Customer ID": "Customer_ID",
        "Country": "Region",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ── 3. Drop rows with no Customer_ID (anonymous transactions) ─────────
    df.dropna(subset=["Customer_ID"], inplace=True)
    df["Customer_ID"] = df["Customer_ID"].astype(str).str.strip()

    # ── 4. Parse Order_Date ───────────────────────────────────────────────
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="mixed", dayfirst=False, errors="coerce")
    df.dropna(subset=["Order_Date"], inplace=True)

    # ── 5. Remove returns / negative quantities ───────────────────────────
    df = df[df["Quantity"] > 0].copy()

    # ── 6. Strip currency symbols from monetary columns ───────────────────
    for col in ["Unit_Price", "Revenue", "Profit"]:
        if col in df.columns:
            df[col] = _strip_currency(df[col])

    # Convert Unit_Price to float if it came in as numeric
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

    # ── 7. Impute missing numeric values with median ──────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # ── 8. Impute missing categorical values with mode ───────────────────
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)

    # ── 9. Normalise casing on string columns ─────────────────────────────
    for col in ["Product_Category", "Region"]:
        if col in df.columns:
            df[col] = _title_case_series(df[col])

    # ── 10. Derive Revenue and Profit if not already present ─────────────
    if "Revenue" not in df.columns:
        df["Revenue"] = np.round(df["Quantity"] * df["Unit_Price"], 2)
    if "Profit" not in df.columns:
        df["Profit"] = np.round(df["Revenue"] * PROFIT_MARGIN, 2)

    # ── 11. Ensure Revenue / Profit non-negative ─────────────────────────
    df["Revenue"] = df["Revenue"].clip(lower=0)
    df["Profit"] = df["Profit"].clip(lower=0)

    # ── 12. Reset index ───────────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)

    # ── 13. Persist clean file ────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[data_loader] Clean data saved -> {CLEAN_PATH}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    cleaned = load_and_clean_data()
    print(cleaned.head())
    print(cleaned.dtypes)
