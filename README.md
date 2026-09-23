# 🛒 End-to-End Leakage-Safe Churn Prediction & Dynamic Retention Optimizer

> **AICTE | IBM SkillsBuild Data Analytics with AI Internship — Capstone Project**

---

## Abstract

This project delivers a production-ready, end-to-end analytics and machine-learning pipeline that predicts customer churn from transactional e-commerce data and optimises retention campaign budgets in real time.  
A strict **temporal isolation** strategy prevents target leakage when computing RFM features, ensuring that the churn label (derived from a future observation window) never contaminates the historical feature window.  
The interactive **Streamlit dashboard** exposes the full pipeline — from raw data ingestion to model diagnostics and campaign ROI — through a browser-based UI requiring zero additional infrastructure.

---

## Problem Statement

Retail businesses accumulate large volumes of transaction data but struggle to translate that data into actionable retention strategies.  
The core challenge is two-fold:

1. **Predictive accuracy vs. leakage:** Naively computed RFM features often include information from the prediction period, inflating model performance metrics and producing unreliable real-world predictions.
2. **Budget allocation:** Even with a reliable churn model, marketing teams lack tools to dynamically balance outreach cost against expected retained revenue.

This project solves both challenges through a leakage-safe RFM methodology and a built-in retention budget optimiser.

---

## Directory Architecture

```
supermarket_analytics/
├── data/
│   ├── supermarket_sales_raw.csv   # Raw UK Online Retail dataset
│   └── clean_data.csv              # Auto-generated sanitised dataset
├── requirements.txt                # Python package dependencies
├── data_loader.py                  # Data ingestion & sanitisation engine
├── analytics.py                    # RFM, ML models, retention optimiser
├── app.py                          # Streamlit interactive dashboard
├── README.md                       # This file
└── Project_Report.docx             # Auto-generated Word report
```

---

## Local Installation & Setup

### Prerequisites
- Python 3.9 or later
- pip

### Steps

```bash
# 1. Clone or download the repository
cd supermarket_analytics

# 2. (Optional) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-generate the clean dataset
python data_loader.py

# 5. Launch the dashboard
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 **Executive Overview** | Monthly revenue trend, category performance, regional breakdown, customer segment pie, top-10 profit categories |
| 🔮 **Churn Scoring** | Colour-coded risk table (High / Medium / Low), sortable by churn probability; CSV export of high-risk list |
| 🧪 **Model Diagnostics** | Interactive confusion matrix, ROC-AUC curve, precision-recall trade-off, retention campaign ROI calculator |
| 💡 **Business Insights** | 5 observations, 5 insights, 3 testable hypotheses, 3 actionable recommendations |
| 📄 **Export Report** | One-click generation and download of `Project_Report.docx` |

---

## ML Pipeline

### Data Ingestion (`data_loader.py`)
1. Loads raw CSV (latin-1 encoded UK Online Retail dataset).  
2. Renames columns to canonical names (`Order_ID`, `Customer_ID`, `Order_Date`, `Product_Category`, `Region`, `Quantity`, `Unit_Price`, `Revenue`, `Profit`).  
3. Drops rows with missing `Customer_ID` and returns / negative quantities.  
4. Imputes numeric nulls with **median**, categorical nulls with **mode**.  
5. Normalises casing; strips currency symbols; clips revenue & profit ≥ 0.  
6. Derives `Revenue = Quantity × Unit_Price` and `Profit = Revenue × 22%` if absent.  
7. Writes sanitised output to `data/clean_data.csv`.

### Leakage-Safe RFM (`analytics.build_leakage_safe_rfm`)

```
Timeline ───────────────────────────────────────────────────────────────►
            Historical Window              │  Observation Window
         (features: R, F, M computed here) │  (Churn label defined here)
                                      cutoff_date
```

- **Historical Window** (`Order_Date < cutoff_date`): computes Recency (days since last purchase), Frequency (unique invoice count), Monetary (total spend).  
- **Observation Window** (`Order_Date ≥ cutoff_date`): a customer is labelled `Churn_Status = 1` if they placed **zero** orders in this window.  
- This strict split guarantees the model never sees future information during feature engineering.

### Churn Models (`analytics.train_churn_models`)
- **Features:** Recency, Frequency, Monetary  
- **Target:** Churn_Status (binary)  
- **Split:** 80 % train / 20 % test (stratified)  
- **Models:**
  - `LogisticRegression` — standardised features, max_iter=1000
  - `DecisionTreeClassifier` — max_depth=5
- **Evaluation:** Accuracy, Precision, Recall, ROC-AUC, Confusion Matrix, ROC Curve
- **Risk Tiers:** High (≥ 70 %), Medium (40–70 %), Low (< 40 %)

### Retention Budget Optimiser (`analytics.optimize_retention_budget`)
Filters customers by probability threshold, caps at the selected budget, and computes:
- **Campaign Cost** = targeted customers × cost per contact
- **Estimated Retained Revenue** = targeted customers × LTV × avg churn probability
- **ROI (%)** = (retained revenue − campaign cost) / campaign cost × 100

---

## Sidebar Controls

| Control | Purpose |
|---------|---------|
| Region / Category multi-select | Filter the data shown in Executive Overview KPIs |
| RFM Cutoff Date | Temporal split point for leakage-safe feature engineering |
| Model Toggle | Switch between Logistic Regression and Decision Tree |
| Max Target Customers | Upper bound on campaign reach |
| Cost per Contact | Per-customer marketing spend |
| Customer LTV | Expected lifetime value of a retained customer |
| Churn Probability Threshold | Minimum score to include a customer in a campaign |
| Upload CSV | Re-run entire pipeline on new raw data |

---

## License
This project was produced as part of the AICTE | IBM SkillsBuild internship programme and is intended for academic and educational use.
