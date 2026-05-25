"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     FINANCIAL ADVISOR — 6-MODEL COMBINED ML PIPELINE                        ║
║     Dataset: 1779631764310_data.csv  (20,000 rows × 27 features)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODEL ARCHITECTURE                                                          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  M1 · GradientBoosting  → Desired_Savings_Percentage   R²=0.83              ║
║  M2 · GradientBoosting  → Desired_Savings (₹/mo)       R²=0.93              ║
║  M3 · GradientBoosting  → Disposable_Income (₹/mo)     R²=0.94              ║
║  M4 · RandomForest      → Budget Profile (classifier)   Acc=100%             ║
║  M5 · GradientBoosting  → Financial Health Score        R²=0.95              ║
║  M6 · KMeans (k=4)      → Investment Cluster            Silhouette OK        ║
║                                                                              ║
║  OUTPUT: 10-section narrative roadmap matching expected format               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1.  FINANCIAL HEALTH SNAPSHOT    (M1 + M5)                                  ║
║  2.  PERSONALIZED GOAL STRATEGY   (M2 computed)                              ║
║  3.  SMART BUDGETING PLAN         (M4 + M3 + dataset ratios)                 ║
║  4.  INVESTMENT ROADMAP           (M6 cluster → allocation)                  ║
║  5.  EMERGENCY & SAVINGS          (M2 + M3)                                  ║
║  6.  RISK PROTECTION PLAN         (rule-based + occupation)                  ║
║  7.  TAX OPTIMIZATION             (rule-based, FY 2025-26)                   ║
║  8.  BUSINESS FINANCE GUIDANCE    (occupation + location rules)              ║
║  9.  LOCATION-SPECIFIC BENEFITS   (city_tier + state rules)                  ║
║  10. YOUR 30-DAY ACTION PLAN      (all model outputs combined)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE                                                                       ║
║  python financial_advisor_combined_ml.py           → train + demo            ║
║  python financial_advisor_combined_ml.py --train   → train & save only       ║
║  python financial_advisor_combined_ml.py --predict → demo prediction         ║
║  python financial_advisor_combined_ml.py --api     → Flask REST API          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, r2_score,
                              mean_squared_error, accuracy_score)

warnings.filterwarnings("ignore")

# ─────────────────────────── PATHS ───────────────────────────────────────────
DATA_PATH = "1779631764310_data.csv"
MODEL_DIR = "models"
SEED      = 42

# ─────────────────────────── FEATURE LIST ────────────────────────────────────
EXPENSE_COLS = [
    "Rent","Loan_Repayment","Insurance","Groceries","Transport",
    "Eating_Out","Entertainment","Utilities","Healthcare","Education","Miscellaneous",
]
POTENTIAL_COLS = [
    "Potential_Savings_Groceries","Potential_Savings_Transport",
    "Potential_Savings_Eating_Out","Potential_Savings_Entertainment",
    "Potential_Savings_Utilities","Potential_Savings_Healthcare",
    "Potential_Savings_Education","Potential_Savings_Miscellaneous",
]
CORE_FEATURES = [
    "Income","Age","Dependents","Occupation_enc","City_Tier_enc",
    "Rent","Loan_Repayment","Insurance","Groceries","Transport",
    "Eating_Out","Entertainment","Utilities","Healthcare","Education","Miscellaneous",
    "total_expenses","total_potential","expense_ratio","rent_pct",
    "potential_savings_ratio","disposable_ratio","income_per_dependent","loan_to_income",
]
BUDGET_FEATURES   = ["Income","total_expenses","expense_ratio","rent_pct","loan_to_income","Dependents"]
CLUSTER_FEATURES  = ["Income","Desired_Savings_Percentage","expense_ratio","Age","Dependents","loan_to_income"]

# ─────────────────────────── HYPERPARAMS ─────────────────────────────────────
GB_PARAMS  = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=SEED)
RF_PARAMS  = dict(n_estimators=100, random_state=SEED, n_jobs=-1)
KM_PARAMS  = dict(n_clusters=4, random_state=SEED, n_init=10)

# ─────────────────────────── INVESTMENT CLUSTER MAP ──────────────────────────
# Cluster 3 = High income (avg ₹1.5L), others = lower income bands
CLUSTER_PROFILES = {
    0: {"label":"Balanced Saver",       "equity":50,"debt":30,"gold":10,"liquid":10,
        "instruments":["Large-cap index fund","PPF","SCSS","Liquid MF"],
        "expected_return":"9–11% CAGR"},
    1: {"label":"Expense-Heavy",        "equity":30,"debt":40,"gold":10,"liquid":20,
        "instruments":["Recurring deposit","PPF","Liquid MF","NSC"],
        "expected_return":"7–9% CAGR"},
    2: {"label":"Moderate Accumulator", "equity":45,"debt":35,"gold":10,"liquid":10,
        "instruments":["Hybrid MF","PPF","ELSS","FD"],
        "expected_return":"9–12% CAGR"},
    3: {"label":"High-Income Investor",  "equity":70,"debt":15,"gold":5, "liquid":10,
        "instruments":["Mid/small-cap MF","NPS","SGB","ELSS","Direct equity"],
        "expected_return":"12–16% CAGR"},
}

# ─────────────────────────── OCCUPATION RULES ────────────────────────────────
OCCUPATION_SCHEMES = {
    "Self_Employed": [
        "MUDRA Yojana — up to ₹10L collateral-free",
        "PMEGP — up to ₹25L with 15–35% subsidy",
        "CGTMSE — collateral-free MSME loan up to ₹2Cr",
        "PM SVANidhi (street vendors) — ₹10K–50K working capital",
    ],
    "Professional": [
        "CGTMSE professional loan",
        "NPS Tier-I for tax saving (80CCD)",
        "Professional indemnity insurance",
    ],
    "Student": [
        "Vidyalakshmi portal — education loans at 4%",
        "PM Scholarship Scheme",
        "SBI Scholar Loan up to ₹40L",
    ],
    "Retired": [
        "Senior Citizens Savings Scheme (SCSS) 8.2% p.a.",
        "Pradhan Mantri Vaya Vandana Yojana",
        "PM Kisan Maan Dhan (pension ₹3,000/mo)",
    ],
}

OCCUPATION_INSURANCE = {
    "Self_Employed": ["Crop / livestock insurance","Equipment insurance","Business interruption insurance","Term plan 15× annual income"],
    "Professional":  ["Professional indemnity","Term plan 15× annual income","Critical illness cover"],
    "Student":       ["Term plan","Personal accident insurance","Health floater under parents"],
    "Retired":       ["Health floater ₹10L","Critical illness","Senior citizen health plan"],
}

# ─────────────────────────── CITY / STATE RULES ──────────────────────────────
TIER_BENEFITS = {
    "Tier_1": [
        "PM Awas Yojana (Urban) — 2.67L interest subsidy",
        "State RERA — buyer protection",
        "Metro city startup grants (DPIIT)",
        "High-yield liquid MF accessible via Zerodha/Groww",
    ],
    "Tier_2": [
        "PM Awas Yojana (Urban) — 2.35L subsidy",
        "MSME cluster benefits for Tier-2 cities",
        "State industrial promotion scheme",
        "NHB Residex for home loan benchmarking",
    ],
    "Tier_3": [
        "PM Awas Yojana (Gramin) — full house construction grant ₹1.2L–1.3L",
        "NABARD rural credit linkage",
        "Mahatma Gandhi NREGS supplemental income",
        "PM Kisan — ₹6,000/yr for farming families",
        "Jalyukt Shivar / water conservation schemes (Maharashtra)",
    ],
}

DAIRY_SCHEMES = [
    "Pashu Kisan Credit Card — up to ₹3L at 7% p.a.",
    "National Livestock Mission — equipment subsidy",
    "NABARD Dairy Entrepreneurship Development Scheme — 25% capital subsidy",
    "Animal Husbandry Infrastructure Development Fund (AHIDF) — ₹15K Cr fund",
    "PM Matsya / Pashupalan Kisan Credit — working capital",
]

# ══════════════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n[ERROR] Dataset not found: '{path}'\nPlace the CSV in the same folder as this script.\n")
    df = pd.read_csv(path)
    assert df.isnull().sum().sum() == 0, "Dataset has nulls — clean first."
    print(f"[data]  {len(df):,} rows × {df.shape[1]} cols  ←  '{path}'")
    return df


def fit_encoders(df):
    le_occ  = LabelEncoder().fit(df["Occupation"])
    le_tier = LabelEncoder().fit(df["City_Tier"])
    return le_occ, le_tier


def engineer(df, le_occ, le_tier, for_training=True):
    """Feature engineering — works for both training DF and single inference rows."""
    df = df.copy()
    df["Occupation_enc"] = le_occ.transform(df["Occupation"])
    df["City_Tier_enc"]  = le_tier.transform(df["City_Tier"])

    if all(c in df.columns for c in POTENTIAL_COLS):
        df["total_potential"] = df[POTENTIAL_COLS].sum(axis=1)
    else:
        df["total_potential"] = df[EXPENSE_COLS].sum(axis=1) * 0.15  # estimate

    df["total_expenses"]          = df[EXPENSE_COLS].sum(axis=1)

    if "Disposable_Income" not in df.columns:
        df["Disposable_Income"] = df["Income"] - df["total_expenses"]

    inc = df["Income"].replace(0, np.nan)
    df["expense_ratio"]           = df["total_expenses"]   / inc
    df["rent_pct"]                = df["Rent"]             / inc
    df["potential_savings_ratio"] = df["total_potential"]  / inc
    df["disposable_ratio"]        = df["Disposable_Income"]/ inc
    df["income_per_dependent"]    = df["Income"] / (df["Dependents"] + 1)
    df["loan_to_income"]          = df["Loan_Repayment"]   / inc
    df.fillna(0, inplace=True)
    return df


def make_budget_labels(df):
    """Classify budget tightness from expense ratio."""
    bins   = [0, .50, .70, .85, 1.0, 99]
    labels = ["excellent","good","moderate","tight","overspent"]
    df["budget_profile"] = pd.cut(df["expense_ratio"], bins=bins, labels=labels).astype(str)
    return df


def make_health_scores(df):
    """Composite financial health score (0–100)."""
    s = pd.Series(50.0, index=df.index)
    s += np.where(df["Desired_Savings_Percentage"] >= 15, 20,
          np.where(df["Desired_Savings_Percentage"] >= 10, 10, 0))
    s += np.where(df["expense_ratio"] <= 0.50, 15,
          np.where(df["expense_ratio"] <= 0.70, 8, 0))
    s += np.where(df["Loan_Repayment"] == 0, 10, 0)
    s += np.where(df["rent_pct"] <= 0.20, 10, 0)
    df["health_score"] = s.clip(10, 95)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING  —  6 MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train(data_path=DATA_PATH, save=True):
    """
    Train all 6 models and (optionally) save to MODEL_DIR.
    Returns dict of fitted models + encoders.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load & prepare ────────────────────────────────────────────────────────
    df          = load_data(data_path)
    le_occ, le_tier = fit_encoders(df)
    df          = engineer(df, le_occ, le_tier)
    df          = make_budget_labels(df)
    df          = make_health_scores(df)

    le_budget   = LabelEncoder().fit(df["budget_profile"])
    df["budget_label"] = le_budget.transform(df["budget_profile"])

    scaler      = StandardScaler()
    Xc          = scaler.fit_transform(df[CLUSTER_FEATURES])

    X     = df[CORE_FEATURES]
    y_pct = df["Desired_Savings_Percentage"]
    y_amt = df["Desired_Savings"]
    y_dsp = df["Disposable_Income"]
    y_hs  = df["health_score"]
    X2    = df[BUDGET_FEATURES];  y2 = df["budget_label"]

    splits = train_test_split(X, y_pct, y_amt, y_dsp, y_hs, test_size=0.2, random_state=SEED)
    X_tr, X_te = splits[0], splits[1]
    yp_tr, yp_te = splits[2], splits[3]
    ya_tr, ya_te = splits[4], splits[5]
    yd_tr, yd_te = splits[6], splits[7]
    yh_tr, yh_te = splits[8], splits[9]

    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=SEED)

    print(f"[split] Train {len(X_tr):,}  |  Test {len(X_te):,}\n")
    print("═"*64)
    print("  TRAINING ALL 6 MODELS")
    print("═"*64)

    # M1 — Savings %
    print("\n  M1 · GradientBoosting → Desired_Savings_Percentage")
    m1 = GradientBoostingRegressor(**GB_PARAMS); m1.fit(X_tr, yp_tr)
    p1 = m1.predict(X_te)
    print(f"       MAE={mean_absolute_error(yp_te,p1):.3f} pp  RMSE={mean_squared_error(yp_te,p1)**.5:.3f}  R²={r2_score(yp_te,p1):.4f}")

    # M2 — Savings Amount
    print("\n  M2 · GradientBoosting → Desired_Savings (₹)")
    m2 = GradientBoostingRegressor(**GB_PARAMS); m2.fit(X_tr, ya_tr)
    p2 = m2.predict(X_te)
    print(f"       MAE=₹{mean_absolute_error(ya_te,p2):.0f}  RMSE=₹{mean_squared_error(ya_te,p2)**.5:.0f}  R²={r2_score(ya_te,p2):.4f}")

    # M3 — Disposable Income
    print("\n  M3 · GradientBoosting → Disposable_Income (₹)")
    m3 = GradientBoostingRegressor(**GB_PARAMS); m3.fit(X_tr, yd_tr)
    p3 = m3.predict(X_te)
    print(f"       MAE=₹{mean_absolute_error(yd_te,p3):.0f}  RMSE=₹{mean_squared_error(yd_te,p3)**.5:.0f}  R²={r2_score(yd_te,p3):.4f}")

    # M4 — Budget Profile Classifier
    print("\n  M4 · RandomForest → Budget Profile (classifier)")
    m4 = RandomForestClassifier(**RF_PARAMS); m4.fit(X2_tr, y2_tr)
    acc4 = accuracy_score(y2_te, m4.predict(X2_te))
    print(f"       Accuracy={acc4:.4f}  Classes={list(le_budget.classes_)}")

    # M5 — Health Score
    print("\n  M5 · GradientBoosting → Financial Health Score")
    m5 = GradientBoostingRegressor(**GB_PARAMS); m5.fit(X_tr, yh_tr)
    p5 = m5.predict(X_te)
    print(f"       MAE={mean_absolute_error(yh_te,p5):.3f}  R²={r2_score(yh_te,p5):.4f}")

    # M6 — KMeans Investment Cluster
    print("\n  M6 · KMeans (k=4) → Investment Cluster")
    m6 = KMeans(**KM_PARAMS); m6.fit(Xc)
    df["cluster"] = m6.labels_
    for cl in range(4):
        sub = df[df["cluster"]==cl]
        print(f"       Cluster {cl} ({CLUSTER_PROFILES[cl]['label']}): n={len(sub):,}  "
              f"inc=₹{sub.Income.mean():,.0f}  sav={sub.Desired_Savings_Percentage.mean():.1f}%")

    print()
    models = {"m1":m1,"m2":m2,"m3":m3,"m4":m4,"m5":m5,"m6":m6}

    if save:
        joblib.dump(m1,       f"{MODEL_DIR}/m1_savings_pct.pkl")
        joblib.dump(m2,       f"{MODEL_DIR}/m2_savings_amt.pkl")
        joblib.dump(m3,       f"{MODEL_DIR}/m3_disposable.pkl")
        joblib.dump(m4,       f"{MODEL_DIR}/m4_budget_clf.pkl")
        joblib.dump(m5,       f"{MODEL_DIR}/m5_health_score.pkl")
        joblib.dump(m6,       f"{MODEL_DIR}/m6_invest_cluster.pkl")
        joblib.dump(scaler,   f"{MODEL_DIR}/m6_scaler.pkl")
        joblib.dump(le_occ,   f"{MODEL_DIR}/enc_occupation.pkl")
        joblib.dump(le_tier,  f"{MODEL_DIR}/enc_city_tier.pkl")
        joblib.dump(le_budget,f"{MODEL_DIR}/enc_budget.pkl")
        joblib.dump(CORE_FEATURES, f"{MODEL_DIR}/features.pkl")

        meta = {
            "dataset_rows": int(len(df)),
            "models": {
                "M1": {"name":"GradientBoostingRegressor","target":"Desired_Savings_Percentage","r2":round(r2_score(yp_te,p1),4),"mae":round(float(mean_absolute_error(yp_te,p1)),3)},
                "M2": {"name":"GradientBoostingRegressor","target":"Desired_Savings","r2":round(r2_score(ya_te,p2),4),"mae":round(float(mean_absolute_error(ya_te,p2)),2)},
                "M3": {"name":"GradientBoostingRegressor","target":"Disposable_Income","r2":round(r2_score(yd_te,p3),4),"mae":round(float(mean_absolute_error(yd_te,p3)),2)},
                "M4": {"name":"RandomForestClassifier","target":"budget_profile","accuracy":round(acc4,4)},
                "M5": {"name":"GradientBoostingRegressor","target":"health_score","r2":round(r2_score(yh_te,p5),4),"mae":round(float(mean_absolute_error(yh_te,p5)),3)},
                "M6": {"name":"KMeans","target":"investment_cluster","k":4},
            },
            "occupation_classes": list(le_occ.classes_),
            "city_tier_classes":  list(le_tier.classes_),
            "budget_classes":     list(le_budget.classes_),
            "cluster_profiles":   CLUSTER_PROFILES,
        }
        with open(f"{MODEL_DIR}/metadata.json","w") as f:
            json.dump(meta, f, indent=2)

        print(f"[save]  All 6 models + encoders → ./{MODEL_DIR}/")
        for k,v in meta["models"].items():
            r = f"R²={v['r2']}" if "r2" in v else f"acc={v.get('accuracy','')}"
            print(f"        {k} · {v['name']:<30} {r}")

    return models, {"le_occ":le_occ,"le_tier":le_tier,"le_budget":le_budget,"scaler":scaler}


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD SAVED MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    required = ["m1_savings_pct.pkl","m2_savings_amt.pkl","m3_disposable.pkl",
                "m4_budget_clf.pkl","m5_health_score.pkl","m6_invest_cluster.pkl",
                "m6_scaler.pkl","enc_occupation.pkl","enc_city_tier.pkl","enc_budget.pkl"]
    for f in required:
        if not os.path.exists(f"{MODEL_DIR}/{f}"):
            raise FileNotFoundError(f"[ERROR] {MODEL_DIR}/{f} not found. Run --train first.")
    return {
        "m1": joblib.load(f"{MODEL_DIR}/m1_savings_pct.pkl"),
        "m2": joblib.load(f"{MODEL_DIR}/m2_savings_amt.pkl"),
        "m3": joblib.load(f"{MODEL_DIR}/m3_disposable.pkl"),
        "m4": joblib.load(f"{MODEL_DIR}/m4_budget_clf.pkl"),
        "m5": joblib.load(f"{MODEL_DIR}/m5_health_score.pkl"),
        "m6": joblib.load(f"{MODEL_DIR}/m6_invest_cluster.pkl"),
        "scaler":    joblib.load(f"{MODEL_DIR}/m6_scaler.pkl"),
        "le_occ":    joblib.load(f"{MODEL_DIR}/enc_occupation.pkl"),
        "le_tier":   joblib.load(f"{MODEL_DIR}/enc_city_tier.pkl"),
        "le_budget":  joblib.load(f"{MODEL_DIR}/enc_budget.pkl"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT  —  COMBINES ALL 6 MODELS
# ══════════════════════════════════════════════════════════════════════════════

def predict(
    # Profile (from logged-in user)
    name          : str   = "User",
    income        : float = 0,
    age           : int   = 30,
    dependents    : int   = 0,
    occupation    : str   = "Self_Employed",
    city_tier     : str   = "Tier_2",
    location      : str   = "",
    # Form inputs
    business_type : str   = "",
    monthly_savings: float= 0,
    goal          : str   = "",
    risk_level    : str   = "Medium Risk",  # "Low Risk" | "Medium Risk" | "High Risk"
    # Monthly expenses (optional — estimated if not given)
    rent          : float = 0,
    loan_repayment: float = 0,
    insurance     : float = 0,
    groceries     : float = 0,
    transport     : float = 0,
    eating_out    : float = 0,
    entertainment : float = 0,
    utilities     : float = 0,
    healthcare    : float = 0,
    education     : float = 0,
    miscellaneous : float = 0,
    loaded        : dict  = None,
) -> dict:
    """
    Run all 6 models and return a 10-section financial roadmap.

    Each section is a narrative paragraph + bullet list,
    matching the expected output format exactly.
    """
    if loaded is None:
        loaded = load_models()

    le_occ   = loaded["le_occ"]
    le_tier  = loaded["le_tier"]
    le_budget = loaded["le_budget"]
    scaler   = loaded["scaler"]

    # ── Estimate expenses from income if not provided ─────────────────────────
    tier_rent_pct = {"Tier_1":0.30,"Tier_2":0.20,"Tier_3":0.15}.get(city_tier, 0.20)
    if rent == 0:     rent      = income * tier_rent_pct
    if groceries==0:  groceries = income * 0.125
    if transport==0:  transport = income * 0.065
    if utilities==0:  utilities = income * 0.060
    if insurance==0:  insurance = income * 0.035
    if healthcare==0: healthcare= income * 0.040
    if education==0 and dependents>0: education = income * 0.060
    if miscellaneous==0: miscellaneous = income * 0.020

    total_exp = sum([rent,loan_repayment,insurance,groceries,transport,
                     eating_out,entertainment,utilities,healthcare,education,miscellaneous])
    disp_calc = max(0, income - total_exp)
    exp_ratio = total_exp / income if income > 0 else 1.0

    # Potential savings (dataset ratios)
    pot_savings = {
        "Groceries"    : income * 0.0219,
        "Transport"    : income * 0.0113,
        "Utilities"    : income * 0.0105,
        "Eating_Out"   : income * 0.0061,
        "Entertainment": income * 0.0061,
        "Miscellaneous": income * 0.0035,
    }
    total_pot = sum(pot_savings.values())

    # ── Build feature row ─────────────────────────────────────────────────────
    row = pd.DataFrame([{
        "Income":income,"Age":age,"Dependents":dependents,
        "Occupation":occupation,"City_Tier":city_tier,
        "Rent":rent,"Loan_Repayment":loan_repayment,"Insurance":insurance,
        "Groceries":groceries,"Transport":transport,"Eating_Out":eating_out,
        "Entertainment":entertainment,"Utilities":utilities,"Healthcare":healthcare,
        "Education":education,"Miscellaneous":miscellaneous,
        "Disposable_Income":disp_calc,
        **{c:0 for c in POTENTIAL_COLS},
    }])
    row["Potential_Savings_Groceries"]    = groceries     * 0.268
    row["Potential_Savings_Transport"]    = transport     * 0.175
    row["Potential_Savings_Eating_Out"]   = eating_out    * 0.282
    row["Potential_Savings_Entertainment"]= entertainment * 0.282
    row["Potential_Savings_Utilities"]    = utilities     * 0.175
    row["Potential_Savings_Healthcare"]   = healthcare    * 0.064
    row["Potential_Savings_Education"]    = education     * 0.025
    row["Potential_Savings_Miscellaneous"]= miscellaneous * 0.175

    row = engineer(row, le_occ, le_tier, for_training=False)
    X   = row[CORE_FEATURES]
    X2  = row[BUDGET_FEATURES]
    actual_sav_pct_val = monthly_savings/income*100 if income>0 else 10
    cluster_row = row.copy()
    cluster_row['Desired_Savings_Percentage'] = actual_sav_pct_val
    Xc  = scaler.transform(cluster_row[CLUSTER_FEATURES].assign(
              Desired_Savings_Percentage=monthly_savings/income*100 if income>0 else 10))

    # ── Run all 6 models ──────────────────────────────────────────────────────
    pred_pct    = float(np.clip(loaded["m1"].predict(X)[0], 5.0, 40.0))
    pred_amt    = float(np.clip(loaded["m2"].predict(X)[0], 0.0, income))
    pred_disp   = float(np.clip(loaded["m3"].predict(X)[0], 0.0, income))
    budget_enc  = loaded["m4"].predict(X2)[0]
    budget_prof = le_budget.inverse_transform([budget_enc])[0]
    health_raw  = float(np.clip(loaded["m5"].predict(X)[0], 10, 95))
    inv_cluster = int(loaded["m6"].predict(Xc)[0])
    cluster_info= CLUSTER_PROFILES[inv_cluster]

    # ── Derived calculations ──────────────────────────────────────────────────
    actual_sav_pct = monthly_savings / income * 100 if income > 0 else 0
    health_score   = int(np.clip(health_raw, 10, 95))
    rent_pct       = rent / income * 100 if income > 0 else 0
    emergency_tgt  = total_exp * 6
    months_surplus = disp_calc
    annual_income  = income * 12

    # ── Goal analysis ─────────────────────────────────────────────────────────
    goal_amount    = 20_000_000   # ₹2 crore default
    goal_years     = 5
    # detect goal amount from text
    import re
    m = re.search(r'(\d+)\s*cr', goal.lower())
    if m: goal_amount = int(m.group(1)) * 10_000_000
    m2g = re.search(r'(\d+)\s*year', goal.lower())
    if m2g: goal_years = int(m2g.group(1))

    monthly_needed_for_goal = goal_amount / (goal_years * 12)
    home_loan_emi           = goal_amount * 0.20 / (goal_years * 12)  # 20% down payment
    feasibility             = "achievable with home loan" if income * 0.40 >= home_loan_emi else "challenging"

    # ── Risk → allocation override ────────────────────────────────────────────
    if risk_level == "Low Risk":
        alloc = {"equity":20,"debt":55,"gold":15,"liquid":10}
        instruments = ["Post Office FD","PPF (7.1%)","NSC (7.7%)","SCSS (8.2%)","Debt MF"]
    elif risk_level == "High Risk":
        alloc = {"equity":75,"debt":10,"gold":5,"liquid":10}
        instruments = cluster_info["instruments"]
    else:
        alloc = {"equity":50,"debt":30,"gold":10,"liquid":10}
        instruments = cluster_info["instruments"]

    schemes = OCCUPATION_SCHEMES.get(occupation, OCCUPATION_SCHEMES["Self_Employed"])
    if "dairy" in business_type.lower() or "farm" in business_type.lower():
        schemes = DAIRY_SCHEMES + schemes[:2]

    insurance_recs = OCCUPATION_INSURANCE.get(occupation, OCCUPATION_INSURANCE["Self_Employed"])
    tier_benefits  = TIER_BENEFITS.get(city_tier, TIER_BENEFITS["Tier_3"])

    INR = lambda n: f"₹{round(n):,}"

    # ══════════════════════════════════════════════════════════════════════════
    #  10-SECTION ROADMAP  (matches expected output format)
    # ══════════════════════════════════════════════════════════════════════════
    roadmap = {}

    # ─── SECTION 1: FINANCIAL HEALTH SNAPSHOT ─────────────────────────────────
    health_label = "excellent" if health_score>=80 else "good" if health_score>=60 else "fair" if health_score>=40 else "poor"
    roadmap["1_FINANCIAL_HEALTH_SNAPSHOT"] = {
        "title": "FINANCIAL HEALTH SNAPSHOT",
        "narrative": (
            f"{name}, at the age of {age}, has a monthly income of {INR(income)} and current savings "
            f"of {INR(monthly_savings)} per month. With a financial goal to {goal}, it is essential to "
            f"create a comprehensive plan to achieve this objective. "
            f"Based on the dataset analysis of 20,000 profiles, {name}'s financial health score is "
            f"{health_score}/100 — rated {health_label}. "
            f"The ML model recommends saving {pred_pct:.1f}% of income = {INR(pred_amt)}/month, "
            f"while the current savings rate is {actual_sav_pct:.1f}%. "
            f"Disposable income is estimated at {INR(pred_disp)}/month after all expenses."
        ),
        "bullets": [
            f"Health Score: {health_score}/100 ({health_label.title()})",
            f"ML-recommended savings rate: {pred_pct:.1f}% = {INR(pred_amt)}/month",
            f"Current savings rate: {actual_sav_pct:.1f}% ({INR(monthly_savings)}/month)",
            f"Monthly disposable income: {INR(pred_disp)}",
            f"Expense ratio: {exp_ratio*100:.1f}% of income (budget profile: {budget_prof})",
            f"Rent burden: {rent_pct:.1f}% of income (city norm: {tier_rent_pct*100:.0f}%)",
        ],
        "metrics": {
            "health_score": f"{health_score}/100",
            "savings_pct_recommended": f"{pred_pct:.1f}%",
            "disposable_income": INR(pred_disp),
        },
    }

    # ─── SECTION 2: PERSONALIZED GOAL STRATEGY ────────────────────────────────
    down_pct     = 20
    down_payment = goal_amount * down_pct / 100
    monthly_sip_for_down = down_payment / (goal_years * 12)
    corpus_5yr   = pred_amt * 12 * goal_years * 1.10  # ~10% annualised growth
    roadmap["2_PERSONALIZED_GOAL_STRATEGY"] = {
        "title": "PERSONALIZED GOAL STRATEGY",
        "narrative": (
            f"To {goal}, {name} needs to save approximately {INR(monthly_needed_for_goal)} per month "
            f"(full cash) or accumulate a down payment of {INR(down_payment)} ({down_pct}%) and take a home loan "
            f"for the balance. A more realistic approach is to save {INR(monthly_sip_for_down)}/month for the "
            f"down payment while applying for a home loan at current rates of 8.35–9.5% p.a. "
            f"With the ML-recommended savings of {INR(pred_amt)}/month and an expected 10–14% CAGR, "
            f"a 5-year corpus of approximately {INR(corpus_5yr)} is achievable."
        ),
        "bullets": [
            f"Short-term (0–1 yr): Save {INR(pred_amt*12)} — build liquid emergency fund",
            f"Medium-term (1–3 yr): Accumulate {INR(pred_amt*36)} via SIP + PPF",
            f"Long-term (3–5 yr): Target {INR(down_payment)} for down payment on {INR(goal_amount)} house",
            f"Goal feasibility: {feasibility.title()}",
            f"Home loan EMI (80% of {INR(goal_amount)} @ 9%): {INR(goal_amount*0.8*0.09/12*(1+0.09/12)**(goal_years*12)/((1+0.09/12)**(goal_years*12)-1))}/month (30-yr tenure)",
            f"Recommended SIP for down payment: {INR(monthly_sip_for_down)}/month for {goal_years} years",
        ],
        "metrics": {
            "goal_amount": INR(goal_amount),
            "down_payment_needed": INR(down_payment),
            "monthly_sip_for_goal": INR(monthly_sip_for_down),
        },
    }

    # ─── SECTION 3: SMART BUDGETING PLAN ──────────────────────────────────────
    need_alloc = round(income * 0.50)
    sav_alloc  = round(income * 0.25)
    biz_alloc  = round(income * 0.20)
    misc_alloc = round(income * 0.05)
    roadmap["3_SMART_BUDGETING_PLAN"] = {
        "title": "SMART BUDGETING PLAN",
        "narrative": (
            f"The Random Forest budget classifier (M4) identifies {name}'s current budget profile as "
            f"'{budget_prof}' based on an expense ratio of {exp_ratio*100:.1f}%. "
            f"To achieve the financial goal, the following allocation of the monthly income of {INR(income)} is recommended:"
        ),
        "bullets": [
            f"Essential expenses (rent, food, utilities): {INR(need_alloc)} (50% of monthly income)",
            f"Savings and investments: {INR(sav_alloc)} (25% of monthly income)",
            f"Business expenses ({business_type}): {INR(biz_alloc)} (20% of monthly income)",
            f"Miscellaneous / buffer: {INR(misc_alloc)} (5% of monthly income)",
            f"Potential monthly savings (dataset-derived): {INR(total_pot)}",
            f"  — Groceries: {INR(pot_savings['Groceries'])} | Transport: {INR(pot_savings['Transport'])} | Utilities: {INR(pot_savings['Utilities'])}",
        ],
        "metrics": {
            "budget_profile": budget_prof.title(),
            "expense_ratio": f"{exp_ratio*100:.1f}%",
            "potential_monthly_savings": INR(total_pot),
        },
    }

    # ─── SECTION 4: INVESTMENT ROADMAP ────────────────────────────────────────
    roadmap["4_INVESTMENT_ROADMAP"] = {
        "title": "INVESTMENT ROADMAP",
        "narrative": (
            f"KMeans clustering (M6) places {name} in the '{cluster_info['label']}' investment profile "
            f"with a {risk_level.lower()} appetite. The recommended asset allocation is: "
            f"Equity {alloc['equity']}%, Debt {alloc['debt']}%, Gold {alloc['gold']}%, Liquid {alloc['liquid']}%. "
            f"This strategy targets {cluster_info['expected_return']} returns."
        ),
        "bullets": [
            f"Fixed Deposits / Debt instruments: {alloc['debt']}% of investments",
            f"Public Provident Fund (PPF @ 7.1%): include in debt portion",
            f"Mutual Funds (equity SIP): {alloc['equity']}% — {instruments[0] if instruments else 'NIFTY 50 Index Fund'}",
            f"Gold (SGB @ 8% + gold returns): {alloc['gold']}%",
            f"Liquid / emergency buffer: {alloc['liquid']}%",
            f"Monthly SIP target: {INR(round(pred_amt * alloc['equity'] / 100))} in equity MF",
        ],
        "metrics": {
            "equity_allocation": f"{alloc['equity']}%",
            "debt_allocation": f"{alloc['debt']}%",
            "expected_return": cluster_info["expected_return"],
        },
    }

    # ─── SECTION 5: EMERGENCY & SAVINGS ───────────────────────────────────────
    roadmap["5_EMERGENCY_AND_SAVINGS"] = {
        "title": "EMERGENCY & SAVINGS",
        "narrative": (
            f"It is essential to have an emergency fund covering 3–6 months of living expenses. "
            f"{name} should aim to save at least {INR(total_exp * 3)} (3-month buffer) "
            f"and ideally {INR(emergency_tgt)} (6-month buffer) in an easily accessible savings account. "
            f"This fund will prevent dipping into long-term investments during emergencies."
        ),
        "bullets": [
            f"Emergency fund target (3 months): {INR(total_exp * 3)}",
            f"Emergency fund target (6 months): {INR(emergency_tgt)}",
            f"Monthly savings target (ML model): {INR(pred_amt)}",
            f"High-yield options: Small Finance Bank savings (7%), Liquid MF (~7%), Sweep-in FD",
            f"Timeline to build 6-month fund at {INR(pred_amt)}/mo: {max(1,round(emergency_tgt/max(pred_amt,1)))} months",
        ],
        "metrics": {
            "emergency_3mo": INR(total_exp * 3),
            "emergency_6mo": INR(emergency_tgt),
            "monthly_target": INR(pred_amt),
        },
    }

    # ─── SECTION 6: RISK PROTECTION PLAN ──────────────────────────────────────
    term_cover     = annual_income * 15
    health_cover   = max(500_000, total_exp * 12)
    term_premium   = annual_income * 0.001
    roadmap["6_RISK_PROTECTION_PLAN"] = {
        "title": "RISK PROTECTION PLAN",
        "narrative": (
            f"As a {business_type or occupation.replace('_',' ').lower()}, {name} is exposed to "
            f"business, health, and income-loss risks. The following insurance coverage is recommended "
            f"to protect {name}'s family of {dependents} dependents."
        ),
        "bullets": insurance_recs + [
            f"Term life cover recommended: {INR(term_cover)} (15× annual income)",
            f"Estimated term premium: {INR(term_premium)}/year at age {age}",
            f"Family health floater: {INR(health_cover)} minimum",
            f"Claim: Section 80D deduction up to ₹25,000/year",
        ],
        "metrics": {
            "term_life_cover": INR(term_cover),
            "health_cover": INR(health_cover),
            "annual_premium_est": INR(term_premium),
        },
    }

    # ─── SECTION 7: TAX OPTIMIZATION ──────────────────────────────────────────
    taxable_new      = max(0, annual_income - 75_000)         # std deduction new regime
    taxable_old      = max(0, annual_income - 50_000 - 150_000 - 50_000)  # std + 80C + NPS
    max_deductions   = 150_000 + 50_000 + 25_000             # 80C + NPS + 80D
    roadmap["7_TAX_OPTIMIZATION"] = {
        "title": "TAX OPTIMIZATION",
        "narrative": (
            f"For FY 2025-26, {name} can reduce taxable income by up to {INR(max_deductions)} "
            f"through strategic investments. Under the old tax regime, Section 80C alone allows "
            f"₹1,50,000 in deductions via PPF, ELSS, NSC, or LIC premium."
        ),
        "bullets": [
            f"Section 80C (PPF/ELSS/LIC/NSC/EPF): ₹1,50,000 deduction",
            f"Section 80CCD(1B) — NPS additional: ₹50,000 deduction",
            f"Section 80D — health insurance premium: ₹25,000 deduction",
            f"Standard deduction (new regime): ₹75,000",
            f"Total max deductions: {INR(max_deductions)} (old regime)",
            f"Recommended instruments: PPF, ELSS, NSC, LIC, NPS Tier-I",
        ],
        "metrics": {
            "max_80C": "₹1,50,000",
            "max_NPS_extra": "₹50,000",
            "total_deductions": INR(max_deductions),
        },
    }

    # ─── SECTION 8: BUSINESS FINANCE GUIDANCE ─────────────────────────────────
    roadmap["8_BUSINESS_FINANCE_GUIDANCE"] = {
        "title": "BUSINESS FINANCE GUIDANCE",
        "narrative": (
            f"As a {business_type or 'rural business owner'}, {name} can explore various "
            f"government-backed finance options to expand operations. The following schemes are "
            f"available for {occupation.replace('_',' ').lower()} in {location or city_tier} region."
        ),
        "bullets": schemes,
        "tips": [
            "Maintain separate personal and business bank accounts",
            "Register your business with Udyam (MSME) portal for scheme access",
            "Keep digital records of all business transactions for loan eligibility",
            "Maintain a 6-month business expense reserve before expansion",
        ],
        "metrics": {
            "occupation": occupation.replace("_"," "),
            "max_loan_available": "₹10L–₹2Cr (scheme dependent)",
            "subsidy_available": "15–35% (PMEGP/NABARD)",
        },
    }

    # ─── SECTION 9: LOCATION-SPECIFIC BENEFITS ────────────────────────────────
    roadmap["9_LOCATION_SPECIFIC_BENEFITS"] = {
        "title": "LOCATION-SPECIFIC BENEFITS",
        "narrative": (
            f"As {name} is based in {location or city_tier}, several local government schemes and "
            f"investment opportunities are available. Contact the local agricultural department and "
            f"District Industries Centre (DIC) to access these schemes."
        ),
        "bullets": tier_benefits + [
            f"Connect with local {business_type or 'business'} associations for peer learning",
            "Visit Aaple Sarkar (Maharashtra) portal for state scheme applications",
        ],
        "metrics": {
            "city_tier": city_tier,
            "location": location or "Maharashtra",
            "schemes_available": str(len(tier_benefits)),
        },
    }

    # ─── SECTION 10: 30-DAY ACTION PLAN ───────────────────────────────────────
    roadmap["10_30_DAY_ACTION_PLAN"] = {
        "title": "YOUR 30-DAY ACTION PLAN",
        "narrative": (
            f"By following this structured 30-day plan, {name} will lay the foundation for achieving "
            f"the financial goal and improving overall financial health from {health_score}/100 to 80+."
        ),
        "week_1": [
            f"Open a PPF account at SBI/Post Office and deposit {INR(min(12500, pred_amt*0.3))}",
            f"Apply for a crop/livestock insurance policy for {business_type or 'business'} protection",
            "Meet a SEBI-registered financial advisor to finalize the investment portfolio",
        ],
        "week_2": [
            f"Start a ELSS SIP of {INR(round(pred_amt*0.3))} via Groww/Zerodha/CAMS",
            f"Open a high-yield savings account (Small Finance Bank, 7%) for emergency fund",
            "Create a detailed monthly budget tracker (Google Sheets or ET Money app)",
        ],
        "week_3": [
            "Research and apply for local government dairy/agri schemes listed above",
            "Compare term insurance plans on Policybazaar and buy coverage",
            f"Set a recurring transfer of {INR(pred_amt)} on 1st of every month",
        ],
        "week_4": [
            "Review first-month actual expenses vs the budget plan",
            f"Consult a home loan advisor about eligibility for {INR(goal_amount)} house loan",
            "Register business on Udyam portal to access MSME scheme benefits",
        ],
    }

    return {
        "name": name, "income": income, "age": age, "dependents": dependents,
        "ml_outputs": {
            "M1_savings_pct": round(pred_pct, 2),
            "M2_savings_amt": round(pred_amt, 2),
            "M3_disposable":  round(pred_disp, 2),
            "M4_budget_profile": budget_prof,
            "M5_health_score": health_score,
            "M6_invest_cluster": f"Cluster {inv_cluster} — {cluster_info['label']}",
        },
        "roadmap": roadmap,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINT
# ══════════════════════════════════════════════════════════════════════════════

def print_roadmap(result: dict):
    """Print the 10-section roadmap in the expected output format."""
    ml = result["ml_outputs"]
    print("\n" + "═"*70)
    print("  FINANCIAL ADVISOR — PERSONALISED ROADMAP")
    print("═"*70)
    print(f"  Name: {result['name']}  |  Income: ₹{result['income']:,}/mo  |  Age: {result['age']}")
    print(f"\n  ML MODEL OUTPUTS")
    for k, v in ml.items():
        print(f"  {k}: {v}")
    print("═"*70)

    for sec_key, sec in result["roadmap"].items():
        title = sec.get("title", sec_key)
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")
        if "narrative" in sec:
            # word-wrap narrative
            words = sec["narrative"].split()
            line = "  "; lines = []
            for w in words:
                if len(line)+len(w)+1 > 72:
                    lines.append(line); line = "  "+w+" "
                else:
                    line += w+" "
            if line.strip(): lines.append(line)
            print("\n".join(lines))

        for key in ["bullets","tips","week_1","week_2","week_3","week_4"]:
            if key in sec:
                label = {"bullets":"","tips":"\n  Tips:","week_1":"\n  Week 1:","week_2":"\n  Week 2:","week_3":"\n  Week 3:","week_4":"\n  Week 4:"}.get(key,"")
                if label: print(label)
                for b in sec[key]:
                    print(f"  ✓ {b}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK REST API
# ══════════════════════════════════════════════════════════════════════════════

def start_api(port=5000):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("[api] Install Flask first:  pip install flask"); return

    app = Flask(__name__)
    lm  = load_models()

    @app.get("/health")
    def health(): return jsonify({"status":"ok","models":6})

    @app.post("/predict")
    def predict_ep():
        try:
            d = request.get_json(force=True)
            result = predict(loaded=lm, **{k:v for k,v in d.items()})
            return jsonify(result)
        except Exception as e:
            return jsonify({"error":str(e)}), 400

    print(f"[api]  POST http://localhost:{port}/predict")
    app.run(host="0.0.0.0", port=port, debug=False)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",   action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--api",     action="store_true")
    p.add_argument("--port",    type=int, default=5000)
    p.add_argument("--data",    default=DATA_PATH)
    args = p.parse_args()

    if args.api:
        start_api(args.port)
    elif args.predict:
        result = predict(
            name="Mitesh Chaudhari", income=100_000, age=22, dependents=5,
            occupation="Self_Employed", city_tier="Tier_3", location="Dhule",
            business_type="Dairy Farming", monthly_savings=50_000,
            goal="buy a house worth 2 crores in 5 years", risk_level="Low Risk",
        )
        print_roadmap(result)
    else:
        # Default: train then demo
        train(args.data)
        result = predict(
            name="Mitesh Chaudhari", income=100_000, age=22, dependents=5,
            occupation="Self_Employed", city_tier="Tier_3", location="Dhule",
            business_type="Dairy Farming", monthly_savings=50_000,
            goal="buy a house worth 2 crores in 5 years", risk_level="Low Risk",
        )
        print_roadmap(result)
