# -*- coding: utf-8 -*-
"""
PoF3 - Tüm Adımlar Tek Script (01→05) | Leakage-Controlled Hybrid FINAL
========================================================================

Amaç:
- Tek komutla Step-01..Step-05 çalıştırmak.
- Survival (Cox/Weibull/RSF): sadece yapısal (02a) -> first-failure survival
- ML PoF: iki mod
    MODE_A (default) : from_install      -> Kurulumdan sonra H gün içinde ilk arıza olur mu?
    MODE_B (optional): next_H_at_Tref    -> T_ref sonrası H gün içinde arıza olur mu? (snapshot gerekir)
- Chronic: Bayesian MTBF + exponential decay + trend + flag
- Risk: PoF×CoF (Müşteri-facing)

Girdiler:
- data/girdiler/ariza_final.xlsx
- data/girdiler/saglam_final.xlsx

Çıktılar:
- data/ara_ciktilar/*.csv + analysis_metadata.json
- data/sonuclar/*.csv + ensemble_pof_final.csv
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# LOAD YAML CONFIG
# ----------------------------
with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ----------------------------
# PATH RESOLUTION
# ----------------------------
DATA_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["base"])
INPUT_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["input"])
INTERMEDIATE_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["intermediate"])
OUTPUT_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["output"])
LOG_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["logs"])

INTERMEDIATE_PATHS = {
    k: os.path.join(INTERMEDIATE_DIR, v)
    for k, v in CFG["intermediate_paths"].items()
}

OUTPUT_PATHS = {
    k: os.path.join(OUTPUT_DIR, v)
    for k, v in CFG["output_paths"].items()
}

SURVIVAL_HORIZONS_DAYS = CFG["survival"]["horizons_days"]
SURVIVAL_HORIZON_LABELS = CFG["survival"]["horizon_labels"]

# -----------------------------
# Project root safe import
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -----------------------------
# Soft dependencies
# -----------------------------
LIFELINES_OK = True
SKSURV_OK = True
XGB_OK = True
CAT_OK = True

try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index
except Exception:
    LIFELINES_OK = False

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
except Exception:
    SKSURV_OK = False

try:
    from xgboost import XGBClassifier
except Exception:
    XGB_OK = False

try:
    from catboost import CatBoostClassifier
except Exception:
    CAT_OK = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy import stats


# =============================================================================
# CONFIG - DEDUPLICATED
# =============================================================================
# Build full paths for data files
DATA_PATHS = {
    k: os.path.join(BASE_DIR, v)
    for k, v in CFG["data_paths"].items()
}

MIN_EQUIPMENT_PER_CLASS = CFG["analysis"]["min_equipment_per_class"]
REQUIRE_INSTALL_BEFORE_DATA_END = CFG["analysis"]["require_install_before_data_end"]
ANALYSIS_METADATA_PATH = os.path.join(BASE_DIR, CFG["analysis"]["analysis_metadata_path"])

CHRONIC_CFG = CFG.get("chronic", {})
CHRONIC_WINDOW_DAYS = CHRONIC_CFG.get("window_days_default", 90)
CHRONIC_WINDOW_DAYS_DEFAULT = CHRONIC_WINDOW_DAYS
CHRONIC_THRESHOLD_EVENTS = CHRONIC_CFG.get("min_events_default", 3)
CHRONIC_MIN_EVENTS_DEFAULT = CHRONIC_THRESHOLD_EVENTS
CHRONIC_MIN_RATE = CHRONIC_CFG.get("min_rate_per_year_default", 1.5)
CHRONIC_MIN_RATE_DEFAULT = CHRONIC_MIN_RATE
CHRONIC_PER_TYPE = CHRONIC_CFG.get("per_type", {})

RISK_CFG = CFG.get("risk", {})
POF_THRESHOLDS = RISK_CFG.get("pof_thresholds", {})
COF_BANDS = RISK_CFG.get("cof_bands", [])

ENSEMBLE_CFG = CFG.get("ensemble", {})
CRITICAL_HEALTH_THRESHOLD = ENSEMBLE_CFG.get("critical_health_threshold", 40)

EXTRA_FAULT_COLS = CFG.get("fault_columns", {}).get("extra_fault_cols", [])
COLUMN_MAPPING = CFG.get("fault_columns", {}).get("column_mapping", {})

# =============================================================================
# LOGGING
# =============================================================================
def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"pof3_tum_adimlar_{ts}.log")

    logger = logging.getLogger("pof3_tum_adimlar")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Use UTF-8 encoding for console handler on Windows
    import io
    ch = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("[DEBUG] LOG FILE: %s", log_path)
    logger.info("=" * 80)
    logger.info("PoF3 - Tum Adimlar Tek Script (01-05) | Hybrid FINAL")
    logger.info("=" * 80)
    return logger


# =============================================================================
# HELPERS
# =============================================================================
def read_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def ensure_dirs():
    # First ensure base directories exist
    for base_dir in [INTERMEDIATE_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(base_dir, exist_ok=True)
    
    # Then ensure subdirectories for each file
    for p in list(INTERMEDIATE_PATHS.values()) + list(OUTPUT_PATHS.values()):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

def convert_duration_minutes(series: pd.Series, logger: logging.Logger) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    logger.info(f"[INFO] Süre medyanı (ham): {med}")
    if pd.notna(med) and med > 10000:
        logger.info("[INFO] Sureler milisaniye -> dakikaya donusuyor.")
        return s / 60000.0
    return s

def clean_equipment_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return (
        s.str.replace(" Arızaları", "", regex=False)
         .str.replace(" Ariza", "", regex=False)
         .str.strip()
    )
def parse_date_safely(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT
def get_chronic_params(equipment_type: str):
    """
    Equipment-aware chronic parameters.
    Falls back to global defaults if no override exists.
    """
    if isinstance(equipment_type, str) and equipment_type in CHRONIC_PER_TYPE:
        cfg = CHRONIC_PER_TYPE[equipment_type]
        return (
            cfg.get("window_days", CHRONIC_WINDOW_DAYS_DEFAULT),
            cfg.get("min_events", CHRONIC_MIN_EVENTS_DEFAULT),
            CHRONIC_MIN_RATE_DEFAULT,
        )

    return (
        CHRONIC_WINDOW_DAYS_DEFAULT,
        CHRONIC_MIN_EVENTS_DEFAULT,
        CHRONIC_MIN_RATE_DEFAULT,
    )
def compute_chronic_flag(
    fault_dates: pd.Series,
    equipment_type: str,
    t_ref: pd.Timestamp,
):
    """
    IEEE-style chronic detection with per-equipment override.
    """
    window_days, min_events, min_rate = get_chronic_params(equipment_type)

    if fault_dates.empty:
        return 0, 0, 0.0

    window_start = t_ref - pd.Timedelta(days=window_days)
    recent_faults = fault_dates[fault_dates >= window_start]

    n_events = len(recent_faults)
    rate_per_year = n_events / (window_days / 365.0)

    is_chronic = int(
        (n_events >= min_events) and (rate_per_year >= min_rate)
    )

    return is_chronic, n_events, rate_per_year
def calculate_cof(customer_count: float) -> int:
    """
    Consequence of Failure based on customer impact bands.
    """
    if pd.isna(customer_count):
        return 1  # Low / Unknown

    for band in COF_BANDS:
        if customer_count <= band["max_customers"]:
            return band["cof"]

    return 1
def calculate_risk_level(pof_category: str, cof: int) -> str:
    """
    Rule-based risk classification (replaces static risk matrix).
    """
    if pof_category in ["High", "Very High"] and cof >= 3:
        return "Critical"

    if pof_category in ["High"] and cof == 2:
        return "High"

    if pof_category in ["Medium"] and cof >= 2:
        return "Medium"

    return "Low"
def compute_health_score(mean_pof: float) -> float:
    """
    Converts mean PoF into customer-facing health score (0–100).
    """
    if pd.isna(mean_pof):
        return 100.0

    return float(np.clip(100.0 * (1.0 - mean_pof), 0, 100))
def enrich_asset_risk_row(row: pd.Series) -> pd.Series:
    """
    Final enrichment applied row-wise to ensemble output.
    """
    pof_cat = categorize_pof(row.get("Mean_PoF"))
    cof = calculate_cof(row.get("Musteri_Sayisi"))
    risk = calculate_risk_level(pof_cat, cof)

    row["PoF_Category"] = pof_cat
    row["CoF"] = cof
    row["Risk_Level"] = risk
    row["Critical_Flag"] = int(row.get("Health_Score", 100) < CRITICAL_HEALTH_THRESHOLD)

    return row

def rename_maintenance_and_attributes(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    col_map = {
        "Bakım Sayısı": "Bakim_Sayisi",
        "Geçmiş İş Emri Tipleri": "Bakim_Is_Emri_Tipleri",
        "İlk Bakım İş Emri Tarihi": "Ilk_Bakim_Tarihi",
        "Son Bakım İş Emri Tarihi": "Son_Bakim_Tarihi",
        "Son Bakım İş Emri Tipi": "Son_Bakim_Tipi",
        "Son Bakımdan İtibaren Geçen Gün Sayısı": "Son_Bakimdan_Gecen_Gun",
        "MARKA": "Marka",
        "component_voltage": "Gerilim_Seviyesi",
        "voltage_level": "Gerilim_Sinifi",
        "kVA_Rating": "kVA_Rating",
    }
    to_rename = {k: v for k, v in col_map.items() if k in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)
        logger.info(f"[MAINTENANCE] Renamed {len(to_rename)} maintenance/attribute columns")

    for date_col in ["Ilk_Bakim_Tarihi", "Son_Bakim_Tarihi"]:
        if date_col in df.columns:
            df[date_col] = df[date_col].apply(parse_date_safely)

    for num_col in ["Bakim_Sayisi", "Son_Bakimdan_Gecen_Gun", "kVA_Rating", "Gerilim_Seviyesi"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    return df

def detect_duration_outliers(df: pd.DataFrame, logger: logging.Logger, output_dir: str) -> pd.DataFrame:
    duration_col = "Süre_Dakika"
    if duration_col not in df.columns:
        return df

    neg_mask = df[duration_col] <= 0
    if neg_mask.sum() > 0:
        logger.warning(f"[PHYSICS] Dropping {neg_mask.sum()} records with duration <= 0 minutes")
        df = df[~neg_mask].copy()

    log_durations = np.log1p(df[duration_col])
    median_log = log_durations.median()
    mad_log = stats.median_abs_deviation(log_durations)
    if mad_log == 0:
        mad_log = 1e-6
    sigma_robust = mad_log * 1.4826
    upper_limit_log = median_log + (6 * sigma_robust)

    upper_limit_min = np.expm1(upper_limit_log)
    logger.info("[OUTLIER] Robust log-normal check:")
    logger.info("  - Median Duration: %.1f min", float(np.expm1(median_log)))
    logger.info("  - Upper Cutoff (6-sigma): %.1f min (%.1f days)", float(upper_limit_min), float(upper_limit_min/60/24))

    outlier_mask = log_durations > upper_limit_log
    n_outliers = int(outlier_mask.sum())
    if n_outliers > 0:
        logger.warning(f"[OUTLIER] Detected {n_outliers} extreme outliers. Dropping & reporting.")
        outlier_path = os.path.join(output_dir, "duration_outliers_report.csv")
        df[outlier_mask].to_csv(outlier_path, index=False, encoding="utf-8-sig")
        df = df[~outlier_mask].copy()
    return df

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# =============================================================================
# STEP-01: LOAD + CLEAN + SURVIVAL BASE + METADATA
# =============================================================================
def load_fault_data(logger: logging.Logger) -> pd.DataFrame:
    path = DATA_PATHS["fault_data"]
    logger.info(f"[STEP-01] Arıza verisi yükleniyor: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # seçilecek kolonlar
    base_cols = ["cbs_id", "Şebeke Unsuru", "Sebekeye_Baglanma_Tarihi", "started at", "ended at", "duration time", "cause code"]
    extra_cols = [c for c in EXTRA_FAULT_COLS if c in df.columns]

    maint_cols = [
        "Bakım Sayısı", "Son Bakım İş Emri Tarihi", "İlk Bakım İş Emri Tarihi",
        "Son Bakım İş Emri Tipi", "MARKA", "kVA_Rating", "component_voltage", "voltage_level"
    ]
    maint_cols = [c for c in maint_cols if c in df.columns]

    use_cols = [c for c in base_cols if c in df.columns] + extra_cols + maint_cols
    df = df[use_cols].copy()

    rename_map = {
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "cause code": "Ariza_Nedeni",
        "duration time": "Süre_Ham",
    }
    for old in extra_cols:
        if old in COLUMN_MAPPING:
            rename_map[old] = COLUMN_MAPPING[old]
    df.rename(columns=rename_map, inplace=True)

    original = len(df)
    logger.info("[INFO] Orijinal arıza kayıtları: %d", original)

    df = df[df["cbs_id"].notna()].copy()
    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()

    df = rename_maintenance_and_attributes(df, logger)

    # parse
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["started at"] = df["started at"].apply(parse_date_safely)
    df["ended at"] = df["ended at"].apply(parse_date_safely)

    df["Süre_Dakika"] = convert_duration_minutes(df["Süre_Ham"], logger)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])

    # critical missing filter
    before = len(df)
    df = df[
        df["Kurulum_Tarihi"].notna() &
        df["started at"].notna() &
        df["ended at"].notna() &
        df["Süre_Dakika"].notna()
    ].copy()
    logger.info("[FILTER] Missing critical dropped: %d", int(before - len(df)))

    # temporal checks
    ended_minus_started = (df["ended at"] - df["started at"]).dt.total_seconds() / 60.0
    invalid_order = df["ended at"] < df["started at"]
    invalid_before_install = df["started at"] < df["Kurulum_Tarihi"]
    duration_mismatch = (ended_minus_started - df["Süre_Dakika"]).abs() > 5

    issues = invalid_order | invalid_before_install | duration_mismatch
    if issues.sum() > 0:
        out_dir = os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"])
        rep = df[issues].copy()
        rep["temporal_issue"] = np.select(
            [invalid_order.loc[issues], invalid_before_install.loc[issues], duration_mismatch.loc[issues]],
            ["ended<started", "started<install", "duration_mismatch"],
            default="multi"
        )
        rep.to_csv(os.path.join(out_dir, "temporal_issues_report.csv"), index=False, encoding="utf-8-sig")
        df = df[~issues].copy()
        logger.warning("[FILTER] Temporal issues dropped: %d", int(issues.sum()))

    # robust outlier
    out_dir = os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"])
    df = detect_duration_outliers(df, logger, out_dir)

    logger.info("[INFO] Final fault records: %d (%.1f%%)", len(df), 100*len(df)/original)
    return df

def load_healthy_data(logger: logging.Logger) -> pd.DataFrame:
    path = DATA_PATHS["healthy_data"]
    logger.info(f"[STEP-01] Sağlam verisi yükleniyor: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    original = len(df)
    logger.info("[INFO] Orijinal sağlam ekipman kayıtları: %d", original)

    if "cbs_id" not in df.columns:
        if "ID" in df.columns:
            logger.warning("[WARN] Sağlam veri 'ID' kolonunu kullanıyor → 'cbs_id'")
            df = df.rename(columns={"ID": "cbs_id"})
        else:
            raise ValueError("Healthy file must have cbs_id or ID.")

    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()
    df = df.rename(columns={
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
    })
    df = rename_maintenance_and_attributes(df, logger)
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])

    before = len(df)
    df = df[df["Kurulum_Tarihi"].notna() & df["cbs_id"].notna()].copy()
    logger.info("[FILTER] Healthy dropped missing install/id: %d", int(before - len(df)))
    return df

def build_fault_events(df_fault: pd.DataFrame) -> pd.DataFrame:
    return df_fault[[
        "cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi",
        "started at", "ended at", "Süre_Dakika", "Ariza_Nedeni"
    ]].rename(columns={
        "started at": "Ariza_Baslangic_Zamani",
        "ended at": "Ariza_Bitis_Zamani",
        "Süre_Dakika": "Kesinti_Suresi_Dakika",
    })

def aggregate_equipment(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["cbs_id"])

    agg = {
        "Kurulum_Tarihi": ("Kurulum_Tarihi", "min"),
        "Ekipman_Tipi": ("Ekipman_Tipi", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
    }

    if source == "fault":
        agg.update({
            "Fault_Count": ("cbs_id", "size"),
            "Ilk_Ariza_Tarihi": ("started at", "min"),
            "Son_Ariza_Tarihi": ("started at", "max"),
        })
        for col in ["Latitude", "Longitude", "Sehir", "Ilce", "Mahalle"]:
            if col in df.columns:
                agg[col] = (col, "first")
        if "Musteri_Sayisi" in df.columns:
            agg["Musteri_Sayisi"] = ("Musteri_Sayisi", "max")

    opt_cols = [
        "Bakim_Sayisi", "Bakim_Is_Emri_Tipleri", "Ilk_Bakim_Tarihi",
        "Son_Bakim_Tarihi", "Son_Bakim_Tipi", "Son_Bakimdan_Gecen_Gun",
        "Marka", "Gerilim_Seviyesi", "Gerilim_Sinifi", "kVA_Rating"
    ]
    for c in opt_cols:
        if c in df.columns:
            if c in ["Bakim_Sayisi", "Son_Bakimdan_Gecen_Gun", "kVA_Rating", "Gerilim_Seviyesi"]:
                agg[c] = (c, "max")
            elif c == "Ilk_Bakim_Tarihi":
                agg[c] = (c, "min")
            elif c == "Son_Bakim_Tarihi":
                agg[c] = (c, "max")
            else:
                agg[c] = (c, lambda x: x.mode().iloc[0] if not x.mode().empty else (x.dropna().iloc[0] if x.dropna().size > 0 else np.nan))

    grouped = df.groupby("cbs_id").agg(**agg).reset_index()
    return grouped

def build_equipment_master(df_fault: pd.DataFrame, df_healthy: pd.DataFrame, logger: logging.Logger, data_end_date: pd.Timestamp) -> pd.DataFrame:
    fault_part = aggregate_equipment(df_fault, "fault")
    healthy_part = aggregate_equipment(df_healthy, "healthy")

    # add missing cols on healthy
    if not healthy_part.empty:
        for c, default in [("Fault_Count", 0), ("Ilk_Ariza_Tarihi", pd.NaT), ("Son_Ariza_Tarihi", pd.NaT)]:
            if c not in healthy_part.columns:
                healthy_part[c] = default

    all_eq = pd.concat([fault_part, healthy_part], ignore_index=True)
    before = len(all_eq)

    all_eq = all_eq.sort_values(["cbs_id", "Fault_Count", "Kurulum_Tarihi"], ascending=[True, False, True]) \
                   .drop_duplicates("cbs_id", keep="first")

    logger.info("[DEDUP] Dropped duplicate equipment rows: %d", int(before - len(all_eq)))

    # flags
    if "Latitude" in all_eq.columns:
        all_eq["Location_Known"] = all_eq["Latitude"].notna().astype(int)
    else:
        all_eq["Location_Known"] = 0

    if "Musteri_Sayisi" in all_eq.columns:
        all_eq["Musteri_Sayisi"] = all_eq["Musteri_Sayisi"].fillna(-1)

    # age by DATA_END_DATE
    all_eq["Ekipman_Yasi_Gun"] = np.where(
        all_eq["Fault_Count"] > 0,
        (all_eq["Son_Ariza_Tarihi"] - all_eq["Kurulum_Tarihi"]).dt.days,
        (data_end_date - all_eq["Kurulum_Tarihi"]).dt.days
    )
    all_eq["Ekipman_Yasi_Gun"] = pd.to_numeric(all_eq["Ekipman_Yasi_Gun"], errors="coerce").fillna(0).clip(lower=0)
    all_eq["Ariza_Gecmisi"] = (all_eq["Fault_Count"] > 0).astype(int)

    # rare classes -> Diger
    counts = all_eq["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info("[INFO] Nadir sınıflar 'Diger' altına alındı: %s", rare)
        all_eq.loc[all_eq["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Diger"

    return all_eq

def build_survival_base(eq: pd.DataFrame, fault_events: pd.DataFrame, logger: logging.Logger, data_end_date: pd.Timestamp) -> pd.DataFrame:
    first_fail = fault_events.groupby("cbs_id")["Ariza_Baslangic_Zamani"].min()
    eq = eq.merge(first_fail.rename("Ilk_Ariza_Tarihi_2"), on="cbs_id", how="left")

    # unify
    eq["Ilk_Ariza_Tarihi"] = eq["Ilk_Ariza_Tarihi_2"].combine_first(eq.get("Ilk_Ariza_Tarihi"))
    eq = eq.drop(columns=["Ilk_Ariza_Tarihi_2"], errors="ignore")

    eq["event"] = eq["Ilk_Ariza_Tarihi"].notna().astype(int)

    eq["duration_days"] = np.where(
        eq["event"] == 1,
        (eq["Ilk_Ariza_Tarihi"] - eq["Kurulum_Tarihi"]).dt.days,
        (data_end_date - eq["Kurulum_Tarihi"]).dt.days
    )
    eq["duration_days"] = pd.to_numeric(eq["duration_days"], errors="coerce")
    eq = eq[eq["duration_days"].notna() & (eq["duration_days"] > 0)].copy()

    # clip >60y
    extreme_threshold = 60 * 365
    too_long = (eq["duration_days"] > extreme_threshold).sum()
    if too_long:
        out_dir = os.path.dirname(INTERMEDIATE_PATHS["survival_base"])
        rep = eq[eq["duration_days"] > extreme_threshold].copy()
        rep["duration_years"] = rep["duration_days"] / 365.25
        rep.to_csv(os.path.join(out_dir, "extreme_duration_report.csv"), index=False, encoding="utf-8-sig")
        eq["duration_days"] = eq["duration_days"].clip(upper=extreme_threshold)
        logger.warning("[WARN] %d kayıt 60y+ -> clipped & reported", int(too_long))

    logger.info("[SURVIVAL] Final survival base: %d | Event rate: %.2f%% (%d)",
                len(eq), 100*eq["event"].mean(), int(eq["event"].sum()))
    return eq

def save_metadata(logger: logging.Logger, data_end_date: pd.Timestamp, equipment_master: pd.DataFrame, survival_base: pd.DataFrame):
    meta = {
        "data_end_date": str(pd.to_datetime(data_end_date).date()),
        "analysis_date": str(pd.to_datetime(data_end_date).date()),
        "total_assets": int(len(equipment_master)),
        "total_events": int(survival_base["event"].sum()),
    }
    os.makedirs(os.path.dirname(ANALYSIS_METADATA_PATH), exist_ok=True)
    with open(ANALYSIS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    logger.info("[META] Analysis metadata saved → %s", ANALYSIS_METADATA_PATH)

def load_metadata(logger: logging.Logger) -> pd.Timestamp:
    if not os.path.exists(ANALYSIS_METADATA_PATH):
        raise FileNotFoundError(ANALYSIS_METADATA_PATH)
    with open(ANALYSIS_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    t_ref = pd.to_datetime(meta["data_end_date"])
    logger.info("[INFO] Using T_ref (DATA_END_DATE): %s", t_ref.date())
    return t_ref


# =============================================================================
# STEP-04: CHRONIC (Bayesian MTBF + Decay + Trend)
# =============================================================================
def compute_chronic_features(fault_events_clean: pd.DataFrame, t_ref: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    """
    fault_events_clean columns:
    - cbs_id
    - Ariza_Baslangic_Zamani
    """
    if fault_events_clean.empty:
        return pd.DataFrame(columns=["cbs_id"])

    fe = fault_events_clean.copy()
    fe["cbs_id"] = fe["cbs_id"].astype(str).str.lower().str.strip()
    fe["Ariza_Baslangic_Zamani"] = pd.to_datetime(fe["Ariza_Baslangic_Zamani"], errors="coerce")
    fe = fe[fe["Ariza_Baslangic_Zamani"].notna()].copy()

    # window
    window_start = t_ref - pd.Timedelta(days=int(CHRONIC_WINDOW_DAYS))
    fe_w = fe[fe["Ariza_Baslangic_Zamani"] >= window_start].copy()
    logger.info("[CHRONIC] Window: last %d days since %s", int(CHRONIC_WINDOW_DAYS), window_start.date())

    # counts in window
    counts_w = fe_w.groupby("cbs_id").size().rename("Ariza_Sayisi_%dg" % int(CHRONIC_WINDOW_DAYS))

    # exponential decay score (newer faults heavier)
    # score = sum(exp(-lambda * age_days))
    lam = 0.05  # ~20d half-ish scale (tuneable)
    age_days = (t_ref - fe_w["Ariza_Baslangic_Zamani"]).dt.days.clip(lower=0)
    fe_w["decay_w"] = np.exp(-lam * age_days)
    decay_score = fe_w.groupby("cbs_id")["decay_w"].sum().rename("Chronic_Decay_Skoru")

    # trend: faults per 30d bins in window -> slope
    fe_w["bin"] = ((fe_w["Ariza_Baslangic_Zamani"] - window_start).dt.days // 30).astype(int)
    bin_counts = fe_w.groupby(["cbs_id", "bin"]).size().reset_index(name="cnt")

    def slope_for_asset(g):
        if g.shape[0] < 2:
            return 0.0
        x = g["bin"].values.astype(float)
        y = g["cnt"].values.astype(float)
        # simple linear regression slope
        x = x - x.mean()
        denom = (x**2).sum()
        if denom == 0:
            return 0.0
        return float((x*y).sum() / denom)

    trend = bin_counts.groupby("cbs_id").apply(slope_for_asset).rename("Chronic_Trend_Slope")

    # Bayesian MTBF (global prior)
    # For each asset: smoothed MTBF ≈ (C + sum(gaps)) / (alpha + n_gaps)
    # We'll approximate from inter-fault gaps on FULL history
    fe_sorted = fe.sort_values(["cbs_id", "Ariza_Baslangic_Zamani"])
    fe_sorted["prev"] = fe_sorted.groupby("cbs_id")["Ariza_Baslangic_Zamani"].shift(1)
    fe_sorted["gap_days"] = (fe_sorted["Ariza_Baslangic_Zamani"] - fe_sorted["prev"]).dt.days
    gaps = fe_sorted.dropna(subset=["gap_days"])
    global_prior = float(gaps["gap_days"].median()) if not gaps.empty else 365.0
    logger.info("[CHRONIC] Global prior MTBF (median gap): %.1f days", global_prior)

    gap_sum = gaps.groupby("cbs_id")["gap_days"].sum().rename("Gap_Sum_Days")
    gap_n = gaps.groupby("cbs_id")["gap_days"].size().rename("Gap_N")

    alpha = 1.0
    C = global_prior
    mtbf_bayes = ((C + gap_sum) / (alpha + gap_n)).rename("MTBF_Bayes_Gun")

    # chronic flag by IEEE-ish: count>=threshold OR rate>=min_rate
    # rate per year in window:
    rate = (counts_w / (CHRONIC_WINDOW_DAYS / 365.25)).rename("Chronic_Rate_Yillik")
    chronic_flag = ((counts_w >= CHRONIC_THRESHOLD_EVENTS) | (rate >= CHRONIC_MIN_RATE)).astype(int).rename("Chronic_Flag")

    out = pd.concat([counts_w, rate, decay_score, trend, mtbf_bayes, chronic_flag], axis=1).reset_index()
    out = out.rename(columns={"index": "cbs_id"})
    out["cbs_id"] = out["cbs_id"].astype(str).str.lower().str.strip()
    return out


# =============================================================================
# STEP-02a: STRUCTURAL FEATURES (Leakage-safe)
# =============================================================================
def step02a_yapisal(equipment_master: pd.DataFrame, t_ref: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    df = equipment_master.copy()

    # Structural set: no fault-history used for survival.
    keep_cols = ["cbs_id", "Ekipman_Tipi", "Gerilim_Sinifi", "Gerilim_Seviyesi", "Marka", "kVA_Rating",
                 "Sehir", "Ilce", "Mahalle", "Location_Known", "Musteri_Sayisi"]

    cols = [c for c in keep_cols if c in df.columns]
    out = df[cols].drop_duplicates("cbs_id").copy()
    out["T_ref"] = str(pd.to_datetime(t_ref).date())

    # Basic sanitization
    out["cbs_id"] = out["cbs_id"].astype(str).str.lower().str.strip()
    for c in ["Gerilim_Seviyesi", "kVA_Rating", "Musteri_Sayisi"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            
    # Fill missing kVA_Rating based on equipment type
    if "kVA_Rating" in out.columns:
        out.loc[(out["kVA_Rating"].isna()) & (out["Ekipman_Tipi"].str.contains("Trafo", na=False)), "kVA_Rating"] = 400.0
        out.loc[(out["kVA_Rating"].isna()) & (out["Ekipman_Tipi"].str.contains("Pano", na=False)), "kVA_Rating"] = 50.0
    
    # Log what we have
    logger.info(f"[STRUCTURAL] Created features with {len(out.columns)} columns for {len(out)} assets")
    for col in out.columns:
        if col not in ["cbs_id", "T_ref"]:
            if out[col].dtype in [np.float64, np.int64]:
                logger.info(f"  - {col}: numeric, {out[col].notna().sum()}/{len(out)} non-null")
            else:
                logger.info(f"  - {col}: categorical, {out[col].nunique()} unique values")

    return out

# =============================================================================
# STEP-02b: TEMPORAL/SNAPSHOT FEATURES (T_ref based)
# =============================================================================
def step02b_zamansal(equipment_master: pd.DataFrame, chronic_df: pd.DataFrame, t_ref: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    df = equipment_master[["cbs_id", "Kurulum_Tarihi", "Son_Bakim_Tarihi", "Son_Bakimdan_Gecen_Gun"]].copy() \
        if "Kurulum_Tarihi" in equipment_master.columns else equipment_master[["cbs_id"]].copy()

    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()

    if "Kurulum_Tarihi" in df.columns:
        df["Kurulum_Tarihi"] = pd.to_datetime(df["Kurulum_Tarihi"], errors="coerce")
        df["Tref_Yas_Gun"] = (t_ref - df["Kurulum_Tarihi"]).dt.days
        df["Tref_Yas_Gun"] = pd.to_numeric(df["Tref_Yas_Gun"], errors="coerce").fillna(0).clip(lower=0)

    # seasonality snapshot
    df["Tref_Ay"] = int(t_ref.month)
    df["Tref_Mevsim"] = pd.cut(
        [t_ref.month],
        bins=[0, 3, 6, 9, 12],
        labels=["Kis", "Ilkbahar", "Yaz", "Sonbahar"],
        include_lowest=True
    )[0]

    # maintenance recency
    if "Son_Bakim_Tarihi" in df.columns:
        df["Son_Bakim_Tarihi"] = pd.to_datetime(df["Son_Bakim_Tarihi"], errors="coerce")
        df["Tref_SonBakim_Gun"] = (t_ref - df["Son_Bakim_Tarihi"]).dt.days
        df["Tref_SonBakim_Gun"] = pd.to_numeric(df["Tref_SonBakim_Gun"], errors="coerce")

    if "Son_Bakimdan_Gecen_Gun" in df.columns:
        # eğer bu kolon zaten doğruysa kullan; değilse Tref_SonBakim_Gun öncelikli olacak
        df["Son_Bakimdan_Gecen_Gun"] = pd.to_numeric(df["Son_Bakimdan_Gecen_Gun"], errors="coerce")

    # join chronic
    if chronic_df is not None and not chronic_df.empty:
        chronic_df = chronic_df.copy()
        chronic_df["cbs_id"] = chronic_df["cbs_id"].astype(str).str.lower().str.strip()
        df = df.merge(chronic_df, on="cbs_id", how="left")

    df["T_ref"] = str(pd.to_datetime(t_ref).date())
    return df


# =============================================================================
# STEP-03: MODELS (Survival + ML)
# =============================================================================
def select_cox_safe_features(df_all: pd.DataFrame, structural_cols: list, logger: logging.Logger) -> pd.DataFrame:
    X = df_all[[c for c in structural_cols if c in df_all.columns]].copy()

    BLACKLIST = {
        "Fault_Count", "Ariza_Gecmisi", "Ilk_Ariza_Tarihi", "Son_Ariza_Tarihi",
        "event", "duration_days",
        "Chronic_Flag", "Chronic_Decay_Skoru", "MTBF_Bayes_Gun", "Chronic_Trend_Slope",
        "Tref_Yas_Gun", "Tref_SonBakim_Gun",
    }
    X = X.drop(columns=[c for c in X.columns if c in BLACKLIST], errors="ignore")

    logger.info(f"[COX] Starting with {len(X.columns)} structural columns")
    
    # Separate numeric and categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    
    logger.info(f"[COX] Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
    
    # For categorical: one-hot encode with limited cardinality
    if categorical_cols:
        for col in categorical_cols[:]:  # Copy list to modify during iteration
            n_unique = X[col].nunique()
            if n_unique > 10:  # Too many categories
                logger.warning(f"[COX] Dropping {col}: too many categories ({n_unique})")
                categorical_cols.remove(col)
                X = X.drop(columns=[col])
            elif n_unique == 1:  # No variance
                logger.warning(f"[COX] Dropping {col}: only one unique value")
                categorical_cols.remove(col)
                X = X.drop(columns=[col])
    
    # One-hot encode remaining categoricals
    if categorical_cols:
        logger.info(f"[COX] One-hot encoding {len(categorical_cols)} categorical columns")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
    
    # Now work with numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        logger.error("[COX] No numeric columns after encoding!")
        raise ValueError("No Cox-safe features left after initial processing.")
    
    # Clean numeric columns
    bad_cols = []
    for col in numeric_cols:
        # Check for NaN/Inf
        if X[col].isna().any() or np.isinf(X[col]).any():
            # Try to fill
            if X[col].isna().sum() < len(X) * 0.5:  # Less than 50% missing
                X[col] = X[col].fillna(X[col].median())
            else:
                bad_cols.append(col)
                continue
        
        # Check variance
        if X[col].var() < 1e-6:
            bad_cols.append(col)
    
    if bad_cols:
        logger.warning(f"[COX] Dropping {len(bad_cols)} columns: NaN/Inf/zero-variance")
        X = X.drop(columns=bad_cols)

    # Final check
    remaining = X.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"[COX] Final feature set: {len(remaining)} columns")
    
    if len(remaining) == 0:
        logger.error("[COX] All features filtered out. Available structural_cols were:")
        logger.error(f"  {structural_cols[:10]}...")  # Show first 10
        raise ValueError("No Cox-safe features left after all filters.")
    
    return X
def conditional_pof_from_survival(S_age: np.ndarray, S_age_h: np.ndarray) -> np.ndarray:
    eps = 1e-12
    ratio = (S_age_h + eps) / (S_age + eps)
    return 1.0 - np.clip(ratio, 0.0, 1.0)

def train_cox_weibull(X_transformed: pd.DataFrame, duration: pd.Series, event: pd.Series, logger: logging.Logger):
    """
    Train Cox and Weibull models on pre-transformed features.
    
    Args:
        X_transformed: Already one-hot encoded feature matrix
        duration: duration_days series
        event: event indicator series
        logger: logger instance
    """
    if not LIFELINES_OK:
        logger.warning("[SKIP] lifelines yok -> Cox/Weibull atlandı")
        return None, None

    work = X_transformed.copy()
    work["duration_days"] = duration.values
    work["event"] = event.values

    # Ensure all numeric
    work = work.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    train_idx, test_idx = train_test_split(
        np.arange(len(work)), test_size=0.25, random_state=42, stratify=event.values
    )
    train = work.iloc[train_idx].copy()
    test = work.iloc[test_idx].copy()

    cox = None
    wb = None

    try:
        cox = CoxPHFitter(penalizer=0.05)
        logger.info("[COX] Training...")
        cox.fit(train, duration_col="duration_days", event_col="event")
        test_scores = cox.predict_partial_hazard(test)
        c_ind = concordance_index(test["duration_days"], -test_scores, test["event"])
        logger.info("[COX] Test Concordance: %.4f", c_ind)
    except Exception as e:
        logger.error("[COX] Failed: %s", e)
        cox = None

    try:
        wb = WeibullAFTFitter(penalizer=0.05)
        logger.info("[WEIBULL] Training...")
        wb.fit(train, duration_col="duration_days", event_col="event")
        wb_pred = wb.predict_median(test)
        wb_cind = concordance_index(test["duration_days"], wb_pred, test["event"])
        logger.info("[WEIBULL] Test Concordance: %.4f", wb_cind)
    except Exception as e:
        logger.error("[WEIBULL] Failed: %s", e)
        wb = None

    return cox, wb
def predict_lifelines_conditional_pof(X_transformed: pd.DataFrame, duration: pd.Series, model, horizons_days: list, model_name: str, cbs_ids: pd.Series) -> pd.DataFrame:
    """
    Predict conditional PoF using pre-transformed features.
    """
    Xd = X_transformed.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # conditional PoF based on duration_days (observed survival time)
    age = duration.fillna(0).clip(lower=0).values.astype(float)

    out = pd.DataFrame({"cbs_id": cbs_ids.values})
    for H in horizons_days:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")

        times = np.unique(np.concatenate([age, age + H]))
        times = times[times >= 0]

        sf = model.predict_survival_function(Xd, times=times)

        # vectorized mapping
        time_to_idx = {t: i for i, t in enumerate(times)}
        S_age = np.array([sf.iloc[time_to_idx[a], j] for j, a in enumerate(age)])
        S_age_h = np.array([sf.iloc[time_to_idx[a], j] for j, a in enumerate(age + H)])

        out[f"{model_name}_pof_{label}"] = conditional_pof_from_survival(S_age, S_age_h)
    return out

def train_rsf_survival(df_all: pd.DataFrame, structural_cols: list, logger: logging.Logger):
    if not SKSURV_OK:
        logger.warning("[SKIP] sksurv yok -> RSF atlandı")
        return None

    cols = [c for c in structural_cols if c in df_all.columns]
    X = df_all[cols].copy()
    pre = build_preprocessor(X)

    y = Surv.from_arrays(
        event=df_all["event"].astype(bool).values,
        time=df_all["duration_days"].values,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=df_all["event"].values
    )

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1
    )

    pipe = Pipeline([("pre", pre), ("rsf", rsf)])
    logger.info("[RSF] Training Survival Forest...")
    pipe.fit(X_train, y_train)

    risk = pipe.predict(X_test)
    ci = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
    logger.info("[RSF] Test Concordance: %.4f", ci)
    return pipe

def predict_rsf_conditional_pof(df_all: pd.DataFrame, rsf_pipe, structural_cols: list, horizons_days: list) -> pd.DataFrame:
    cols = [c for c in structural_cols if c in df_all.columns]
    X = df_all[cols].copy()

    age = df_all["duration_days"].fillna(0).clip(lower=0).values.astype(float)

    X_tr = rsf_pipe.named_steps["pre"].transform(X)
    sfs = rsf_pipe.named_steps["rsf"].predict_survival_function(X_tr, return_array=False)

    out = pd.DataFrame({"cbs_id": df_all["cbs_id"].values})
    for H in horizons_days:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        pofs = []
        for i, sf in enumerate(sfs):
            t = sf.x
            s = sf.y
            a = age[i]
            b = age[i] + H
            S_a = np.interp(a, t, s, left=s[0], right=s[-1])
            S_b = np.interp(b, t, s, left=s[0], right=s[-1])
            pofs.append(conditional_pof_from_survival(np.array([S_a]), np.array([S_b]))[0])
        out[f"rsf_pof_{label}"] = np.array(pofs)
    return out

# -------------------------
# ML targets
# -------------------------
def make_targets_from_install(df_all: pd.DataFrame, horizons_days: list) -> pd.DataFrame:
    y = pd.DataFrame({"cbs_id": df_all["cbs_id"].values})
    for H in horizons_days:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        y[f"y_ilk_ariza_{label}"] = ((df_all["event"] == 1) & (df_all["duration_days"] <= H)).astype(int)
    return y

def train_ml_from_install(df_all: pd.DataFrame, feature_cols: list, horizons_days: list, logger: logging.Logger) -> dict:
    """
    MODE_A: from_install
    Leakage control: no duration_days, no event, no fault history.
    """
    blacklist = {
        "event", "duration_days",
        "Fault_Count", "Ariza_Gecmisi", "Ilk_Ariza_Tarihi", "Son_Ariza_Tarihi",
    }
    safe_cols = [c for c in feature_cols if c in df_all.columns and c not in blacklist]
    X = df_all[safe_cols].copy()

    # Fill NaN in categorical columns to ensure consistency
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown')

    ydf = make_targets_from_install(df_all, horizons_days)

    models = {}  # (model_name, horizon_label) -> fitted pipeline
    perf = []

    for H in horizons_days:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        y = ydf[f"y_ilk_ariza_{label}"].values
        pos = int(y.sum())
        if pos < 30:
            logger.warning("[ML] Horizon %s skipped (positives=%d)", label, pos)
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Build preprocessor for each horizon to ensure consistency
        pre = build_preprocessor(X)

        if XGB_OK:
            xgb = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=42, n_jobs=1, eval_metric="auc"
            )
            pipe = Pipeline([("pre", pre), ("mdl", xgb)])
            pipe.fit(X_train, y_train)
            p = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, p)
            logger.info("  > [ML-XGB from_install] %s AUC: %.4f", label, auc)
            models[("xgb", label)] = pipe
            perf.append(("xgb", label, auc))

        if CAT_OK:
            cat = CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.05,
                loss_function="Logloss", verbose=False, random_seed=42
            )
            pipe = Pipeline([("pre", pre), ("mdl", cat)])
            pipe.fit(X_train, y_train)
            p = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, p)
            logger.info("  > [ML-CAT from_install] %s AUC: %.4f", label, auc)
            models[("cat", label)] = pipe
            perf.append(("cat", label, auc))

    return {"models": models, "perf": perf, "safe_cols": safe_cols}

def predict_ml_from_install(df_all: pd.DataFrame, ml_pack: dict, horizons_days: list) -> pd.DataFrame:
    # Ensure all required columns exist with proper types
    X = df_all[ml_pack["safe_cols"]].copy()

    # Ensure consistent column types as during training
    for col in X.columns:
        if X[col].dtype == 'object' and col != 'cbs_id':
            # Fill NaN in categorical columns
            X[col] = X[col].fillna('Unknown')
        elif pd.api.types.is_numeric_dtype(X[col]):
            # Keep numeric as-is (preprocessor will handle imputation)
            pass

    out = pd.DataFrame({"cbs_id": df_all["cbs_id"].values})

    for H in horizons_days:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")

        preds = []
        if ("xgb", label) in ml_pack["models"]:
            p = ml_pack["models"][("xgb", label)].predict_proba(X)[:, 1]
            preds.append(p)
        if ("cat", label) in ml_pack["models"]:
            p = ml_pack["models"][("cat", label)].predict_proba(X)[:, 1]
            preds.append(p)

        if preds:
            out[f"ml_pof_{label}"] = np.mean(np.vstack(preds), axis=0)
    return out


# =============================================================================
# STEP-05: RISK (PoF×CoF)
# =============================================================================
def categorize_pof(p: float) -> str:
    if p <= POF_THRESHOLDS["Low"]:
        return "Low"
    if p <= POF_THRESHOLDS["Medium"]:
        return "Medium"
    if p <= POF_THRESHOLDS["High"]:
        return "High"
    return "Very High"

def compute_cof(row: pd.Series) -> str:
    """
    Basit CoF:
    - Musteri_Sayisi varsa kullan
    - yoksa Location_Known + Gerilim_Sinifi proxy
    """
    cust = row.get("Musteri_Sayisi", -1)
    volt = str(row.get("Gerilim_Sinifi", "")).upper()
    known = int(row.get("Location_Known", 0))

    # müşteri sayısı varsa daha güçlü
    if pd.notna(cust) and float(cust) >= 0:
        c = float(cust)
        if c < 50:
            return "Low"
        if c < 200:
            return "Medium"
        if c < 1000:
            return "High"
        return "Critical"

    # proxy
    if known == 0:
        return "Medium"
    if "OG" in volt:
        return "High"
    return "Medium"



# =============================================================================
# ENSEMBLE SCORE
# =============================================================================
def make_health_score(df_pred: pd.DataFrame) -> pd.DataFrame:
    out = df_pred.copy()
    pof_cols = [c for c in out.columns if "_pof_" in c]
    if not pof_cols:
        out["Mean_PoF"] = 0.0
        out["Health_Score"] = 100.0
        return out
    out["Mean_PoF"] = out[pof_cols].mean(axis=1)
    out["Health_Score"] = (100.0 * (1.0 - out["Mean_PoF"])).clip(0, 100)
    return out


# =============================================================================
# MAIN (01→05)
# =============================================================================
def main():
    ensure_dirs()
    logger = setup_logger()

    # -------------------------
    # STEP-01
    # -------------------------
    df_fault = load_fault_data(logger)
    df_healthy = load_healthy_data(logger)

    # Data range detection
    data_start = df_fault["started at"].min()
    data_end = df_fault["started at"].max()
    logger.info("[DATA RANGE] DATA_START_DATE = %s", pd.to_datetime(data_start).date())
    logger.info("[DATA RANGE] DATA_END_DATE   = %s", pd.to_datetime(data_end).date())
    t_ref = pd.to_datetime(data_end)

    # observability: drop healthy installed after data_end
    if REQUIRE_INSTALL_BEFORE_DATA_END and "Kurulum_Tarihi" in df_healthy.columns:
        before = len(df_healthy)
        df_healthy = df_healthy[df_healthy["Kurulum_Tarihi"] <= t_ref].copy()
        logger.info("[OBSERVABILITY] Dropped %d healthy assets installed after DATA_END_DATE", int(before - len(df_healthy)))

    fault_events = build_fault_events(df_fault)
    equipment_master = build_equipment_master(df_fault, df_healthy, logger, t_ref)
    survival_base = build_survival_base(equipment_master, fault_events, logger, t_ref)

    # persist step-01 outputs
    fault_events.to_csv(INTERMEDIATE_PATHS["fault_events_clean"], index=False, encoding="utf-8-sig")
    # To check if key exists first:
    if "healthy_equipment_clean" in INTERMEDIATE_PATHS:
        df_healthy.to_csv(INTERMEDIATE_PATHS["healthy_equipment_clean"], index=False, encoding="utf-8-sig")
    equipment_master.to_csv(INTERMEDIATE_PATHS["equipment_master"], index=False, encoding="utf-8-sig")
    survival_base.to_csv(INTERMEDIATE_PATHS["survival_base"], index=False, encoding="utf-8-sig")

    fault_events.to_csv(OUTPUT_PATHS["ariza_kayitlari"], index=False, encoding="utf-8-sig")
    equipment_master.to_csv(OUTPUT_PATHS["ekipman_listesi"], index=False, encoding="utf-8-sig")
    survival_base.to_csv(OUTPUT_PATHS["sagkalim_taban"], index=False, encoding="utf-8-sig")
    df_healthy.to_csv(OUTPUT_PATHS["saglam_ekipman_listesi"], index=False, encoding="utf-8-sig")

    save_metadata(logger, t_ref, equipment_master, survival_base)

    # -------------------------
    # STEP-04 (Chronic)  (02b için önce lazım)
    # -------------------------
    chronic_df = compute_chronic_features(fault_events, t_ref, logger)
    chronic_out_dir = os.path.dirname(OUTPUT_PATHS["chronic_summary"])
    os.makedirs(chronic_out_dir, exist_ok=True)
    chronic_df.to_csv(OUTPUT_PATHS["chronic_summary"], index=False, encoding="utf-8-sig")
    chronic_df[chronic_df.get("Chronic_Flag", 0) == 1].to_csv(OUTPUT_PATHS["chronic_only"], index=False, encoding="utf-8-sig")

    # -------------------------
    # STEP-02a / 02b
    # -------------------------
    x_struct = step02a_yapisal(equipment_master, t_ref, logger)
    x_temp = step02b_zamansal(equipment_master, chronic_df, t_ref, logger)

    # intermediate persist (tek script ama debug için alt çıktı)
    struct_path = os.path.join(os.path.dirname(INTERMEDIATE_PATHS["ozellikler_pof3"]), "ozellikler_yapisal.csv")
    temp_path = os.path.join(os.path.dirname(INTERMEDIATE_PATHS["ozellikler_pof3"]), "ozellikler_zamansal.csv")
    x_struct.to_csv(struct_path, index=False, encoding="utf-8-sig")
    x_temp.to_csv(temp_path, index=False, encoding="utf-8-sig")

    # combined feature table (customer neutral)
    features_all = x_struct.merge(x_temp, on="cbs_id", how="left", suffixes=("", "_temp"))
    features_all.to_csv(INTERMEDIATE_PATHS["ozellikler_pof3"], index=False, encoding="utf-8-sig")
    logger.info("[STEP-02] Features saved: %s", INTERMEDIATE_PATHS["ozellikler_pof3"])

    # -------------------------
    # STEP-03 (Models)
    # -------------------------
    # merge survival + features
    df_all = survival_base.copy()
    df_all["cbs_id"] = df_all["cbs_id"].astype(str).str.lower().str.strip()
    features_all["cbs_id"] = features_all["cbs_id"].astype(str).str.lower().str.strip()

    # Drop overlapping columns from survival_base before merge (keep only survival-specific)
    survival_cols_to_keep = ["cbs_id", "event", "duration_days", "Kurulum_Tarihi", 
                            "Fault_Count", "Ilk_Ariza_Tarihi", "Son_Ariza_Tarihi",
                            "Ekipman_Yasi_Gun", "Ariza_Gecmisi"]
    survival_cols_to_keep = [c for c in survival_cols_to_keep if c in df_all.columns]
    df_all = df_all[survival_cols_to_keep]

    # Now merge - no more _x and _y suffixes!
    df_all = df_all.merge(features_all, on="cbs_id", how="left")

    logger.info(f"[DEBUG] After clean merge, df_all columns: {list(df_all.columns[:20])}...")

    # define scopes
    if "T_ref" in x_struct.columns:
        x_struct = x_struct.drop(columns=["T_ref"])
    if "T_ref" in x_temp.columns:
        x_temp = x_temp.drop(columns=["T_ref"])

    structural_cols = [c for c in x_struct.columns if c not in ["cbs_id"]]
    temporal_cols = [c for c in x_temp.columns if c not in ["cbs_id", "Kurulum_Tarihi"]]

    logger.info(f"[DEBUG] x_struct columns: {list(x_struct.columns)}")
    logger.info(f"[DEBUG] x_temp columns: {list(x_temp.columns)}")
    logger.info(f"[DEBUG] structural_cols: {structural_cols}")
    logger.info(f"[DEBUG] temporal_cols: {temporal_cols}")

    preds = pd.DataFrame({"cbs_id": df_all["cbs_id"].values})

    # Survival: Cox/Weibull using strict Cox-safe numeric
# Survival: Cox/Weibull using strict Cox-safe numeric
    if LIFELINES_OK:
        X_cox = select_cox_safe_features(df_all, structural_cols, logger)
        
        # Train with transformed features
        cox, wb = train_cox_weibull(X_cox, df_all["duration_days"], df_all["event"], logger)

        if cox is not None:
            cox_pred = predict_lifelines_conditional_pof(
                X_cox, df_all["duration_days"], cox, 
                SURVIVAL_HORIZONS_DAYS, "cox", df_all["cbs_id"]
            )
            preds = preds.merge(cox_pred, on="cbs_id", how="left")
            # Save individual horizon files...
            
        if wb is not None:
            wb_pred = predict_lifelines_conditional_pof(
                X_cox, df_all["duration_days"], wb,
                SURVIVAL_HORIZONS_DAYS, "weibull", df_all["cbs_id"]
            )
            preds = preds.merge(wb_pred, on="cbs_id", how="left")
        # ... rest of cox code
    # Around line 1330 in main(), add these debug lines:
    logger.info(f"[DEBUG] x_struct columns: {list(x_struct.columns)}")
    logger.info(f"[DEBUG] x_temp columns: {list(x_temp.columns)}")
    logger.info(f"[DEBUG] structural_cols: {structural_cols}")
    logger.info(f"[DEBUG] df_all columns: {list(df_all.columns)}")
    # Around line 1327-1330, replace with:
    # define scopes
    if "T_ref" in x_struct.columns:
        x_struct = x_struct.drop(columns=["T_ref"])
    if "T_ref" in x_temp.columns:
        x_temp = x_temp.drop(columns=["T_ref"])

    structural_cols = [c for c in x_struct.columns if c not in ["cbs_id"]]
    temporal_cols = [c for c in x_temp.columns if c not in ["cbs_id", "Kurulum_Tarihi"]]

    logger.info(f"[DEBUG] Structural features: {structural_cols}")
    logger.info(f"[DEBUG] Temporal features: {temporal_cols}")

    preds = pd.DataFrame({"cbs_id": df_all["cbs_id"].values})

    # Survival: Cox/Weibull using strict Cox-safe numeric
    if LIFELINES_OK:
        X_cox = select_cox_safe_features(df_all, structural_cols, logger)
        cox, wb = train_cox_weibull(X_cox, df_all["duration_days"], df_all["event"], logger)

        if cox is not None:
            cox_pred = predict_lifelines_conditional_pof(
                X_cox, df_all["duration_days"], cox,
                SURVIVAL_HORIZONS_DAYS, "cox", df_all["cbs_id"]
            )
            preds = preds.merge(cox_pred, on="cbs_id", how="left")
            for H in SURVIVAL_HORIZONS_DAYS:
                label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
                key = f"cox_{label}"
                col = f"cox_pof_{label}"
                if key in OUTPUT_PATHS and col in cox_pred.columns:
                    cox_pred[["cbs_id", col]].rename(columns={col: "PoF"}).to_csv(OUTPUT_PATHS[key], index=False, encoding="utf-8-sig")

        if wb is not None:
            wb_pred = predict_lifelines_conditional_pof(
                X_cox, df_all["duration_days"], wb,
                SURVIVAL_HORIZONS_DAYS, "weibull", df_all["cbs_id"]
            )
            preds = preds.merge(wb_pred, on="cbs_id", how="left")

    else:
        logger.warning("[SKIP] lifelines yok -> Cox/Weibull atlandı")

    # RSF survival
    if SKSURV_OK:
        rsf_pipe = train_rsf_survival(df_all, structural_cols, logger)
        if rsf_pipe is not None:
            rsf_pred = predict_rsf_conditional_pof(df_all, rsf_pipe, structural_cols, SURVIVAL_HORIZONS_DAYS)
            preds = preds.merge(rsf_pred, on="cbs_id", how="left")
            for H in SURVIVAL_HORIZONS_DAYS:
                label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
                key = f"rsf_{label}"
                col = f"rsf_pof_{label}"
                if key in OUTPUT_PATHS and col in rsf_pred.columns:
                    rsf_pred[["cbs_id", col]].rename(columns={col: "PoF"}).to_csv(OUTPUT_PATHS[key], index=False, encoding="utf-8-sig")
    else:
        logger.warning("[SKIP] sksurv yok -> RSF atlandı")

    # ML MODE_A: from_install (structural + safe temporal snapshot)
    logger.info("=" * 60)
    logger.info("[ML] MODE_A = from_install (leakage-controlled)")
    logger.info("=" * 60)

    # ML features: structural + very safe snapshot (month/season + chronic summary allowed here)
    # (Survival’da chronic yoktu; ML’de chronic = OK çünkü hedef “fail within H” classification.)
    ml_feature_cols = []
    ml_feature_cols += [c for c in structural_cols if c in df_all.columns]
    ml_feature_cols += [c for c in ["Tref_Ay", "Tref_Mevsim", "Tref_SonBakim_Gun",
                                    "Ariza_Sayisi_%dg" % int(CHRONIC_WINDOW_DAYS),
                                    "Chronic_Decay_Skoru", "Chronic_Trend_Slope", "MTBF_Bayes_Gun", "Chronic_Flag"]
                        if c in df_all.columns]

    ml_pack = train_ml_from_install(df_all, ml_feature_cols, SURVIVAL_HORIZONS_DAYS, logger)
    ml_pred = predict_ml_from_install(df_all, ml_pack, SURVIVAL_HORIZONS_DAYS)
    preds = preds.merge(ml_pred, on="cbs_id", how="left")

    # -------------------------
    # STEP-05: Ensemble + Risk
    # -------------------------
    scored = make_health_score(preds)

    # attach reporting columns
    report_cols = [c for c in [
        "Ekipman_Tipi", "Sehir", "Ilce", "Mahalle", "Gerilim_Sinifi",
        "Ekipman_Yasi_Gun", "Fault_Count", "Ariza_Gecmisi",
        "Location_Known", "Musteri_Sayisi"
    ] if c in df_all.columns]

    report = df_all[["cbs_id"] + report_cols].drop_duplicates("cbs_id").merge(scored, on="cbs_id", how="left")


    out_path = os.path.join(os.path.dirname(list(OUTPUT_PATHS.values())[0]), "ensemble_pof_final.csv")
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[SUCCESS] Saved Ensemble Output → %s", out_path)

    # summary
    crit = int((report["Health_Score"] < 40).sum()) if "Health_Score" in report.columns else 0
    mean_h = float(report["Health_Score"].mean()) if "Health_Score" in report.columns else 100.0
    logger.info("[ENSEMBLE] Critical Assets (Score < 40): %d", crit)
    logger.info("[ENSEMBLE] Mean Health Score: %.1f", mean_h)
    logger.info("[DONE] Pipeline Finished Successfully.")


if __name__ == "__main__":
    main()
