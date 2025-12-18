# -*- coding: utf-8 -*-
"""
PoF3 - Clean Production Pipeline | Temporal Validation + Equipment Stratification
==================================================================================
Single script: Data Loading → Feature Engineering → Survival Models → Risk Assessment
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
# Soft dependencies
try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index
    LIFELINES_OK = True
except ImportError:
    LIFELINES_OK = False

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    SKSURV_OK = True
except ImportError:
    SKSURV_OK = False

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from catboost import CatBoostClassifier
    CAT_OK = True
except ImportError:
    CAT_OK = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["base"])
INPUT_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["input"])
INTERMEDIATE_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["intermediate"])
OUTPUT_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["output"])
LOG_DIR = os.path.join(BASE_DIR, CFG["paths"]["data"]["logs"])

DATA_PATHS = {k: os.path.join(BASE_DIR, v) for k, v in CFG["data_paths"].items()}
INTERMEDIATE_PATHS = {k: os.path.join(INTERMEDIATE_DIR, v) for k, v in CFG["intermediate_paths"].items()}
OUTPUT_PATHS = {k: os.path.join(OUTPUT_DIR, v) for k, v in CFG["output_paths"].items()}

SURVIVAL_HORIZONS_DAYS = CFG["survival"]["horizons_days"]
SURVIVAL_HORIZON_LABELS = CFG["survival"]["horizon_labels"]
MIN_EQUIPMENT_PER_CLASS = CFG["analysis"]["min_equipment_per_class"]
ANALYSIS_METADATA_PATH = os.path.join(BASE_DIR, CFG["analysis"]["analysis_metadata_path"])

CHRONIC_CFG = CFG.get("chronic", {})
CHRONIC_WINDOW_DAYS = CHRONIC_CFG.get("window_days_default", 90)
CHRONIC_THRESHOLD_EVENTS = CHRONIC_CFG.get("min_events_default", 3)
CHRONIC_MIN_RATE = CHRONIC_CFG.get("min_rate_per_year_default", 1.5)

FEATURE_REGISTRY = {
    "temporal_leakage": ["event", "duration_days", "Ilk_Ariza_Tarihi", "Son_Ariza_Tarihi", 
                        "Fault_Count", "Ariza_Gecmisi"],
    "chronic_features": ["Chronic_Flag", "Chronic_Decay_Skoru", "MTBF_Bayes_Gun", 
                        "Chronic_Trend_Slope", "Chronic_Rate_Yillik"],
    "structural_features": ["cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi", "Gerilim_Sinifi", 
                           "Gerilim_Seviyesi", "Marka", "kVA_Rating", "Sehir", "Ilce", 
                           "Mahalle", "Location_Known", "Musteri_Sayisi"],
}
# =============================================================================
# FAILURE DEFINITION FIX - Based on Your Actual Cause Codes
# =============================================================================
"""
CRITICAL: Your data shows 87% of "faults" are PROTECTIVE OPERATIONS, not failures!

Current Problem:
- Sigorta "failure" rate: 30.3%
- Model predicts: "Which fuses will open?" (WRONG TASK)

After This Fix:
- Sigorta REAL failure rate: ~2-3%
- Model predicts: "Which equipment will physically degrade?" (CORRECT TASK)
"""

# =============================================================================
# FILTER DEFINITIONS (Based on diagnostic_script.py output)
# =============================================================================

# Equipment actually broke/degraded
REAL_FAILURE_CODES = [
    # Disconnectors & Switches
    "OG Ayırıcı Arızası",           # 794 records - Disconnector failure
    "AG Yük Ayırıcı Arızası",       # 51 records - Load switch failure
    
    # Transformers
    "OG Trafo Arızası",             # 140 records - Transformer failure
    
    # Conductors & Lines
    "İletken Kopması",              # 28 records - Conductor breakage
    "AG Tel Kopuğu",                # 27 records - Wire breakage
    "OG İletken Kopması",           # 4 records - MV conductor breakage
    "AG Nötr İletken Kopması",      # 5 records - Neutral conductor breakage
    "AG Yeraltı Kablo Arızası",     # 3 records - Underground cable failure
    "AG Yeraltı Branşman Kablo Arızası",  # 3 records
    "Kablo Başlığı Arızası",        # 3 records - Cable termination failure
    
    # Poles & Infrastructure
    "Direk Hasarı Kırılması",       # 8 records - Pole damage/breakage
    "AG Direk Kırılması",           # 18 records - Pole breakage
    
    # Panels & Boxes
    "AG Box Arızası",               # 14 records - Box failure
    "AG Pano Arızası",              # 4 records - Panel failure
    "NH Altlık Arızası",            # 42 records - NH base failure
    
    # Other Equipment Failures
    "AG Travers Arızası",           # 4 records - Crossarm failure
    "AG Sehim Bozukluğu",           # 16 records - Sag defect
]

# Protective operations (fuses/breakers doing their job)
PROTECTIVE_OPERATIONS = [
"""     # Fuse operations (87% of all records!)
    "AG Pano Kol Sigorta Atığı",    # 5,414 records - Fuse opened (NORMAL!)
    "OG Sigorta Atması",            # 2,470 records - Fuse tripped (NORMAL!)
    "OG Sigorta Atığı",             # 1,996 records - Fuse tripped (NORMAL!)
    "AG Pano Faz Sigorta Atığı",    # 50 records - Phase fuse tripped
    "AG Box SDK Giriş Sigorta Atığı",  # 11 records
    "AG Box SDK Abone Çıkış Sigorta Atığı",  # 8 records
    "AG Box / Sdk Giriş Sigorta Atığı",  # 7 records
    "AG Sigorta Atığı",             # 7 records
    
    # Breaker operations
    "AG Termik Açması",             # 42 records - Thermal trip (NORMAL!)
    "TMS Açması",                   # 37 records - Circuit breaker trip
    "OG Fider Açması",              # 7 records - Feeder breaker trip """
]

# Maintenance/planned events
MAINTENANCE_EVENTS = [
    "Planlı Kesinti / Müdahale",    # 16 records
    "Planlı Kesinti Müdahale",      # 16 records
    "Direk Değişimi",               # 42 + 6 records - Pole replacement
    "Şebeke Bakım Çalışması",       # 1 record
]

# External causes (not equipment failure)
EXTERNAL_CAUSES = [
    "Üçüncü Şahısların Vermiş Olduğu Hasarlar",  # 7 records - Third party damage
]

# Unknown/other
OTHER_EVENTS = [
    "Enerji Kesintisi Yapılmamıştır",  # 10 records - No outage occurred
]

# =============================================================================
# FILTERING FUNCTION
# =============================================================================

def filter_real_failures(df_fault: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Filter to ONLY real equipment failures (not protective operations)
    
    Returns:
        DataFrame with only records where equipment physically failed
    """
    
    if "cause code" not in df_fault.columns:
        logger.error("[FILTER] 'cause code' column not found - cannot filter!")
        logger.error("[FILTER] Using ALL fault records (PoF will be inflated!)")
        return df_fault
    
    original = len(df_fault)
    
    # Keep only real failures
    df_real = df_fault[df_fault["cause code"].isin(REAL_FAILURE_CODES)].copy()
    
    filtered_out = original - len(df_real)
    
    logger.info("="*60)
    logger.info("[FAILURE FILTER] Real Equipment Failures Only")
    logger.info("="*60)
    logger.info(f"Original records: {original:,}")
    logger.info(f"Real failures: {len(df_real):,} ({100*len(df_real)/original:.1f}%)")
    logger.info(f"Filtered out: {filtered_out:,} ({100*filtered_out/original:.1f}%)")
    
    # Show what was filtered
    excluded = df_fault[~df_fault["cbs_id"].isin(df_real["cbs_id"])]
    logger.info("\n[TOP EXCLUDED CAUSES]")
    for cause, count in excluded["cause code"].value_counts().head(5).items():
        logger.info(f"  - {cause}: {count:,}")
    
    logger.info("="*60 + "\n")
    
    return df_real

# =============================================================================
# LOGGING
# =============================================================================
def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"pof3_{ts}.log")

    logger = logging.getLogger("pof3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    import io
    ch = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("="*80)
    logger.info("PoF3 Pipeline - Clean Production Version")
    logger.info("="*80)
    return logger

# =============================================================================
# UTILITIES
# =============================================================================
def ensure_dirs():
    for d in [INTERMEDIATE_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    for p in list(INTERMEDIATE_PATHS.values()) + list(OUTPUT_PATHS.values()):
        os.makedirs(os.path.dirname(p), exist_ok=True)

def parse_date_safely(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except:
        return pd.NaT

def clean_equipment_type(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip()
            .str.replace(" Arızaları", "", regex=False)
            .str.replace(" Ariza", "", regex=False)
            .str.strip())

def convert_duration_minutes(series: pd.Series, logger: logging.Logger) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    if pd.notna(med) and med > 10000:
        logger.info("[DURATION] Converting from milliseconds to minutes")
        return s / 60000.0
    return s

# =============================================================================
# TEMPORAL SPLIT (Core of Leakage Prevention)
# =============================================================================
def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ✅ FIXED: Returns INDEX LABELS (not positions) based on sorted time.
    Ensures .loc selection works correctly downstream.
    """
    if "Kurulum_Tarihi" not in df.columns:
        raise ValueError("Temporal split requires 'Kurulum_Tarihi' column")
    
    # 1. Parse dates safely
    install_dates = pd.to_datetime(df["Kurulum_Tarihi"], errors="coerce")
    if install_dates.isna().all():
        raise ValueError("All Kurulum_Tarihi values are invalid")
    
    # 2. Create a temporary dataframe for sorting (preserve original index)
    df_sorted = df.copy()
    df_sorted["_install_clean"] = install_dates
    
    # Sort by date, but KEEP the original index
    df_sorted = df_sorted.sort_values("_install_clean")
    
    # 3. Determine Cutoff Index
    cutoff_pos = int(len(df_sorted) * (1 - test_size))
    
    # 4. Split and retrieve the ORIGINAL INDEX LABELS
    train_labels = df_sorted.iloc[:cutoff_pos].index.values
    test_labels = df_sorted.iloc[cutoff_pos:].index.values
    
    if logger:
        cutoff_date = df_sorted.iloc[cutoff_pos]["_install_clean"]
        logger.info(f"[TEMPORAL SPLIT] Cutoff: {cutoff_date.date()}")
        logger.info(f"[TEMPORAL SPLIT] Train: {len(train_labels)} | Test: {len(test_labels)}")
        
        if "event" in df.columns:
            # Using .loc because we have labels now
            train_ev = df.loc[train_labels, "event"].mean()
            test_ev = df.loc[test_labels, "event"].mean()
            logger.info(f"[TEMPORAL SPLIT] Train events: {train_ev:.1%} | Test events: {test_ev:.1%}")
    
    return train_labels, test_labels

# =============================================================================
# DATA LOADING
# =============================================================================
def load_fault_data(logger: logging.Logger) -> pd.DataFrame:
    """Load and clean fault records"""
    path = DATA_PATHS["fault_data"]
    logger.info(f"[LOAD] Fault data: {path}")
    
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    
    # Select essential columns
    base_cols = ["cbs_id", "Şebeke Unsuru", "Sebekeye_Baglanma_Tarihi", 
                 "started at", "ended at", "duration time", "cause code"]
    maint_cols = ["Bakım Sayısı", "Son Bakım İş Emri Tarihi", "MARKA", 
                  "kVA_Rating", "component_voltage", "voltage_level"]
    
    use_cols = [c for c in base_cols + maint_cols if c in df.columns]
    df = df[use_cols].copy()
    
    # Rename
    df = df.rename(columns={
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        #"cause code": "Ariza_Nedeni",
        "duration time": "Süre_Ham",
        "Bakım Sayısı": "Bakim_Sayisi",
        "MARKA": "Marka",
        "component_voltage": "Gerilim_Seviyesi",
        "voltage_level": "Gerilim_Sinifi",
    })
    
    # Parse dates
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["started at"] = df["started at"].apply(parse_date_safely)
    df["ended at"] = df["ended at"].apply(parse_date_safely)
    df["Süre_Dakika"] = convert_duration_minutes(df["Süre_Ham"], logger)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])
    
    # Filter invalid records
    original = len(df)
    df = df[df["cbs_id"].notna()].copy()
    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()
    
    df = df[
        df["Kurulum_Tarihi"].notna() &
        df["started at"].notna() &
        df["ended at"].notna() &
        df["Süre_Dakika"].notna()
    ].copy()
    
    logger.info(f"[LOAD] Fault records: {len(df)}/{original} ({100*len(df)/original:.1f}%)")
    return df

def load_healthy_data(logger: logging.Logger) -> pd.DataFrame:
    """Load healthy equipment (no fault history)"""
    path = DATA_PATHS["healthy_data"]
    logger.info(f"[LOAD] Healthy data: {path}")
    
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    
    if "cbs_id" not in df.columns:
        df = df.rename(columns={"ID": "cbs_id"})
    
    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()
    df = df.rename(columns={
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "MARKA": "Marka",
    })
    
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])
    df = df[df["Kurulum_Tarihi"].notna() & df["cbs_id"].notna()].copy()
    
    logger.info(f"[LOAD] Healthy equipment: {len(df)}")
    return df

# =============================================================================
# STEP 01: EQUIPMENT MASTER + SURVIVAL BASE
# =============================================================================
def build_equipment_master(
    df_fault: pd.DataFrame,
    df_healthy: pd.DataFrame,
    logger: logging.Logger,
    data_end_date: pd.Timestamp
) -> pd.DataFrame:
    """Combine fault + healthy equipment into master registry"""
    
    # Aggregate fault equipment
    fault_agg = df_fault.groupby("cbs_id").agg(
        Kurulum_Tarihi=("Kurulum_Tarihi", "min"),
        Ekipman_Tipi=("Ekipman_Tipi", "first"),
        Fault_Count=("cbs_id", "size"),
        Ilk_Ariza_Tarihi=("started at", "min"),
        Son_Ariza_Tarihi=("started at", "max"),
        Marka=("Marka", "first"),
        Gerilim_Seviyesi=("Gerilim_Seviyesi", "max"),
        Gerilim_Sinifi=("Gerilim_Sinifi", "first"),
    ).reset_index()
    
    # Aggregate healthy equipment
    healthy_agg = df_healthy.groupby("cbs_id").agg(
        Kurulum_Tarihi=("Kurulum_Tarihi", "min"),
        Ekipman_Tipi=("Ekipman_Tipi", "first"),
        Marka=("Marka", "first"),
    ).reset_index()
    healthy_agg["Fault_Count"] = 0
    
    # Combine
    all_eq = pd.concat([fault_agg, healthy_agg], ignore_index=True)
    all_eq = all_eq.sort_values(["cbs_id", "Fault_Count"], ascending=[True, False]) \
                   .drop_duplicates("cbs_id", keep="first")
    
    # Collapse rare equipment types
    counts = all_eq["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[COLLAPSE] Rare types → 'Diger': {rare}")
        all_eq.loc[all_eq["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Diger"
    
    logger.info(f"[MASTER] Equipment registry: {len(all_eq)} assets")
    return all_eq

# =============================================================================
# UPDATED build_survival_base (Replace your current version)
# =============================================================================

def build_survival_base(
    equipment_master: pd.DataFrame,
    df_fault: pd.DataFrame,
    logger,
    data_end_date
) -> pd.DataFrame:
    """
    Create survival dataset - ONLY counts REAL equipment failures
    
    ✅ FIXED: Filters out protective operations (fuse trips, breaker openings)
    """
    
    # ✅ CRITICAL FIX: Filter to real failures only
    df_fault_real = filter_real_failures(df_fault, logger)
    
    # First REAL failure per equipment
    first_real_fail = df_fault_real.groupby("cbs_id")["started at"].min().rename("Ilk_Gercek_Ariza_Tarihi")
    
    # Keep essential columns
    keep_cols = ["cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi", "Fault_Count"]
    if "Gerilim_Sinifi" in equipment_master.columns:
        keep_cols.append("Gerilim_Sinifi")
    if "Marka" in equipment_master.columns:
        keep_cols.append("Marka")
    
    df = equipment_master[[c for c in keep_cols if c in equipment_master.columns]].copy()
    df = df.merge(first_real_fail, on="cbs_id", how="left")
    
    # Event = 1 if equipment had REAL failure (not just protective operation)
    df["event"] = df["Ilk_Gercek_Ariza_Tarihi"].notna().astype(int)
    
    # Duration to REAL failure (or censoring)
    df["duration_days"] = np.where(
        df["event"] == 1,
        (df["Ilk_Gercek_Ariza_Tarihi"] - df["Kurulum_Tarihi"]).dt.days,
        (data_end_date - df["Kurulum_Tarihi"]).dt.days
    )
    
    df = df[df["duration_days"] > 0].copy()
    df["duration_days"] = df["duration_days"].clip(upper=60*365)
    
    logger.info(f"[SURVIVAL] Base: {len(df)} | REAL Failures: {df['event'].sum()} ({100*df['event'].mean():.1f}%)")
    
    return df


# =============================================================================
# STEP 02: FEATURE ENGINEERING - SINGLE DATAFRAME APPROACH
# =============================================================================
# =============================================================================
# STEP 02: FEATURE ENGINEERING - SINGLE DATAFRAME APPROACH
# =============================================================================

def compute_chronic_features(
    df_fault: pd.DataFrame,
    t_ref: pd.Timestamp,
    logger: logging.Logger
) -> pd.DataFrame:
    """Chronic equipment detection (Bayesian MTBF + Decay)"""
    
    window_start = t_ref - pd.Timedelta(days=CHRONIC_WINDOW_DAYS)
    fe = df_fault[df_fault["started at"] >= window_start].copy()
    
    if len(fe) == 0:
        logger.warning(f"[CHRONIC] No faults in window")
        return pd.DataFrame(columns=["cbs_id", "Ariza_Sayisi_90g", "Chronic_Rate_Yillik", "Chronic_Decay_Skoru", "Chronic_Flag"])
    
    # Count faults in window
    counts = fe.groupby("cbs_id").size().rename("Ariza_Sayisi_90g")
    
    # Exponential decay score (newer faults weighted more)
    age_days = (t_ref - fe["started at"]).dt.days.clip(lower=0)
    fe["decay"] = np.exp(-0.05 * age_days)
    decay_score = fe.groupby("cbs_id")["decay"].sum().rename("Chronic_Decay_Skoru")
    
    # Rate per year
    rate = (counts / (CHRONIC_WINDOW_DAYS / 365.25)).rename("Chronic_Rate_Yillik")
    
    # Chronic flag
    chronic_flag = ((counts >= CHRONIC_THRESHOLD_EVENTS) | (rate >= CHRONIC_MIN_RATE)).astype(int).rename("Chronic_Flag")
    
    out = pd.concat([counts, rate, decay_score, chronic_flag], axis=1).reset_index()
    logger.info(f"[CHRONIC] Window: {CHRONIC_WINDOW_DAYS}d | Chronic assets: {chronic_flag.sum()}")
    return out


def add_survival_columns_inplace(
    df: pd.DataFrame,
    df_fault_filtered: pd.DataFrame,
    data_end_date: pd.Timestamp,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Add event/duration directly to equipment_master
    No separate survival_base needed!
    """
    # First REAL failure per equipment
    first_fail = df_fault_filtered.groupby("cbs_id")["started at"].min()
    
    # Add to existing dataframe
    df["Ilk_Gercek_Ariza_Tarihi"] = df["cbs_id"].map(first_fail)
    
    # Calculate event flag
    df["event"] = df["Ilk_Gercek_Ariza_Tarihi"].notna().astype(int)
    
    # Calculate duration
    df["duration_days"] = np.where(
        df["event"] == 1,
        (df["Ilk_Gercek_Ariza_Tarihi"] - df["Kurulum_Tarihi"]).dt.days,
        (data_end_date - df["Kurulum_Tarihi"]).dt.days
    )
    
    # Clean up
    df = df[df["duration_days"] > 0].copy()
    df["duration_days"] = df["duration_days"].clip(upper=60*365)
    
    logger.info(f"[SURVIVAL] Added to master: {len(df)} assets, {df['event'].sum()} events ({100*df['event'].mean():.1f}%)")
    return df


def add_temporal_features_inplace(
    df: pd.DataFrame,
    t_ref: pd.Timestamp,
    chronic_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Add temporal features directly to existing dataframe
    Age calculation happens here - no separate merge!
    """
    # Calculate age from Kurulum_Tarihi (which is already in df)
    df["Tref_Yas_Gun"] = (t_ref - df["Kurulum_Tarihi"]).dt.days.clip(lower=0)
    df["Tref_Ay"] = t_ref.month
    
    # Merge chronic features if available
    if chronic_df is not None and len(chronic_df) > 0:
        # Use merge but keep all df columns (left join)
        df = df.merge(chronic_df, on="cbs_id", how="left")
        # Fill NaN for equipment with no chronic history
        chronic_cols = ["Ariza_Sayisi_90g", "Chronic_Rate_Yillik", "Chronic_Decay_Skoru", "Chronic_Flag"]
        for col in chronic_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    
    logger.info(f"[FEATURES] Temporal: Added age + chronic features")
    return df
# =============================================================================
# STEP 03: MODEL TRAINING
# =============================================================================
def remove_multicollinear_features(X: pd.DataFrame, threshold: float = 10.0, logger=None) -> pd.DataFrame:
    """
    Remove features with high VIF (Variance Inflation Factor)
    VIF > 10 indicates severe multicollinearity
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Only numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    
    if X_numeric.shape[1] < 2:
        return X  # Need at least 2 features for VIF
    
    # Calculate VIF iteratively
    while True:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_numeric.columns
        vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) 
                          for i in range(X_numeric.shape[1])]
        
        max_vif = vif_data["VIF"].max()
        
        if max_vif > threshold:
            drop_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            if logger:
                logger.info(f"[VIF] Dropping {drop_feature} (VIF={max_vif:.1f})")
            X_numeric = X_numeric.drop(columns=[drop_feature])
            X = X.drop(columns=[drop_feature])
        else:
            break
    
    return X
def select_top_features(X_train, y_train, X_test, top_k=20, logger=None):
    """
    Select top K features using Random Forest importance
    Useful when you have many features after one-hot encoding
    """
    from sklearn.ensemble import RandomForestClassifier
    
    if X_train.shape[1] <= top_k:
        return X_train, X_test  # Already fewer than top_k
    
    # Train quick RF to get importances
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get top K features
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(top_k).index.tolist()
    
    if logger:
        logger.info(f"[FEATURE IMPORTANCE] Selected top {len(top_features)} features")
    
    return X_train[top_features], X_test[top_features]
def remove_highly_correlated_features(X: pd.DataFrame, threshold=0.95, logger=None):
    """
    Remove features with correlation > threshold to another feature
    Keeps the first of each correlated pair
    """
    # Only numeric
    X_numeric = X.select_dtypes(include=[np.number])
    
    if X_numeric.shape[1] < 2:
        return X
    
    # Calculate correlation matrix
    corr_matrix = X_numeric.corr().abs()
    
    # Upper triangle (avoid duplicates)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > threshold)]
    
    if to_drop and logger:
        logger.info(f"[CORRELATION] Dropping {len(to_drop)} highly correlated features: {to_drop}")
    
    return X.drop(columns=to_drop)



def build_preprocessor(X: pd.DataFrame):
    """Sklearn pipeline for numeric + categorical features"""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols)
        ],
        remainder="drop"
    )

def select_survival_safe_features(df: pd.DataFrame, structural_cols: list, logger: logging.Logger) -> list:
    """Filter to leakage-free features"""
    forbidden = (FEATURE_REGISTRY["temporal_leakage"] + 
                 FEATURE_REGISTRY["chronic_features"])
    
    safe_cols = [c for c in structural_cols if c in df.columns and c not in forbidden]
    logger.info(f"[FEATURE SELECT] Safe: {len(safe_cols)}/{len(structural_cols)}")
    return safe_cols

def select_cox_safe_features(df: pd.DataFrame, structural_cols: list, logger: logging.Logger) -> pd.DataFrame:
    """
    ✅ FIXED: Added VarianceThreshold to drop constant features.
    """
    from sklearn.feature_selection import VarianceThreshold

    base_cols = select_survival_safe_features(df, structural_cols, logger)
    
    # Add Kurulum_Tarihi for temporal split logic
    if "Kurulum_Tarihi" in df.columns and "Kurulum_Tarihi" not in base_cols:
        base_cols = base_cols + ["Kurulum_Tarihi"]
    
    X = df[base_cols].copy()
    
    # Preserve Kurulum_Tarihi before encoding/filtering
    kurulum_col = None
    if "Kurulum_Tarihi" in X.columns:
        kurulum_col = X["Kurulum_Tarihi"].copy()
        X = X.drop(columns=["Kurulum_Tarihi"])
    
    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols[:]:
        if X[col].nunique() > 20:
            logger.warning(f"[COX] Dropping {col}: high cardinality (>20)")
            X = X.drop(columns=[col])
            cat_cols.remove(col)
    
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    
    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Add this inside select_cox_safe_features, before VIF check
    for col in X.columns:
        if X[col].nunique() <= 1:
            logger.info(f"[DROP] {col} is constant (single value)")
            X = X.drop(columns=[col])
    # ✅ FILTER LOW VARIANCE (Constant Columns)
    # Drop features where 99% of values are the same
    selector = VarianceThreshold(threshold=(.99 * (1 - .99)))
    # 5. ✅ Remove low variance
    #X = remove_low_variance_features(X, logger)
    
    # 6. ⚠️ NEW: Remove multicollinear features
    X = remove_multicollinear_features(X, threshold=10.0, logger=logger)
    
    # 7. ⚠️ NEW: Remove highly correlated features
    X = remove_highly_correlated_features(X, threshold=0.95, logger=logger)
    try:
        selector.fit(X)
        kept_cols = X.columns[selector.get_support()]
        dropped_count = len(X.columns) - len(kept_cols)
        if dropped_count > 0:
            logger.info(f"[COX] Dropped {dropped_count} low-variance columns")
        X = X[kept_cols]
    except ValueError:
        # Happens if X is empty or all constant
        pass

    # Re-add Kurulum_Tarihi at end
    if kurulum_col is not None:
        X["Kurulum_Tarihi"] = kurulum_col
    
    return X

def train_cox_weibull(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    logger: logging.Logger
):
    """
    ✅ FIXED: Uses .loc for splitting with Index Labels.
    """
    if not LIFELINES_OK:
        return None, None
    
    work = X.copy()
    work["duration_days"] = duration.values
    work["event"] = event.values
    
    # Check for temporal column
    has_kurulum = "Kurulum_Tarihi" in work.columns
    if has_kurulum:
        kurulum_backup = work["Kurulum_Tarihi"].copy()
    
    # Ensure numeric matrix for Lifelines
    # (Exclude Kurulum_Tarihi from the numeric conversion step to avoid errors)
    numeric_cols = work.columns.drop(["Kurulum_Tarihi"]) if has_kurulum else work.columns
    work[numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    try:
        if has_kurulum:
            # Restore date column for splitting
            work["Kurulum_Tarihi"] = kurulum_backup
            # Get LABELS
            train_labels, test_labels = temporal_train_test_split(work, test_size=0.25, logger=logger)
            
            # Use .loc because we have Labels
            train_data = work.loc[train_labels]
            test_data = work.loc[test_labels]
        else:
            raise ValueError("Kurulum_Tarihi missing")
            
    except Exception as e:
        logger.warning(f"[COX] Temporal split failed ({e}), using random split")
        # Random split on INDEX LABELS
        train_labels, test_labels = train_test_split(
            work.index.values, test_size=0.25, random_state=42, stratify=event.values
        )
        train_data = work.loc[train_labels]
        test_data = work.loc[test_labels]
    
    # Drop helper column before training
    train_data = train_data.drop(columns=["Kurulum_Tarihi"], errors="ignore")
    test_data = test_data.drop(columns=["Kurulum_Tarihi"], errors="ignore")
    
    # Cox Training
    cox = None
    try:
        cox = CoxPHFitter(penalizer=0.05)
        cox.fit(train_data, duration_col="duration_days", event_col="event")
        test_scores = cox.predict_partial_hazard(test_data)
        c_ind = concordance_index(test_data["duration_days"], -test_scores, test_data["event"])
        logger.info(f"[COX] Test Concordance: {c_ind:.4f}")
    except Exception as e:
        logger.error(f"[COX] Training failed: {e}")
    
    # Weibull Training
    wb = None
    try:
        wb = WeibullAFTFitter(penalizer=0.05)
        wb.fit(train_data, duration_col="duration_days", event_col="event")
        wb_pred = wb.predict_median(test_data)
        wb_cind = concordance_index(test_data["duration_days"], wb_pred, test_data["event"])
        logger.info(f"[WEIBULL] Test Concordance: {wb_cind:.4f}")
    except Exception as e:
        logger.error(f"[WEIBULL] Training failed: {e}")
    
    return cox, wb

def train_rsf_survival(
    df: pd.DataFrame,
    structural_cols: list,
    logger: logging.Logger
):
    """
    ✅ FIXED: Handles DataFrame (.loc) vs Numpy Array (positional) indexing correctly.
    """
    if not SKSURV_OK:
        return None
    
    cols = select_survival_safe_features(df, structural_cols, logger)
    X = df[cols].copy()
    
    # Drop columns that are completely empty/NaN to satisfy sksurv
    X = X.dropna(axis=1, how='all')
    
    # Create structured array for target
    y = Surv.from_arrays(
        event=df["event"].astype(bool).values,
        time=df["duration_days"].values
    )
    
    try:
        # Get LABELS
        train_labels, test_labels = temporal_train_test_split(df, test_size=0.25, logger=logger)
        
        # DataFrame uses .loc with labels
        X_train = X.loc[train_labels]
        X_test = X.loc[test_labels]
        
        # Numpy array needs Integer Positions -> Convert labels to positions
        train_pos = df.index.get_indexer(train_labels)
        test_pos = df.index.get_indexer(test_labels)
        
        y_train = y[train_pos]
        y_test = y[test_pos]
        
    except Exception as e:
        logger.warning(f"[RSF] Temporal split failed ({e}), using random")
        # Fallback using standard split (returns arrays directly)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=df["event"].values
        )
    
    # Build pipeline
    pre = build_preprocessor(X_train)
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    pipe = Pipeline([("pre", pre), ("rsf", rsf)])
    try:
        pipe.fit(X_train, y_train)
        
        risk = pipe.predict(X_test)
        ci = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
        logger.info(f"[RSF] Test Concordance: {ci:.4f}")
        return pipe
    except Exception as e:
        logger.error(f"[RSF] Training failed: {e}")
        return None
def train_ml_models(
    df: pd.DataFrame,
    feature_cols: list,
    horizons_days: list,
    logger: logging.Logger
):
    """Train XGBoost + CatBoost for time-horizon PoF prediction"""
    X = df[feature_cols].copy()
    
    # Clean features
    X = X.drop(columns=X.select_dtypes(include=['datetime64']).columns, errors='ignore')
    X = X.select_dtypes(include=[np.number, 'object'])
    
    models = {}
    for H in horizons_days:
        y_h = ((df["event"] == 1) & (df["duration_days"] <= H)).astype(int)
        
        if y_h.sum() < 50:
            logger.info(f"[ML] Skipping {H//30}ay: insufficient positives ({y_h.sum()})")
            continue
        
        # Temporal split
        try:
            train_idx, test_idx = temporal_train_test_split(df, test_size=0.25, logger=None)
        except:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)), test_size=0.25, random_state=42, stratify=y_h
            )
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_h.iloc[train_idx], y_h.iloc[test_idx]
        
        pre = build_preprocessor(X_train)
        X_tr = pre.fit_transform(X_train)
        X_te = pre.transform(X_test)
        # ⚠️ NEW: Feature selection based on importance
        if X_tr.shape[1] > 30:  # Only if many features
            X_tr, X_te = select_top_features(
                X_tr, y_train, X_te, 
                top_k=min(30, X_tr.shape[1]), 
                logger=logger
            )
        # XGBoost
        if XGB_OK:
            xgb = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                eval_metric="auc", random_state=42
            )
            xgb.fit(X_tr, y_train)
            auc = roc_auc_score(y_test, xgb.predict_proba(X_te)[:, 1])
            logger.info(f"[ML] XGB {H//30}ay AUC: {auc:.4f}")
            models[("xgb", f"{H//30}ay")] = Pipeline([("pre", pre), ("mdl", xgb)])
        
        # CatBoost
        if CAT_OK:
            cat = CatBoostClassifier(
                iterations=300, depth=4, learning_rate=0.05,
                verbose=False, random_seed=42
            )
            cat.fit(X_tr, y_train)
            auc = roc_auc_score(y_test, cat.predict_proba(X_te)[:, 1])
            logger.info(f"[ML] CAT {H//30}ay AUC: {auc:.4f}")
            models[("cat", f"{H//30}ay")] = Pipeline([("pre", pre), ("mdl", cat)])
    
    return {"models": models, "safe_cols": X.columns.tolist()} if models else None

# =============================================================================
# BACKTESTING (Temporal Validation Proof)
# =============================================================================
class TemporalBacktester:
    """Walk-forward validation to prove model doesn't cheat"""
    
    def __init__(self, df_fault: pd.DataFrame, df_healthy: pd.DataFrame, logger: logging.Logger):
        self.df_fault = df_fault
        self.df_healthy = df_healthy
        self.logger = logger
        self.results = []
    
    def _generate_snapshot(self, cutoff_date: pd.Timestamp):
        """Create training dataset - SINGLE DATAFRAME"""
        faults_past = self.df_fault[self.df_fault["started at"] <= cutoff_date].copy()
        faults_filtered = filter_real_failures(faults_past, self.logger)
        
        # Build complete dataset in one go
        equipment_master = build_equipment_master(faults_past, self.df_healthy, self.logger, cutoff_date)
        df_snapshot = add_survival_columns_inplace(equipment_master, faults_filtered, cutoff_date, self.logger)
        
        chronic_df = compute_chronic_features(faults_past, cutoff_date, self.logger)
        df_snapshot = add_temporal_features_inplace(df_snapshot, cutoff_date, chronic_df, self.logger)
        
        return df_snapshot
    
    def _train_simple_model(self, df: pd.DataFrame, features: list):
        """Train lightweight XGBoost for backtesting"""
        X = df[features].copy()
        
        # Remove non-numeric
        X = X.drop(columns=X.select_dtypes(include=['datetime64']).columns, errors='ignore')
        X = X.select_dtypes(include=[np.number]).fillna(0)
        
        if X.empty or len(X.columns) == 0:
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy='constant', constant=0), []
        
        model = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", random_state=42)
        model.fit(X, df["event"])
        
        return model, X.columns.tolist()
    
    def run(self, start_year: int, end_year: int, horizon_days: int = 365):
        """Run walk-forward validation"""
        self.logger.info("="*60)
        self.logger.info(f"[BACKTEST] Walk-Forward Validation ({start_year}-{end_year})")
        self.logger.info("="*60)
        
        for year in range(start_year, end_year + 1):
            cutoff_date = pd.Timestamp(f"{year}-01-01")
            test_end_date = cutoff_date + pd.Timedelta(days=horizon_days)
            
            # Training data (history up to Jan 1st)
            df_train = self._generate_snapshot(cutoff_date)
            
            # Ground truth (who actually failed in next 12 months)
            future_faults = self.df_fault[
                (self.df_fault["started at"] > cutoff_date) & 
                (self.df_fault["started at"] <= test_end_date)
            ]
            failed_ids = set(future_faults["cbs_id"].unique())
            y_true = df_train["cbs_id"].isin(failed_ids).astype(int)
            
            if y_true.sum() < 10:
                self.logger.warning(f"[BACKTEST] {year}: Insufficient failures ({y_true.sum()})")
                continue
            
            # Train model
            structural_cols = [c for c in df_train.columns 
                             if c not in ["cbs_id", "event", "duration_days", "Kurulum_Tarihi"]]
            model, valid_features = self._train_simple_model(df_train, structural_cols)
            
            # Evaluate
            if valid_features:
                X_test = df_train[valid_features].fillna(0)
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = np.zeros(len(y_true))
            
            auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
            
            # Precision@Top-K
            top_100_idx = np.argsort(probs)[-100:]
            hits = y_true.iloc[top_100_idx].sum()
            
            self.logger.info(f"[BACKTEST] {year} → AUC: {auc:.3f} | Top-100 Hits: {hits}")
            
            self.results.append({
                "Year": year,
                "AUC": auc,
                "Top100_Hits": hits,
                "Total_Failures": y_true.sum()
            })
        
        return pd.DataFrame(self.results)

# =============================================================================
# EQUIPMENT-SPECIFIC MODELING
# =============================================================================
def get_equipment_stats(df: pd.DataFrame, equipment_master: pd.DataFrame, logger: logging.Logger) -> dict:
    """Statistics per equipment type"""
    stats = {}
    for eq_type in df["Ekipman_Tipi"].unique():
        df_eq = df[df["Ekipman_Tipi"] == eq_type]
        em_eq = equipment_master[equipment_master["Ekipman_Tipi"] == eq_type]
        
        stats[eq_type] = {
            "n_total": len(df_eq),
            "n_events": int(df_eq["event"].sum()),
            "event_rate": float(df_eq["event"].mean()),
            "has_marka": int(em_eq["Marka"].notna().sum()) if "Marka" in em_eq.columns else 0,
        }
    
    logger.info("[EQUIPMENT STATS]")
    for eq_type, s in stats.items():
        logger.info(f"  {eq_type}: N={s['n_total']}, Events={s['n_events']} ({100*s['event_rate']:.1f}%), Marka={s['has_marka']}")
    
    return stats

def train_equipment_specific_models(
    df_eq: pd.DataFrame,
    structural_cols: list,
    temporal_cols: list,
    eq_type: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Train models for specific equipment type"""
    
    predictions = pd.DataFrame({"cbs_id": df_eq["cbs_id"]})
    
    # Survival models
    try:
        X_cox = select_cox_safe_features(df_eq, structural_cols, logger)
        cox, wb = train_cox_weibull(X_cox, df_eq["duration_days"], df_eq["event"], logger)
        
        if cox:
            cox_pred = predict_survival_pof(cox, X_cox, df_eq["duration_days"], 
                                           SURVIVAL_HORIZONS_DAYS, "cox", df_eq["cbs_id"])
            predictions = predictions.merge(cox_pred, on="cbs_id", how="left")
    except Exception as e:
        logger.warning(f"[{eq_type}] Cox/Weibull failed: {e}")
    
    try:
        rsf = train_rsf_survival(df_eq, structural_cols, logger)
        if rsf:
            rsf_pred = predict_rsf_pof(df_eq, rsf, structural_cols, SURVIVAL_HORIZONS_DAYS)
            predictions = predictions.merge(rsf_pred, on="cbs_id", how="left")
    except Exception as e:
        logger.warning(f"[{eq_type}] RSF failed: {e}")
    
    # ML models (if sufficient data)
    ml_features = structural_cols + [c for c in temporal_cols if c not in ["Kurulum_Tarihi"]]
    pos_12m = ((df_eq["event"] == 1) & (df_eq["duration_days"] <= 365)).sum()
    
    if pos_12m >= 50:
        try:
            ml_pack = train_ml_models(df_eq, ml_features, SURVIVAL_HORIZONS_DAYS, logger)
            if ml_pack:
                ml_pred = predict_ml_pof(df_eq, ml_pack, SURVIVAL_HORIZONS_DAYS)
                predictions = predictions.merge(ml_pred, on="cbs_id", how="left")
        except Exception as e:
            logger.warning(f"[{eq_type}] ML failed: {e}")
    else:
        logger.info(f"[{eq_type}] ML skipped: insufficient positives ({pos_12m} < 50)")
    
    return predictions

def analyze_marka_effect(df_eq: pd.DataFrame, eq_type: str, logger: logging.Logger) -> pd.DataFrame:
    """Brand risk analysis (explanatory)"""
    df_marka = df_eq[df_eq["Marka"].notna()].copy()
    
    if len(df_marka) < 30:
        return pd.DataFrame()
    
    marka_stats = df_marka.groupby("Marka").agg(
        Failures=("event", "sum"),
        Total=("event", "count"),
        Failure_Rate=("event", "mean"),
        Median_Age=("duration_days", "median")
    ).reset_index()
    
    marka_stats = marka_stats[marka_stats["Total"] >= 5].sort_values("Failure_Rate", ascending=False)
    
    logger.info(f"[{eq_type}] MARKA: {len(marka_stats)} brands analyzed")
    for _, row in marka_stats.head(3).iterrows():
        logger.info(f"  - {row['Marka']}: {row['Failure_Rate']:.1%} (N={int(row['Total'])})")
    
    marka_stats["Ekipman_Tipi"] = eq_type
    return marka_stats

def analyze_bakim_effect(equipment_master: pd.DataFrame, eq_type: str, logger: logging.Logger) -> pd.DataFrame:
    """Maintenance effect analysis (explanatory)"""
    if "Bakim_Sayisi" not in equipment_master.columns:
        return pd.DataFrame()
    
    df_eq = equipment_master[equipment_master["Ekipman_Tipi"] == eq_type].copy()
    df_bakim = df_eq[df_eq["Bakim_Sayisi"].notna() & (df_eq["Bakim_Sayisi"] > 0)]
    
    if len(df_bakim) < 30:
        return pd.DataFrame()
    
    df_bakim["Bakim_Bin"] = pd.cut(
        df_bakim["Bakim_Sayisi"],
        bins=[0, 1, 3, 5, 10, 100],
        labels=["1", "2-3", "4-5", "6-10", "10+"]
    )
    
    bakim_stats = df_bakim.groupby("Bakim_Bin", observed=True).agg(
        Asset_Count=("cbs_id", "count")
    ).reset_index()
    
    logger.info(f"[{eq_type}] BAKIM: {len(df_bakim)} assets with maintenance history")
    
    bakim_stats["Ekipman_Tipi"] = eq_type
    return bakim_stats

# =============================================================================
# STEP 04: PREDICTION
# =============================================================================
def predict_survival_pof(model, X: pd.DataFrame, duration: pd.Series, horizons: list, model_name: str, cbs_ids: pd.Series) -> pd.DataFrame:
    """Predict conditional PoF from survival model"""
    X_clean = X.drop(columns=["Kurulum_Tarihi"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0)
    age = duration.fillna(0).clip(lower=0).values
    
    out = pd.DataFrame({"cbs_id": cbs_ids.values})
    for H in horizons:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        times = np.unique(np.concatenate([age, age + H]))
        sf = model.predict_survival_function(X_clean, times=times)
        
        S_age = np.array([sf.iloc[np.searchsorted(times, a), j] for j, a in enumerate(age)])
        S_age_h = np.array([sf.iloc[np.searchsorted(times, a+H), j] for j, a in enumerate(age)])
        
        pof = 1.0 - np.clip((S_age_h + 1e-12) / (S_age + 1e-12), 0, 1)
        out[f"{model_name}_pof_{label}"] = pof
    
    return out

def predict_rsf_pof(df: pd.DataFrame, rsf_pipe, structural_cols: list, horizons: list) -> pd.DataFrame:
    """Predict conditional PoF from RSF model"""
    cols = [c for c in structural_cols if c in df.columns]
    X = df[cols].copy()
    age = df["duration_days"].fillna(0).clip(lower=0).values
    
    X_tr = rsf_pipe.named_steps["pre"].transform(X)
    sfs = rsf_pipe.named_steps["rsf"].predict_survival_function(X_tr, return_array=False)
    
    out = pd.DataFrame({"cbs_id": df["cbs_id"].values})
    for H in horizons:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        pofs = []
        for i, sf in enumerate(sfs):
            t, s = sf.x, sf.y
            a, b = age[i], age[i] + H
            S_a = np.interp(a, t, s, left=s[0], right=s[-1])
            S_b = np.interp(b, t, s, left=s[0], right=s[-1])
            pofs.append(1.0 - np.clip((S_b + 1e-12) / (S_a + 1e-12), 0, 1))
        out[f"rsf_pof_{label}"] = pofs
    return out

def predict_ml_pof(df: pd.DataFrame, ml_pack: dict, horizons: list) -> pd.DataFrame:
    """Predict PoF from ML models"""
    X = df[ml_pack["safe_cols"]].copy()
    out = pd.DataFrame({"cbs_id": df["cbs_id"].values})
    
    for H in horizons:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        preds = []
        
        if ("xgb", label) in ml_pack["models"]:
            preds.append(ml_pack["models"][("xgb", label)].predict_proba(X)[:, 1])
        if ("cat", label) in ml_pack["models"]:
            preds.append(ml_pack["models"][("cat", label)].predict_proba(X)[:, 1])
        
        if preds:
            out[f"ml_pof_{label}"] = np.mean(np.vstack(preds), axis=0)
    
    return out

def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """Ensemble PoF → Health Score (0-100)"""
    pof_cols = [c for c in df.columns if "_pof_" in c]
    if pof_cols:
        df["Mean_PoF"] = df[pof_cols].mean(axis=1)
        df["Health_Score"] = (100 * (1 - df["Mean_PoF"])).clip(0, 100)
    else:
        df["Mean_PoF"] = 0.0
        df["Health_Score"] = 100.0
    return df

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    ensure_dirs()
    logger = setup_logger()
    
    # Load data (once!)
    logger.info("[STEP 1] Loading data...")
    df_fault = load_fault_data(logger)
    df_healthy = load_healthy_data(logger)
    
    data_end_date = df_fault["started at"].max()
    logger.info(f"[CONFIG] Data range: {df_fault['started at'].min().date()} → {data_end_date.date()}")
    
    # =============================================================================
    # BACKTESTING (Proof that model doesn't cheat)
    # =============================================================================
    logger.info("\n" + "="*60)
    logger.info("BACKTESTING - Temporal Validation")
    logger.info("="*60)
    
    max_year = data_end_date.year
    backtester = TemporalBacktester(df_fault, df_healthy, logger)
    backtest_results = backtester.run(start_year=max_year-3, end_year=max_year-1)
    
    backtest_path = os.path.join(OUTPUT_DIR, "backtest_results_temporal.csv")
    backtest_results.to_csv(backtest_path, index=False)
    logger.info(f"[BACKTEST] Results saved: {backtest_path}\n")
    
    # =============================================================================
    # PRODUCTION PIPELINE - SINGLE DATAFRAME APPROACH
    # =============================================================================
    logger.info("="*60)
    logger.info("PRODUCTION - Training on Full History")
    logger.info("="*60 + "\n")
    
    logger.info("[STEP 2] Building complete dataset...")
    
    # Step 1: Build equipment master (has all base columns)
    equipment_master = build_equipment_master(df_fault, df_healthy, logger, data_end_date)
    
    # Step 2: Filter faults ONCE
    df_fault_filtered = filter_real_failures(df_fault, logger)
    
    # Step 3: Add survival columns IN-PLACE
    df_all = add_survival_columns_inplace(equipment_master.copy(), df_fault_filtered, data_end_date, logger)
    
    # Step 4: Calculate chronic features
    logger.info("[STEP 3] Engineering features...")
    chronic_df = compute_chronic_features(df_fault, data_end_date, logger)
    
    # Step 5: Add temporal features IN-PLACE (including age calculation)
    df_all = add_temporal_features_inplace(df_all, data_end_date, chronic_df, logger)
    
    # ✅ NO MERGING! df_all has everything already
    
    # Define feature columns for modeling
    structural_cols = ["Ekipman_Tipi", "Gerilim_Sinifi", "Gerilim_Seviyesi", "Marka"]
    structural_cols = [c for c in structural_cols if c in df_all.columns]
    
    temporal_cols = ["Tref_Yas_Gun", "Tref_Ay", "Ariza_Sayisi_90g", 
                     "Chronic_Rate_Yillik", "Chronic_Decay_Skoru", "Chronic_Flag"]
    temporal_cols = [c for c in temporal_cols if c in df_all.columns]
    
    logger.info(f"[DATASET] Single dataframe: {len(df_all)} rows × {len(df_all.columns)} cols")
    logger.info(f"[DATASET] Structural features: {structural_cols}")
    logger.info(f"[DATASET] Temporal features: {temporal_cols}")
    logger.info(f"[DATASET] Has Ekipman_Tipi: {'Ekipman_Tipi' in df_all.columns}")
    logger.info(f"[DATASET] Has Kurulum_Tarihi: {'Kurulum_Tarihi' in df_all.columns}\n")
    # =============================================================================
    # EQUIPMENT-STRATIFIED MODELING
    # =============================================================================
    logger.info("="*60)
    logger.info("STEP 4 - Equipment-Stratified Modeling")
    logger.info("="*60 + "\n")
    
    # Get equipment statistics
    eq_stats = get_equipment_stats(df_all, equipment_master, logger)
    
    # Train global fallback models first
    logger.info("\n[GLOBAL] Training fallback models...")
    X_cox_global = select_cox_safe_features(df_all, structural_cols, logger)
    cox_global, wb_global = train_cox_weibull(X_cox_global, df_all["duration_days"], df_all["event"], logger)
    rsf_global = train_rsf_survival(df_all, structural_cols, logger)
    
    ml_features_global = structural_cols + [c for c in temporal_cols if c not in ["Kurulum_Tarihi"]]
    ml_pack_global = train_ml_models(df_all, ml_features_global, SURVIVAL_HORIZONS_DAYS, logger)
    
    global_models = {
        "cox": cox_global,
        "weibull": wb_global,
        "rsf": rsf_global,
        "ml": ml_pack_global,
        "X_cox_cols": X_cox_global.columns.tolist()
    }
    
    # Train equipment-specific models
    MIN_SAMPLES = 100
    MIN_EVENTS = 30
    
    all_predictions = []
    all_marka_analyses = []
    all_bakim_analyses = []
    
    for eq_type in sorted(df_all["Ekipman_Tipi"].unique()):
        df_eq = df_all[df_all["Ekipman_Tipi"] == eq_type].copy()
        stats = eq_stats[eq_type]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{eq_type}] N={stats['n_total']}, Events={stats['n_events']} ({100*stats['event_rate']:.1f}%)")
        logger.info(f"{'='*60}")
        
        if stats["n_total"] < MIN_SAMPLES or stats["n_events"] < MIN_EVENTS:
            logger.warning(f"[{eq_type}] Using global fallback (insufficient data)")
            
            # Use global models
            preds = pd.DataFrame({"cbs_id": df_eq["cbs_id"]})

            try:
                # 1. Generate features for this specific equipment type
                X_eq = select_cox_safe_features(df_eq, structural_cols, logger)

                # --- FIX STARTS HERE ---
                # 2. Get the columns expected by the Global Model
                required_cols = global_models["X_cox_cols"]

                # 3. Add missing columns with 0
                missing_cols = set(required_cols) - set(X_eq.columns)
                for c in missing_cols:
                    X_eq[c] = 0

                # 4. Reorder strictly to match Global Model
                X_eq = X_eq[required_cols]
                # --- FIX ENDS HERE ---

                if cox_global:
                    cox_pred = predict_survival_pof(cox_global, X_eq, df_eq["duration_days"],
                                                    SURVIVAL_HORIZONS_DAYS, "cox", df_eq["cbs_id"])
                    preds = preds.merge(cox_pred, on="cbs_id", how="left")
            
            except Exception as e:
                logger.warning(f"[{eq_type}] Global Cox failed: {e}")
                # If global fails, we might end up with just IDs, which is fine (Health=100 default)

            # Mark the model type
            preds["Model_Type"] = "Global_Fallback"
            preds["Ekipman_Tipi"] = eq_type
            
            # Add to the master list
            all_predictions.append(preds)
            continue
        
        # Train equipment-specific models
        logger.info(f"[{eq_type}] Training equipment-specific models...")
        preds = train_equipment_specific_models(df_eq, structural_cols, temporal_cols, eq_type, logger)
        preds["Model_Type"] = "Equipment_Specific"
        preds["Ekipman_Tipi"] = eq_type
        all_predictions.append(preds)
                        
        # Explanatory analyses
        if stats["has_marka"] >= 30:
            marka_analysis = analyze_marka_effect(df_eq, eq_type, logger)
            if not marka_analysis.empty:
                all_marka_analyses.append(marka_analysis)
        
        bakim_analysis = analyze_bakim_effect(equipment_master, eq_type, logger)
        if not bakim_analysis.empty:
            all_bakim_analyses.append(bakim_analysis)
        
        # Save per-equipment output
        preds_full = df_eq[["cbs_id", "Ekipman_Tipi", "Fault_Count"]].merge(preds, on="cbs_id", how="left")
        preds_full = compute_health_score(preds_full)
        
        safe_name = eq_type.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"pof_{safe_name}.csv")
        preds_full.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"[{eq_type}] Saved: {out_path}")
    
    # =============================================================================
    # ENSEMBLE & FINAL OUTPUT
    # =============================================================================
# =============================================================================
    # ENSEMBLE & FINAL OUTPUT
    # =============================================================================
    logger.info("\n" + "="*60)
    logger.info("STEP 5 - Final Ensemble")
    logger.info("="*60 + "\n")
    
    if not all_predictions:
        logger.error("[ERROR] No predictions generated!")
        return
    
    # Combine all equipment types
    predictions = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"[ENSEMBLE] Combined {len(predictions)} predictions from {len(all_predictions)} equipment types")
    
    # Compute health scores
    predictions = compute_health_score(predictions)
    
    # Add reporting columns
    report_cols = ["Ekipman_Tipi", "Gerilim_Sinifi", "Fault_Count", "Kurulum_Tarihi"]
    report = df_all[["cbs_id"] + [c for c in report_cols if c in df_all.columns]].drop_duplicates("cbs_id")
    
    # ✅ FIX: Drop 'Ekipman_Tipi' from predictions before merging to avoid _x/_y suffixes
    preds_clean = predictions.drop(columns=["Ekipman_Tipi"], errors="ignore")
    
    report = report.merge(preds_clean, on="cbs_id", how="left")
    
    # Save main output
    out_path = os.path.join(OUTPUT_DIR, "pof_predictions_final.csv")
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OUTPUT] Main predictions: {out_path}")
    
    # ... (rest of the saving logic remains the same)
    
    # Save explanatory analyses
    if all_marka_analyses:
        marka_combined = pd.concat(all_marka_analyses, ignore_index=True)
        marka_path = os.path.join(OUTPUT_DIR, "marka_analysis.csv")
        marka_combined.to_csv(marka_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OUTPUT] Marka analysis: {marka_path}")
    
    if all_bakim_analyses:
        bakim_combined = pd.concat(all_bakim_analyses, ignore_index=True)
        bakim_path = os.path.join(OUTPUT_DIR, "bakim_analysis.csv")
        bakim_combined.to_csv(bakim_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OUTPUT] Bakim analysis: {bakim_path}")
    
    # Save intermediate files
    equipment_master.to_csv(os.path.join(OUTPUT_DIR, "equipment_master.csv"), index=False, encoding="utf-8-sig")
    #survival_base.to_csv(os.path.join(OUTPUT_DIR, "survival_base.csv"), index=False, encoding="utf-8-sig")
    #features_all.to_csv(os.path.join(OUTPUT_DIR, "features_all.csv"), index=False, encoding="utf-8-sig")
    # REPLACE WITH:
    df_all.to_csv(os.path.join(OUTPUT_DIR, "model_input_data_full.csv"), index=False, encoding="utf-8-sig")
    # Summary statistics
    critical = (report["Health_Score"] < 40).sum()
    mean_health = report["Health_Score"].mean()
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total assets: {len(report):,}")
    logger.info(f"Critical assets (Health<40): {critical:,} ({100*critical/len(report):.1f}%)")
    logger.info(f"Mean Health Score: {mean_health:.1f}")

    # Check if Ekipman_Tipi exists in report (it should from the merge)
    ekip_col = None
    for col in report.columns:
        if "Ekipman_Tipi" in col:
            ekip_col = col
            break

    if ekip_col:
        logger.info(f"Equipment types: {report[ekip_col].nunique()}")
    else:
        logger.info(f"Equipment types: {df_all['Ekipman_Tipi'].nunique()}")

    logger.info("="*60)

if __name__ == "__main__":
    main()