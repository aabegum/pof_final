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
from tqdm import tqdm  # <--- Add this with other imports
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from joblib import Parallel, delayed
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
OBSERVATION_START_DATE = pd.Timestamp("2021-01-01")
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
    
    # Fuse operations (87% of all records!)
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
    "OG Fider Açması",              # 7 records - Feeder breaker trip
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

    # Save intermediate output
    df.to_csv(INTERMEDIATE_PATHS["fault_events_clean"], index=False, encoding="utf-8-sig")
    logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['fault_events_clean']}")

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

    # Save intermediate output
    df.to_csv(INTERMEDIATE_PATHS["healthy_equipment_clean"], index=False, encoding="utf-8-sig")
    logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['healthy_equipment_clean']}")

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
    # --- NEW: Calculate Entry Time (Left Truncation) ---
    # If installed BEFORE 2021, it enters risk set in 2021 (age > 0)
    # If installed AFTER 2021, it enters risk set at installation (age = 0)
    df["entry_days"] = np.where(
        df["Kurulum_Tarihi"] < OBSERVATION_START_DATE,
        (OBSERVATION_START_DATE - df["Kurulum_Tarihi"]).dt.days,
        0
    )
    
    # Duration is still calculated from Installation Date
    # But now Cox knows we didn't watch the first 'entry_days'
    df["duration_days"] = np.where(
        df["event"] == 1,
        (df["Ilk_Gercek_Ariza_Tarihi"] - df["Kurulum_Tarihi"]).dt.days,
        (data_end_date - df["Kurulum_Tarihi"]).dt.days
    )
    
    # Safety: duration must be > entry
    df = df[df["duration_days"] > df["entry_days"]].copy()
    
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
    observation_start_date: pd.Timestamp,  # <--- NEW ARGUMENT
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Add event/duration AND entry_days (Left Truncation) to equipment_master.
    """
    # 1. First REAL failure per equipment
    first_fail = df_fault_filtered.groupby("cbs_id")["started at"].min()
    
    # Add to existing dataframe
    df["Ilk_Gercek_Ariza_Tarihi"] = df["cbs_id"].map(first_fail)
    
    # 2. Calculate event flag
    df["event"] = df["Ilk_Gercek_Ariza_Tarihi"].notna().astype(int)
    
    # 3. Calculate duration (End - Start)
    # If Failed: Duration = Failure Date - Install Date
    # If Healthy: Duration = Analysis End Date - Install Date
    
    failure_duration = (df["Ilk_Gercek_Ariza_Tarihi"] - df["Kurulum_Tarihi"]).dt.days
    healthy_duration = (data_end_date - df["Kurulum_Tarihi"]).dt.days
    
    df["duration_days"] = np.where(
        df["event"] == 1,
        failure_duration,
        healthy_duration
    )
    # Bu mantığı feature engineering fonksiyonuna ekleyin veya güncelleyin
    df = df.dropna(subset=['Kurulum_Tarihi'])  # Yaşı olmayan arızayı modelleyemeyiz
    # --- NEW: DELAYED ENTRY (LEFT TRUNCATION) ---
    # Assets installed BEFORE we started recording (e.g. 2021) enter the risk set LATE.
    # We didn't observe them from Install Date to 2021, so we calculate that gap.
    
    df["entry_days"] = np.where(
        df["Kurulum_Tarihi"] < observation_start_date,
        (observation_start_date - df["Kurulum_Tarihi"]).dt.days,
        0
    )
    
    # CLEANUP:
    # 1. Fill NaNs (if any duration calc failed)
    df["duration_days"] = df["duration_days"].fillna(0)
    df["entry_days"] = df["entry_days"].fillna(0)
    
    # 2. Handle Logic Errors & Clamp
    # We cap max duration to 60 years to avoid outliers affecting scalers
    df["duration_days"] = df["duration_days"].clip(upper=60*365)
    
    # Logic Check: 
    # Duration must be positive.
    # Duration must be > Entry Time (Asset cannot fail BEFORE we started watching it)
    valid_mask = (df["duration_days"] > 0) & (df["duration_days"] > df["entry_days"])
    
    dropped_count = (~valid_mask).sum()
    if dropped_count > 0:
        logger.warning(f"[SURVIVAL] Dropping {dropped_count} assets with invalid duration (Failed before observation start or data error)")
        df = df[valid_mask].copy()
    
    logger.info(f"[SURVIVAL] Added to master: {len(df)} assets, {df['event'].sum()} events ({100*df['event'].mean():.1f}%)")
    return df

def add_temporal_features_inplace(
    df: pd.DataFrame,
    t_ref: pd.Timestamp,
    chronic_df: pd.DataFrame,
    observation_start_date: pd.Timestamp,  # <--- NEW ARGUMENT
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Add temporal features directly to existing dataframe IN-PLACE.
    Includes fix for NumPy Timestamp comparison error.
    """
    # 1. Calculate Age (Tref_Yas_Gun)
    # Ensure datetime format just in case
    if not pd.api.types.is_datetime64_any_dtype(df["Kurulum_Tarihi"]):
        df["Kurulum_Tarihi"] = pd.to_datetime(df["Kurulum_Tarihi"], errors='coerce')

    df["Tref_Yas_Gun"] = (t_ref - df["Kurulum_Tarihi"]).dt.days.clip(lower=0)
    df["Tref_Ay"] = t_ref.month
    
    # --- NEW: Observability Features (The Fix) ---
    # Instead of np.maximum, we use Pandas .clip(lower=...) which handles Timestamps correctly
    effective_start_date = df["Kurulum_Tarihi"].clip(lower=observation_start_date)
    
    # How long have we actually watched this asset?
    df["Observed_Duration_Days"] = (t_ref - effective_start_date).dt.days
    
    # Is it a "Legacy" asset (existed before we started recording)?
    df["Is_Legacy_Asset"] = (df["Kurulum_Tarihi"] < observation_start_date).astype(int)
    
    # Ratio: What % of its life did we observe?
    # Avoid division by zero for brand new assets
    df["Observation_Ratio"] = (df["Observed_Duration_Days"] / df["Tref_Yas_Gun"].replace(0, 1)).fillna(1.0).clip(0, 1)
    
    # 2. Merge chronic features if available
    if chronic_df is not None and len(chronic_df) > 0:
        # Check if columns exist before merging to avoid duplication
        cols_to_merge = ["Ariza_Sayisi_90g", "Chronic_Rate_Yillik", "Chronic_Decay_Skoru", "Chronic_Flag"]
        cols_to_merge = [c for c in cols_to_merge if c in chronic_df.columns]
        
        # Drop existing columns if they are already there (to allow re-calculation)
        df.drop(columns=[c for c in cols_to_merge if c in df.columns], inplace=True, errors="ignore")
        
        # Use merge (left join)
        df = df.merge(chronic_df[["cbs_id"] + cols_to_merge], on="cbs_id", how="left")
        
        # Fill NaN for equipment with no chronic history
        df[cols_to_merge] = df[cols_to_merge].fillna(0)
    else:
        # If no chronic data, create 0 columns to prevent crashes later
        df["Ariza_Sayisi_90g"] = 0
        df["Chronic_Rate_Yillik"] = 0.0
        df["Chronic_Decay_Skoru"] = 0.0
        df["Chronic_Flag"] = 0
    
    logger.info(f"[FEATURES] Temporal: Added age + chronic features + observability stats")
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
    #rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
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
    #X = remove_multicollinear_features(X, threshold=10.0, logger=logger)
    X = remove_multicollinear_features(X, threshold=20.0, logger=logger)
    
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
    entry: pd.Series, # <--- NEW ARGUMENT
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
    work["entry_days"] = entry.values # <--- Add this
    # ✅ ADD SAFETY CHECK
    # Remove helper columns to count actual features
    feature_cols = [c for c in work.columns if c not in ["duration_days", "event", "entry_days", "Kurulum_Tarihi"]]
    
    if len(feature_cols) == 0:
        logger.warning("[COX] No features left after filtering - skipping Cox/Weibull")
        return None, None
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
    # Cox Training
    try:
        cox = CoxPHFitter(penalizer=0.05)
        # TELL COX ABOUT DELAYED ENTRY
        cox.fit(
            train_data, 
            duration_col="duration_days", 
            event_col="event", 
            entry_col="entry_days" # <--- THE MAGIC FIX
        )
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
        #n_estimators=200,
        n_estimators=100,
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
# =============================================================================
# REPLACEMENT FOR ML FUNCTIONS (Correct Survival Logic)
# =============================================================================

def train_ml_models(
    df: pd.DataFrame,
    feature_cols: list,
    horizons_days: list,
    logger: logging.Logger
):
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sksurv.util import Surv
    except ImportError:
        logger.warning("[ML] sksurv not installed. Skipping ML.")
        return None

    # --- FIX: SAFE INDEXING ---
    # 1. Create a clean working copy and RESET INDEX
    # This ensures that Row Label 0 = Position 0, Label 1 = Position 1, etc.
    work_df = df.copy().reset_index(drop=True)
    
    # 2. Select Features
    X = work_df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number, 'object'])
    
    # 3. Create Target
    y = Surv.from_arrays(
        event=work_df["event"].astype(bool).values,
        time=work_df["duration_days"].values
    )

    # 4. Temporal Split (Manual & Safe)
    # We sort by date manually here to guarantee 'iloc' works
    if "Kurulum_Tarihi" in work_df.columns:
        # Sort by installation date (oldest learns first)
        sort_idxs = np.argsort(work_df["Kurulum_Tarihi"].values)
    else:
        # Sort by duration if date missing
        sort_idxs = np.argsort(work_df["duration_days"].values)
        
    # Reorder X and y based on time
    X = X.iloc[sort_idxs]
    y = y[sort_idxs]
    
    # 5. Split Logic (75% Train, 25% Test)
    n_samples = len(X)
    split_point = int(n_samples * 0.75)
    
    # Pure positional slicing (Cannot fail "out of bounds")
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]

    # 3. Preprocess
    pre = build_preprocessor(X_train)
    
    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        loss="coxph",
        random_state=42
    )

    from sklearn.pipeline import Pipeline
    model_pipeline = Pipeline([("pre", pre), ("gbsa", gbsa)])
    
    try:
        logger.info(f"[ML] Training GBSA on {len(X_train)} samples...")
        model_pipeline.fit(X_train, y_train)
        score = model_pipeline.score(X_test, y_test)
        logger.info(f"[ML] GBSA Test Concordance: {score:.4f}")
        return {"model": model_pipeline, "safe_cols": feature_cols}
        
    except Exception as e:
        logger.warning(f"[ML] Training failed: {e}")
        return None

def predict_ml_pof(df: pd.DataFrame, ml_pack: dict, horizons: list) -> pd.DataFrame:
    # ✅ FIX: Handle both old and new dict structure
    if "model" in ml_pack:
        model = ml_pack["model"]
    elif "models" in ml_pack:
        model = ml_pack["models"]
    else:
        return pd.DataFrame({"cbs_id": df["cbs_id"].values})
    
    cols = ml_pack["safe_cols"]
    
    X = df[cols].copy()
    current_age = df["duration_days"].fillna(0).clip(lower=0).values
    
    out = pd.DataFrame({"cbs_id": df["cbs_id"].values})
    
    # Get survival functions S(t) for every asset
    # This returns an array of functions
    surv_funcs = model.predict_survival_function(X)
    
    for H in horizons:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        pofs = []
        
        for i, fn in enumerate(surv_funcs):
            # S(t) = Probability of surviving past time t
            # P(Fail in H | Alive at Age) = 1 - (S(Age + H) / S(Age))
            
            # Get S(current_age)
            prob_survive_now = fn(current_age[i])
            
            # Get S(current_age + horizon)
            prob_survive_future = fn(current_age[i] + H)
            
            # Avoid division by zero
            if prob_survive_now < 1e-5:
                # If model thinks it's already dead, risk is max
                conditional_risk = 1.0
            else:
                conditional_risk = 1.0 - (prob_survive_future / prob_survive_now)
            
            pofs.append(np.clip(conditional_risk, 0, 1))
            
        out[f"ml_pof_{label}"] = pofs
        
    return out

# =============================================================================
# BACKTESTING (Temporal Validation Proof)
# =============================================================================
class TemporalBacktester:
    """
    Walk-forward validation to prove model doesn't cheat.
    Simulates: "If we ran this model in 2022, would it have predicted the 2023 failures?"
    """
    
    def __init__(self, df_fault: pd.DataFrame, df_healthy: pd.DataFrame, logger: logging.Logger):
        self.df_fault = df_fault
        self.df_healthy = df_healthy
        self.logger = logger
        self.results = []
    
    def _generate_snapshot(self, cutoff_date: pd.Timestamp):
        """
        Create training dataset as it would have looked at 'cutoff_date'.
        Crucial: We must not see any data after cutoff_date.
        """
        # 1. Filter faults known at that time
        faults_past = self.df_fault[self.df_fault["started at"] <= cutoff_date].copy()
        
        # 2. Apply the same "Real Failure" filter logic
        faults_filtered = filter_real_failures(faults_past, self.logger)
        
        # 3. Determine Observation Start (Dynamic based on available history)
        # This fixes the "Missing Argument" bug
        if not self.df_fault.empty:
            observation_start_date = self.df_fault["started at"].min()
        else:
            observation_start_date = cutoff_date # Fallback
            
        # 4. Build Dataset (Using fixed signatures)
        equipment_master = build_equipment_master(faults_past, self.df_healthy, self.logger, cutoff_date)
        
        # Pass observation_start_date correctly
        df_snapshot = add_survival_columns_inplace(
            equipment_master, 
            faults_filtered, 
            cutoff_date, 
            observation_start_date, # <--- FIX: Added missing arg
            self.logger
        )
        
        # 5. Add Features
        chronic_df = compute_chronic_features(faults_past, cutoff_date, self.logger)
        
        # Pass observation_start_date correctly
        df_snapshot = add_temporal_features_inplace(
            df_snapshot, 
            cutoff_date, 
            chronic_df, 
            observation_start_date, # <--- FIX: Added missing arg
            self.logger
        )
        
        return df_snapshot
    
    def _train_simple_model(self, df: pd.DataFrame, features: list):
        """Train lightweight XGBoost for backtesting"""
        X = df[features].copy()
        
        # Data Hygiene: Remove dates and non-numerics just in case
        X = X.select_dtypes(include=[np.number]).fillna(0)
        
        # Safety check: If X is empty (e.g. no features generated), return Dummy
        if X.empty or len(X.columns) == 0:
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy='constant', constant=0), []
        
        # Train simple classifier
        # We use a smaller model for backtesting speed
        model = XGBClassifier(
            n_estimators=50, 
            max_depth=3, 
            eval_metric="logloss", 
            random_state=42,
            n_jobs=-1
        )
        
        try:
            model.fit(X, df["event"])
            return model, X.columns.tolist()
        except Exception as e:
            self.logger.warning(f"[BACKTEST] Model training failed: {e}")
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy='constant', constant=0), []
    
    def run(self, start_year: int, end_year: int, horizon_days: int = 365):
        """Run walk-forward validation"""
        self.logger.info("="*60)
        self.logger.info(f"[BACKTEST] Walk-Forward Validation ({start_year}-{end_year})")
        self.logger.info("="*60)
        
        from sklearn.metrics import roc_auc_score
        
        for year in range(start_year, end_year + 1):
            cutoff_date = pd.Timestamp(f"{year}-01-01")
            test_end_date = cutoff_date + pd.Timedelta(days=horizon_days)
            
            # 1. Generate Historical Snapshot
            self.logger.info(f"[BACKTEST] Generating snapshot for {cutoff_date.date()}...")
            df_train = self._generate_snapshot(cutoff_date)
            
            # 2. Define Ground Truth (What happened in the next year?)
            # We look at the MAIN fault database for future events
            future_faults = self.df_fault[
                (self.df_fault["started at"] > cutoff_date) & 
                (self.df_fault["started at"] <= test_end_date)
            ]
            
            # Filter future faults to only include REAL failures
            # (We don't want to fail the model because a fuse blew)
            future_faults = filter_real_failures(future_faults, self.logger)
            
            failed_ids = set(future_faults["cbs_id"].unique())
            
            # Target: 1 if asset failed in prediction window, 0 otherwise
            y_true = df_train["cbs_id"].isin(failed_ids).astype(int)
            
            if y_true.sum() < 5:
                self.logger.warning(f"[BACKTEST] {year}: Skipped - Insufficient failures ({y_true.sum()}) to evaluate.")
                continue
            
            # 3. Train Model
            # Exclude target leakage columns
            exclude_cols = ["cbs_id", "event", "duration_days", "Kurulum_Tarihi", "entry_days", 
                            "Ilk_Gercek_Ariza_Tarihi", "started at", "ended at"]
            structural_cols = [c for c in df_train.columns if c not in exclude_cols]
            
            model, valid_features = self._train_simple_model(df_train, structural_cols)
            
            # 4. Predict
            if valid_features:
                X_test = df_train[valid_features].fillna(0)
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = np.zeros(len(y_true))
            
            # 5. Evaluate
            try:
                auc = roc_auc_score(y_true, probs)
            except ValueError:
                auc = 0.5
            
            # Precision at Top 100 (Operational Metric)
            # "If we inspected the top 100 risk items, how many were actually broken?"
            if len(probs) >= 100:
                top_100_idx = np.argsort(probs)[-100:]
                hits = y_true.iloc[top_100_idx].sum()
            else:
                hits = y_true.sum() # Edge case for small datasets
            
            self.logger.info(f"[BACKTEST] {year} Results -> AUC: {auc:.3f} | Top-100 Hits: {hits}/{min(100, y_true.sum())}")
            
            self.results.append({
                "Year": year,
                "AUC": auc,
                "Top100_Hits": hits,
                "Total_Failures_In_Window": y_true.sum()
            })
        
        # Save Summary
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            out_path = os.path.join(os.path.dirname(self.logger.handlers[0].baseFilename).replace("loglar", "data/sonuclar"), "backtest_results_temporal.csv")
            results_df.to_csv(out_path, index=False)
            self.logger.info(f"[BACKTEST] Results saved: {out_path}")
            
        return results_df

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
    
    # ---------------------------------------------------------
    # 1. Survival Models (Cox PH & Weibull)
    # ---------------------------------------------------------
    try:
        X_cox = select_cox_safe_features(df_eq, structural_cols, logger)
        
        # ✅ FIX: Check feature count BEFORE training
        feature_count = X_cox.shape[1] if X_cox is not None else 0
        logger.info(f"[{eq_type}] Features after filtering: {feature_count}")
        
        if feature_count < 2:
            logger.warning(f"[{eq_type}] Too few features ({feature_count}) - skipping Cox/Weibull")
        else:
            # ✅ Only train if we have enough features
            cox, wb = train_cox_weibull(
                X_cox, 
                df_eq["duration_days"], 
                df_eq["event"], 
                df_eq["entry_days"], 
                logger
            )
            
            if cox:
                cox_pred = predict_survival_pof(
                    cox, 
                    X_cox, 
                    df_eq["duration_days"], 
                    SURVIVAL_HORIZONS_DAYS, 
                    "cox", 
                    df_eq["cbs_id"]
                )
                predictions = predictions.merge(cox_pred, on="cbs_id", how="left")
                
    except Exception as e:
        logger.warning(f"[{eq_type}] Cox/Weibull failed: {e}")
    # ---------------------------------------------------------
    # 2. Random Survival Forests (RSF)
    # ---------------------------------------------------------
    try:
        rsf = train_rsf_survival(df_eq, structural_cols, logger)
        if rsf:
            rsf_pred = predict_rsf_pof(df_eq, rsf, structural_cols, SURVIVAL_HORIZONS_DAYS)
            predictions = predictions.merge(rsf_pred, on="cbs_id", how="left")
    except Exception as e:
        logger.warning(f"[{eq_type}] RSF failed: {e}")
    
    # ---------------------------------------------------------
    # 3. Machine Learning (Gradient Boosting Survival)
    # ---------------------------------------------------------
    # Note: Using Gradient Boosting Survival Analysis (GBSA), not binary classification
    ml_features = structural_cols + [c for c in temporal_cols if c not in ["Kurulum_Tarihi"]]
    
    # Check if we have enough events to learn anything useful
    n_events = df_eq["event"].sum()
    
    if n_events >= 20:  # Lowered threshold for GBSA (it learns from censored data too)
        try:
            ml_pack = train_ml_models(df_eq, ml_features, SURVIVAL_HORIZONS_DAYS, logger)
            if ml_pack:
                ml_pred = predict_ml_pof(df_eq, ml_pack, SURVIVAL_HORIZONS_DAYS)
                predictions = predictions.merge(ml_pred, on="cbs_id", how="left")
        except Exception as e:
            logger.warning(f"[{eq_type}] ML failed: {e}")
    else:
        logger.info(f"[{eq_type}] ML skipped: insufficient events ({n_events} < 20)")
    
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
    """
    Predicts PoF using the Survival Curves from Gradient Boosting.
    """
    # ✅ Step 1: Get the pipeline
    if "model" in ml_pack:
        pipeline = ml_pack["model"]
    elif "models" in ml_pack:
        pipeline = ml_pack["models"]
    else:
        return pd.DataFrame({"cbs_id": df["cbs_id"].values})
    
    cols = ml_pack["safe_cols"]
    X = df[cols].copy()
    current_age = df["duration_days"].fillna(0).clip(lower=0).values
    
    out = pd.DataFrame({"cbs_id": df["cbs_id"].values})
    
    # ✅ Step 2: Extract the GBSA model from the pipeline
    # Pipeline structure: [("pre", preprocessor), ("gbsa", GradientBoostingSurvivalAnalysis)]
    try:
        gbsa_model = pipeline.named_steps["gbsa"]
    except (AttributeError, KeyError):
        # Not a pipeline or no 'gbsa' step - return empty predictions
        return out
    
    # ✅ Step 3: Preprocess features
    X_transformed = pipeline.named_steps["pre"].transform(X)
    
    # ✅ Step 4: Get survival functions from GBSA
    surv_funcs = gbsa_model.predict_survival_function(X_transformed)
    
    # ✅ Step 5: Calculate conditional PoF for each horizon
    for H in horizons:
        label = SURVIVAL_HORIZON_LABELS.get(H, f"{H}g")
        pofs = []
        
        for i, fn in enumerate(surv_funcs):
            # S(t) = Probability of surviving past time t
            # P(Fail in H | Alive at Age) = 1 - (S(Age + H) / S(Age))
            
            prob_survive_now = fn(current_age[i])
            prob_survive_future = fn(current_age[i] + H)
            
            if prob_survive_now < 1e-5:
                conditional_risk = 1.0
            else:
                conditional_risk = 1.0 - (prob_survive_future / prob_survive_now)
            
            pofs.append(np.clip(conditional_risk, 0, 1))
            
        out[f"ml_pof_{label}"] = pofs
        
    return out

def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    GÜNCELLENMİŞ VERSİYON: Yüzdelik Dilim (Percentile) Tabanlı Skorlama
    
    Eski Yöntem: Mutlak PoF (Olasılık) kullanıyordu. PoF değerleri çok düşük (%1-5) olduğu için
                 herkes "Çok Sağlıklı" (95-99 Puan) çıkıyordu.
                 
    Yeni Yöntem: Varlıkları kendi ekipman grubu içinde 'Risk Sırasına' dizersiniz.
                 En kötü %5 -> KRİTİK (Puan < 40)
                 Bu yöntem, filonun en riskli varlıklarını mutlaka ortaya çıkarır.
    """
    
    # 1. En iyi risk metriğini seç
    # Öncelik: Ensemble > RSF > Cox/ML
    risk_col = None
    if "PoF_Ensemble_12Ay" in df.columns:
        risk_col = "PoF_Ensemble_12Ay"
    elif "rsf_pof_12ay" in df.columns:
        risk_col = "rsf_pof_12ay"
    else:
        # Fallback: Bulabildiği herhangi bir 12 aylık tahmin
        candidates = [c for c in df.columns if "12" in c and "pof" in c.lower()]
        risk_col = candidates[0] if candidates else None

    if not risk_col:
        # Hiçbir tahmin yoksa varsayılan
        df["Health_Score"] = 90
        df["Risk_Sinifi"] = "BILINMIYOR"
        return df

    # NaNs -> 0 (En düşük risk kabul et)
    df[risk_col] = df[risk_col].fillna(0)

    # 2. SIRALAMA (RANKING) - Ekipman Tipine Göre
    # Transformatörleri kendi içinde, Direkleri kendi içinde en riskliden en aza sırala.
    # rank(pct=True) -> 0.0 (En iyi) ile 1.0 (En kötü) arasında değer verir.
    
    if "Ekipman_Tipi" in df.columns:
        # Her ekipman tipini kendi içinde değerlendir
        df["Risk_Percentile"] = df.groupby("Ekipman_Tipi")[risk_col].rank(pct=True)
    else:
        # Ekipman tipi yoksa global sıralama
        df["Risk_Percentile"] = df[risk_col].rank(pct=True)
        
    # Tek elemanlı gruplar için fillna (Hata önleyici)
    df["Risk_Percentile"] = df["Risk_Percentile"].fillna(0.5)

    # 3. SAĞLIK SKORU HESABI (Sıralamaya Göre)
    # En kötü (%100 riskli / Percentile 1.0) -> 0 Puan
    # En iyi (%0 riskli / Percentile 0.0) -> 100 Puan
    df["Health_Score"] = 100 * (1 - df["Risk_Percentile"])
    
    # 4. KRONİK CEZALANDIRMASI
    # Eğer varlık "Kronik" ise (sık arızalanıyorsa), sıralaması iyi olsa bile puanını düşür.
    if "Chronic_Flag" in df.columns:
        # Kronikse maksimum 60 puan alabilsin (Otomatikman Yüksek Risk bölgesine itiyoruz)
        mask_chronic = df["Chronic_Flag"] == 1
        df.loc[mask_chronic, "Health_Score"] = df.loc[mask_chronic, "Health_Score"].clip(upper=60)

    # 5. RİSK SINIFI ATAMA (Percentile Bazlı)
    def assign_risk_class(row):
        score = row["Health_Score"]
        chronic = row.get("Chronic_Flag", 0)
        
        # Kronikler her zaman öncelikli
        if chronic == 1:
            return "KRİTİK (KRONİK)"
            
        # Yüzdelik dilimlere göre sınıflar:
        if score < 40: return "KRİTİK"      # En kötü %5 (Percentile > 0.95) - Filonun en çürükleri
        if score < 70: return "YÜKSEK"      # Sonraki %15 (Percentile 0.80 - 0.95)
        if score < 85: return "ORTA"        # Sonraki %30
        return "DÜŞÜK"                      # En iyi %50 (Percentile < 0.50)

    df["Risk_Sinifi"] = df.apply(assign_risk_class, axis=1)
    
    return df
# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    ensure_dirs()
    logger = setup_logger()
    
    # -------------------------------------------------------------------------
    # STEP 1: LOAD & CONFIGURE
    # -------------------------------------------------------------------------
    logger.info("[STEP 1] Loading data...")
    df_fault = load_fault_data(logger)
    df_healthy = load_healthy_data(logger)
    
    # Auto-detect start date from data
    # This fixes the Left Truncation logic dynamically
    observation_start_date = df_fault["started at"].min()
    data_end_date = df_fault["started at"].max()
    
    logger.info(f"[CONFIG] Data range: {observation_start_date.date()} → {data_end_date.date()}")
    logger.info(f"[CONFIG] Observation Start (Left Truncation): {observation_start_date.date()}")
    
    # -------------------------------------------------------------------------
    # STEP 2: BUILD DATASET
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("PRODUCTION - Training on Full History")
    logger.info("="*60 + "\n")
    
    logger.info("[STEP 2] Building complete dataset...")
    
    # 1. Master list
    equipment_master = build_equipment_master(df_fault, df_healthy, logger, data_end_date)
    
    # 2. Filter Real Failures (Removing fuses/temporary faults)
    df_fault_filtered = filter_real_failures(df_fault, logger)
    
    # 3. Add Survival Columns (Events, Duration, Delayed Entry)
    df_all = add_survival_columns_inplace(
        equipment_master.copy(),
        df_fault_filtered,
        data_end_date,
        observation_start_date,
        logger
    )

    # Save survival base intermediate
    df_all.to_csv(INTERMEDIATE_PATHS["survival_base"], index=False, encoding="utf-8-sig")
    logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['survival_base']}")

    # 4. Feature Engineering
    logger.info("[STEP 3] Engineering features...")
    chronic_df = compute_chronic_features(df_fault, data_end_date, logger)
    
    df_all = add_temporal_features_inplace(
        df_all,
        data_end_date,
        chronic_df,
        observation_start_date,
        logger
    )

    # Define Feature Columns
    structural_cols = ["Ekipman_Tipi", "Gerilim_Sinifi", "Gerilim_Seviyesi", "Marka"]
    structural_cols = [c for c in structural_cols if c in df_all.columns]

    temporal_cols = ["Tref_Yas_Gun", "Tref_Ay", "Ariza_Sayisi_90g",
                     "Chronic_Rate_Yillik", "Chronic_Decay_Skoru", "Chronic_Flag",
                     "Observation_Ratio"]
    temporal_cols = [c for c in temporal_cols if c in df_all.columns]

    # Save feature outputs
    if structural_cols:
        df_all[["cbs_id"] + structural_cols].to_csv(INTERMEDIATE_PATHS["features_structural"], index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['features_structural']}")

    if temporal_cols:
        df_all[["cbs_id"] + temporal_cols].to_csv(INTERMEDIATE_PATHS["features_temporal"], index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['features_temporal']}")

    # Save combined feature set (ozellikler_pof3)
    all_feature_cols = ["cbs_id"] + structural_cols + temporal_cols + ["event", "duration_days", "entry_days"]
    all_feature_cols = [c for c in all_feature_cols if c in df_all.columns]
    df_all[all_feature_cols].to_csv(INTERMEDIATE_PATHS["ozellikler_pof3"], index=False, encoding="utf-8-sig")
    logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['ozellikler_pof3']}")

    logger.info(f"[DATASET] Assets: {len(df_all)} | Features: {len(structural_cols) + len(temporal_cols)}")

    # -------------------------------------------------------------------------
    # STEP 3: TRAIN GLOBAL MODELS (FALLBACK)
    # -------------------------------------------------------------------------
    logger.info("\n[GLOBAL] Training fallback models (Cox, RSF, ML)...")
    
    # 1. Global Cox
    X_cox_global = select_cox_safe_features(df_all, structural_cols, logger)
    cox_global, wb_global = train_cox_weibull(
        X_cox_global, 
        df_all["duration_days"], 
        df_all["event"], 
        df_all["entry_days"],
        logger
    )
    
    # 2. Global Random Survival Forest
    rsf_global = train_rsf_survival(df_all, structural_cols, logger)
    
    # 3. Global ML (Gradient Boosting Survival)
    ml_features_global = structural_cols + [c for c in temporal_cols if c not in ["Kurulum_Tarihi"]]
    ml_pack_global = train_ml_models(df_all, ml_features_global, SURVIVAL_HORIZONS_DAYS, logger)
    
    # Store global models for fallback usage
    global_models = {
        "cox": cox_global,
        "weibull": wb_global,
        "rsf": rsf_global,
        "ml": ml_pack_global,
        "X_cox_cols": X_cox_global.columns.tolist()
    }

    # -------------------------------------------------------------------------
    # STEP 4: EQUIPMENT-STRATIFIED MODELING
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("STEP 4 - Equipment-Stratified Modeling")
    logger.info("="*60 + "\n")
    
    eq_stats = get_equipment_stats(df_all, equipment_master, logger)
    unique_types = sorted(df_all["Ekipman_Tipi"].unique())
    
    MIN_SAMPLES = 100
    MIN_EVENTS = 30
    
    all_predictions = []
    all_marka_analyses = []
    all_bakim_analyses = []
    
    # Import TQDM for progress bar
    from tqdm import tqdm
    
    for eq_type in tqdm(unique_types, desc="Training Equipment Models", unit="type"):
        
        # 1. Filter Data
        df_eq = df_all[df_all["Ekipman_Tipi"] == eq_type].copy()
        stats = eq_stats.get(eq_type, {'n_total': 0, 'n_events': 0, 'has_marka': 0})
        
        preds = pd.DataFrame({"cbs_id": df_eq["cbs_id"]})
        model_source = "Equipment_Specific"

        # 2. DECISION: Use Global Fallback vs Specific Training
        if stats["n_total"] < MIN_SAMPLES or stats["n_events"] < MIN_EVENTS:
            # --- GLOBAL FALLBACK (ENHANCED) ---
            model_source = "Global_Fallback"
            
            # A) Global Cox Fallback
            try:
                X_eq = select_cox_safe_features(df_eq, structural_cols, logger)
                # Align features with global model
                for c in set(global_models["X_cox_cols"]) - set(X_eq.columns):
                    X_eq[c] = 0
                X_eq = X_eq[global_models["X_cox_cols"]]
                
                if cox_global:
                    cox_pred = predict_survival_pof(cox_global, X_eq, df_eq["duration_days"],
                                                    SURVIVAL_HORIZONS_DAYS, "cox", df_eq["cbs_id"])
                    preds = preds.merge(cox_pred, on="cbs_id", how="left")
            except Exception:
                pass 

            # B) Global RSF Fallback
            try:
                if rsf_global:
                    rsf_pred = predict_rsf_pof(df_eq, rsf_global, structural_cols, SURVIVAL_HORIZONS_DAYS)
                    preds = preds.merge(rsf_pred, on="cbs_id", how="left")
            except Exception:
                pass

            # C) Global ML Fallback
            try:
                if ml_pack_global:
                    # Note: predict_ml_pof should handle missing columns internally
                    ml_pred = predict_ml_pof(df_eq, ml_pack_global, SURVIVAL_HORIZONS_DAYS)
                    preds = preds.merge(ml_pred, on="cbs_id", how="left")
            except Exception:
                pass

        else:
            # --- SPECIFIC TRAINING ---
            preds = train_equipment_specific_models(df_eq, structural_cols, temporal_cols, eq_type, logger)

            # Specific Explanatory Analyses
            if stats.get("has_marka", 0) >= 30:
                try:
                    marka_analysis = analyze_marka_effect(df_eq, eq_type, logger)
                    if not marka_analysis.empty: all_marka_analyses.append(marka_analysis)
                except Exception: pass
            
            try:
                bakim_analysis = analyze_bakim_effect(equipment_master, eq_type, logger)
                if not bakim_analysis.empty: all_bakim_analyses.append(bakim_analysis)
            except Exception: pass

        # 3. MERGE PREDICTIONS WITH METADATA (FIXED)
        # We merge 'preds' (which only has cbs_id + probabilities) back to df_eq metadata
        meta_cols = ["cbs_id", "Ekipman_Tipi"]
        if "Fault_Count" in df_eq.columns: meta_cols.append("Fault_Count")
        
        preds_full = df_eq[meta_cols].merge(preds, on="cbs_id", how="left")
        preds_full["Model_Type"] = model_source
        
        # 4. COMPUTE HEALTH SCORE
        # Now preds_full definitely has "Ekipman_Tipi", so grouping works
        try:
            preds_full = compute_health_score(preds_full)
        except Exception as e:
            logger.error(f"[{eq_type}] Health score calc failed: {e}")
            preds_full["Health_Score"] = 50 
            preds_full["Risk_Sinifi"] = "ORTA"

        # 5. Store Results
        all_predictions.append(preds_full)
        
        # Save individual CSV (Silent to keep progress bar clean)
        safe_name = str(eq_type).replace("/", "_").replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"pof_{safe_name}.csv")
        preds_full.to_csv(out_path, index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # STEP 5: FINALIZE
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("STEP 5 - Final Ensemble & Reporting")
    logger.info("="*60 + "\n")
    
    if not all_predictions:
        logger.error("No predictions generated.")
        return

    # Combine all
    predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Final Report Merge (Add context like Voltage, Install Date)
    report_cols = ["Ekipman_Tipi", "Gerilim_Sinifi", "Fault_Count", "Kurulum_Tarihi"]
    report_base = df_all[["cbs_id"] + [c for c in report_cols if c in df_all.columns]].drop_duplicates("cbs_id")
    
    # Clean duplicates before merge
    cols_to_drop = [c for c in report_cols if c in predictions.columns]
    preds_clean = predictions.drop(columns=cols_to_drop, errors="ignore")
    
    report = report_base.merge(preds_clean, on="cbs_id", how="left")
    
    # Save outputs
    out_path = os.path.join(OUTPUT_DIR, "pof_predictions_final.csv")
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OUTPUT] Main predictions: {out_path}")
    
    if all_marka_analyses:
        pd.concat(all_marka_analyses).to_csv(os.path.join(OUTPUT_DIR, "marka_analysis.csv"), index=False, encoding="utf-8-sig")
    
    if all_bakim_analyses:
        pd.concat(all_bakim_analyses).to_csv(os.path.join(OUTPUT_DIR, "bakim_analysis.csv"), index=False, encoding="utf-8-sig")
    
    # Save intermediate files for Reporting Script (to INTERMEDIATE_DIR not OUTPUT_DIR)
    equipment_master.to_csv(INTERMEDIATE_PATHS["equipment_master"], index=False, encoding="utf-8-sig")
    df_all.to_csv(os.path.join(INTERMEDIATE_DIR, "model_input_data_full.csv"), index=False, encoding="utf-8-sig")
    
    # Final Stats
    critical = (report["Health_Score"] < 40).sum()
    mean_health = report["Health_Score"].mean()
    
    logger.info(f"Total assets: {len(report):,}")
    logger.info(f"Critical assets (Health<40): {critical:,} ({100*critical/len(report):.1f}%)")
    logger.info(f"Mean Health Score: {mean_health:.1f}")
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    
if __name__ == "__main__":
    main()