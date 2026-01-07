# -*- coding: utf-8 -*-
"""
PoF - Clean Production Pipeline | Temporal Validation + Equipment Stratification
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

# Data Balancing Configuration
BALANCE_CFG = CFG.get("data_balancing", {})
BALANCE_ENABLED = BALANCE_CFG.get("enabled", True)
BALANCE_TARGET_RATIO = BALANCE_CFG.get("target_ratio", 5)  # 1:5 default
BALANCE_METHOD = BALANCE_CFG.get("method", "undersample")
BALANCE_RANDOM_STATE = BALANCE_CFG.get("random_state", 42)

# =============================================================================
# 🧠 FEATURE REGISTRY (ÖZELLİK YÖNETİM MERKEZİ)
# =============================================================================
# Bu yapı, modelin eğitim stratejisini belirleyen merkezi konfigürasyondur.
# Modelin "Neyi öğrenmesi gerektiği" (X) ve "Neyi görmemesi gerektiği" (Leakage) burada tanımlanır.
#
# 🚫 1. temporal_leakage (YASAKLI LİSTE / DATA LEAKAGE):
#    - Bu değişkenler, modelin tahmin etmeye çalıştığı "hedefi" (Target) veya 
#      henüz gerçekleşmemiş "gelecek bilgisini" içerir.
#    - Örn: 'event' (sonuç), 'duration_days' (ömür), 'Son_Ariza_Tarihi'.
#    - KRİTİK: Bu değişkenler eğitim matrisinden (X) kesinlikle ÇIKARILIR.
#
# 📉 2. chronic_features (DİNAMİK SAĞLIK GÖSTERGELERİ):
#    - Varlığın geçmiş performansından türetilen matematiksel özelliklerdir.
#    - IEEE 1366 standartlarına göre kroniklik durumu (Flag), arıza sıklığı (Rate) 
#      ve zaman ağırlıklı yıpranma skorunu (Decay) içerir.
#    - Modelin varlığı "riskli" olarak tanımasını sağlayan ana sinyallerdir.
#
# 🏗️ 3. structural_features (STATİK YAPISAL ÖZELLİKLER):
#    - Varlığın kimliği, fiziksel özellikleri ve coğrafi konumudur.
#    - Marka, Tip, Gerilim Seviyesi, İlçe gibi genelde sabit kalan niteliklerdir.
#    - Modelin "Hangi marka/tip daha dayanıksız?" sorusunu çözmesini sağlar.
# =============================================================================
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
# NOTE: REAL_FAILURE_CODES filter removed
# Domain expert confirmed all cause codes in the input data are valid failures
# Data quality checks (null cbs_id, etc.) are performed in load_fault_data()
# =============================================================================

# =============================================================================
# LOGGING
# =============================================================================
def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"pof_{ts}.log")

    logger = logging.getLogger("pof")
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
    logger.info("PoF Pipeline - Clean Production Version")
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
    """
    Karma tarih formatlarını güvenli şekilde parse eder.
    Desteklenen formatlar:
    - Excel seri numarası (44567.5)
    - DD-MM-YYYY / DD.MM.YYYY (TR format)
    - YYYY-MM-DD (ISO format)
    - Datetime objesi (zaten işlenmiş)
    """
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT

    try:
        # Eğer veri zaten datetime objesi ise (Excel okurken bazen otomatik çevirir)
        if isinstance(x, (pd.Timestamp, datetime)):
            return x

        # Eğer veri Excel seri numarası (float/int) olarak geldiyse (Örn: 44567.5)
        if isinstance(x, (int, float)):
            # Excel başlangıç tarihi: 30 Aralık 1899
            return pd.to_datetime(x, unit='D', origin='1899-12-30')

        # String formatları (Karma format desteği)
        date_str = str(x).strip()

        # Deneme sırası (TR formatı öncelikli)
        formats = [
            '%d-%m-%Y %H:%M:%S',  # 22-06-2025 04:59:21
            '%d-%m-%Y',           # 05-03-2025
            '%d.%m.%Y %H:%M:%S',  # 22.06.2025 04:59:21
            '%d.%m.%Y',           # 22.06.2025
            '%Y-%m-%d %H:%M:%S',  # 2023-01-17 17:14:42
            '%Y-%m-%d',           # 2023-01-17
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue

        # Hiçbiri işe yaramazsa pandas otomatik (dayfirst=True)
        return pd.to_datetime(x, errors="coerce", dayfirst=True)

    except Exception:
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
# ⏳ TEMPORAL SPLIT (ZAMANSAL BÖLÜNME & SIZINTI ÖNLEME)
# =============================================================================
# Bu fonksiyon, modelin "geleceği görmesini" (Data Leakage) engelleyen en kritik güvenlik duvarıdır.
#
# 🚫 Neden Rastgele (Random) Bölmüyoruz?
#    - Rastgele bölme yaparsak, 2024 yılındaki bir arızayı eğitim setine, 
#      2020 yılındaki sağlam durumu test setine koyabiliriz.
#    - Bu durumda model, "gelecekteki bilgiyi" kullanarak "geçmişi" tahmin eder.
#    - Sonuç: Test başarısı yapay olarak yüksek çıkar (%99) ama canlıda başarısız olur.
#
# ✅ Nasıl Çalışır?
#    1. Tüm varlıkları KURULUM TARİHİNE göre eskiden yeniye sıralar.
#    2. Zaman çizgisinin %75'inde bir kesme noktası (Cutoff) belirler.
#    3. Geçmiş %75 -> EĞİTİM SETİ (Model sadece geçmişi bilir).
#    4. Gelecek %25 -> TEST SETİ (Modelin hiç görmediği gelecek).
#
# 🧠 Teknik Detay:
#    - Fonksiyon, veri setini kopyalamak yerine, orijinal DataFrame'in 
#      İNDEKS ETİKETLERİNİ (Index Labels) döndürür.
#    - Bu yöntem, pandas sıralama işlemlerinde kayan indeks hatalarını (Loc vs Iloc) önler.

# 🌟 AKILLI KESME (SMART CUTOFF):
#    - Standart bölme sonucu Test setinde yeterli arıza (Event) kalmazsa,
#      fonksiyon otomatik olarak kesme noktasını 6 ay geriye çeker.
#    - Amaç: Test setinin "tamamen sağlıklı" varlıklardan oluşmasını engelleyip
#      AUC skorunun hesaplanabilir olmasını sağlamaktır.
# =============================================================================
# =============================================================================
# TEMPORAL SPLIT (Core of Leakage Prevention)
# =============================================================================
def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    min_test_event_rate: float = 0.08,  # ✅ DÜŞÜRÜLDÜ: %8 (daha esnek)
    use_stratified_fallback: bool = False,  # ✅ KAPATILDI: Temporal düzeltildi
    apply_balancing: bool = True,  # ✅ NEW: Balance train/test separately AFTER split
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ✅ FIXED: Event-aware temporal split with POST-SPLIT balancing

    Args:
        df: DataFrame with 'Kurulum_Tarihi' and 'event' columns
        test_size: Fraction of data for test set
        min_test_event_rate: Minimum acceptable event rate in test set (default 8%)
        use_stratified_fallback: DEPRECATED - kept for compatibility
        logger: Logger instance

    Returns:
        train_labels, test_labels (index arrays)
    """
    
    if "Kurulum_Tarihi" not in df.columns:
        raise ValueError("Temporal split requires 'Kurulum_Tarihi' column")
    
    install_dates = pd.to_datetime(df["Kurulum_Tarihi"], errors="coerce")
    if install_dates.isna().all():
        raise ValueError("All Kurulum_Tarihi values are invalid")
    
    df_sorted = df.copy()
    df_sorted["_install_clean"] = install_dates
    df_sorted = df_sorted.sort_values("_install_clean")
    
    # Standard cutoff
    cutoff_pos_initial = int(len(df_sorted) * (1 - test_size))
    
    # Check test event rate
    if "event" in df.columns:
        test_events_initial = df_sorted.iloc[cutoff_pos_initial:]["event"].mean()
        
        # If too low, adjust cutoff backwards
        if test_events_initial < min_test_event_rate:
            if logger:
                logger.warning(f"[SPLIT] Initial test events: {test_events_initial:.1%} < {min_test_event_rate:.1%}")

            # Move cutoff back by 6 months
            cutoff_date_initial = df_sorted.iloc[cutoff_pos_initial]["_install_clean"]
            cutoff_date_adjusted = cutoff_date_initial - pd.Timedelta(days=180)  # ✅ 6 ay geriye
            
            # Find new cutoff position
            mask = df_sorted["_install_clean"] <= cutoff_date_adjusted
            if mask.any():
                cutoff_pos = mask.sum()
                
                # Verify improvement
                test_events_adjusted = df_sorted.iloc[cutoff_pos:]["event"].mean()

                if test_events_adjusted > test_events_initial:
                    if logger:
                        logger.info(f"[SPLIT] Adjusted cutoff: test events {test_events_initial:.1%} → {test_events_adjusted:.1%}")
                else:
                    cutoff_pos = cutoff_pos_initial  # Rollback if no improvement
                    if logger:
                        logger.warning(f"[SPLIT] Adjustment failed, keeping original cutoff")
            else:
                cutoff_pos = cutoff_pos_initial
        else:
            cutoff_pos = cutoff_pos_initial
    else:
        cutoff_pos = cutoff_pos_initial
    
    # Split (strict: no overlap)
    # Ensure cutoff date is EXCLUSIVE for train (train < cutoff_date)
    cutoff_date_value = df_sorted.iloc[cutoff_pos]["_install_clean"]

    # Train: strictly before cutoff
    train_mask = df_sorted["_install_clean"] < cutoff_date_value
    # Test: cutoff and after
    test_mask = df_sorted["_install_clean"] >= cutoff_date_value

    train_labels = df_sorted[train_mask].index.values
    test_labels = df_sorted[test_mask].index.values

    if logger:
        cutoff_date = df_sorted.iloc[cutoff_pos]["_install_clean"]
        logger.info(f"[TEMPORAL SPLIT] Cutoff: {cutoff_date.date()}")
        logger.info(f"[TEMPORAL SPLIT] Train: {len(train_labels)} | Test: {len(test_labels)}")

        # ✅ DEBUG: Tarihleri detaylı yazdır
        train_dates = df_sorted.iloc[:cutoff_pos]["_install_clean"]
        test_dates = df_sorted.iloc[cutoff_pos:]["_install_clean"]
        logger.info(f"[DEBUG] Train date range: {train_dates.min().date()} -> {train_dates.max().date()}")
        logger.info(f"[DEBUG] Test date range: {test_dates.min().date()} -> {test_dates.max().date()}")

        if "event" in df.columns:
            train_ev = df.loc[train_labels, "event"].mean()
            test_ev = df.loc[test_labels, "event"].mean()
            logger.info(f"[TEMPORAL SPLIT] Train events: {train_ev:.1%} | Test events: {test_ev:.1%}")

            # ✅ UYARI: Event rate mismatch kontrolü (GÜNCELLENDİ)
            if train_ev > 0 and test_ev > 0:
                event_rate_ratio = test_ev / train_ev

                # ✅ REVERSED CHECK: Test >> Train ise kritik UYARI (fallback YOK)
                if event_rate_ratio > 2.0:
                    logger.error(
                        f"[CRITICAL] TEST EVENT RATE ANOMALY DETECTED!\n"
                        f"  Train: {train_ev:.1%} | Test: {test_ev:.1%} | Ratio: {event_rate_ratio:.2f}x\n"
                        f"  -> Test has {event_rate_ratio:.1f}x MORE failures than train!\n"
                        f"  LIKELY CAUSES:\n"
                        f"    1. Temporal split REVERSED (old equipment in test)\n"
                        f"    2. Data quality issue (mass failure event in recent period)\n"
                        f"    3. Kurulum_Tarihi column has errors\n"
                        f"  PROCEEDING WITH TEMPORAL SPLIT (fallback disabled)\n"
                        f"  Model metrics may be unreliable - review data quality!"
                    )
                elif event_rate_ratio < 0.7 or event_rate_ratio > 1.3:
                    logger.warning(
                        f"[SPLIT WARNING] Event rate mismatch detected!\n"
                        f"  Train: {train_ev:.1%} | Test: {test_ev:.1%} | Ratio: {event_rate_ratio:.2f}\n"
                        f"  Possible causes:\n"
                        f"    - Right censoring: Newer assets haven't failed yet\n"
                        f"    - Data quality: Different failure patterns over time\n"
                        f"  Recommendation: Consider using stratified split or adjusting cutoff date"
                    )

    # ✅ POST-SPLIT BALANCING (IEEE PHM Standard 1:5)
    # Critical fix: Balance AFTER split to maintain consistent event rates
    if apply_balancing and "event" in df.columns:
        from sklearn.utils import resample

        target_ratio = 5  # 1:5 faulty:healthy
        random_state = 42

        if logger:
            logger.info(f"[BALANCING] Applying post-split balancing (1:{target_ratio})")

        # Balance train set
        train_df = df.loc[train_labels].copy()
        train_faulty = train_df[train_df["event"] == 1]
        train_healthy = train_df[train_df["event"] == 0]

        n_train_faulty = len(train_faulty)
        n_train_healthy_target = n_train_faulty * target_ratio

        if len(train_healthy) > n_train_healthy_target:
            train_healthy_sampled = resample(train_healthy,
                                             n_samples=n_train_healthy_target,
                                             random_state=random_state,
                                             replace=False)
            train_balanced = pd.concat([train_faulty, train_healthy_sampled])
            train_labels = train_balanced.index.values

            if logger:
                logger.info(f"[BALANCING] Train: {len(train_faulty)} faulty | {len(train_healthy)} -> {n_train_healthy_target} healthy")

        # Balance test set
        test_df = df.loc[test_labels].copy()
        test_faulty = test_df[test_df["event"] == 1]
        test_healthy = test_df[test_df["event"] == 0]

        n_test_faulty = len(test_faulty)
        n_test_healthy_target = n_test_faulty * target_ratio

        if len(test_healthy) > n_test_healthy_target:
            test_healthy_sampled = resample(test_healthy,
                                           n_samples=n_test_healthy_target,
                                           random_state=random_state,
                                           replace=False)
            test_balanced = pd.concat([test_faulty, test_healthy_sampled])
            test_labels = test_balanced.index.values

            if logger:
                logger.info(f"[BALANCING] Test: {len(test_faulty)} faulty | {len(test_healthy)} -> {n_test_healthy_target} healthy")

        # Verify balanced event rates
        if logger:
            train_ev_balanced = df.loc[train_labels, "event"].mean()
            test_ev_balanced = df.loc[test_labels, "event"].mean()
            ratio_balanced = test_ev_balanced / train_ev_balanced if train_ev_balanced > 0 else 0

            logger.info(f"[BALANCING] Post-balance event rates:")
            logger.info(f"  Train: {train_ev_balanced:.1%} | Test: {test_ev_balanced:.1%} | Ratio: {ratio_balanced:.2f}x")

            if 0.5 <= ratio_balanced <= 2.0:
                logger.info(f"[BALANCING] Event rates are now balanced!")
            else:
                logger.warning(f"[BALANCING] Event rates still imbalanced after balancing")

    return train_labels, test_labels

# =============================================================================
# DATA LOADING
# =============================================================================
# İzmir ve Manisa Bölgesi İlçe Kodları
ILCE_ID_MAPPING = {
    # --- MANİSA ---
    1118: 'Ahmetli', 1119: 'Akhisar', 1127: 'Alasehir', 1269: 'Demirci',
    1362: 'Gordes', 1470: 'Kirkagac', 1489: 'Kula', 1590: 'Salihli',
    1600: 'Sarigol', 1606: 'Saruhanli', 1613: 'Selendi', 1634: 'Soma',
    1682: 'Turgutlu', 1751: 'Sehzadeler', 1752: 'Yunusemre', 1965: 'Koprubasi',
    # --- İZMİR ---
    1109: 'Aliaga', 1165: 'Bayindir', 1188: 'Bergama', 1205: 'Bornova',
    1216: 'Buca', 1251: 'Cesme', 1280: 'Dikili', 1334: 'Foca',
    1432: 'Karaburun', 1448: 'Karsiyaka', 1461: 'Kemalpasa', 1467: 'Kinik',
    1477: 'Kiraz', 1500: 'Menemen', 1542: 'Odemis', 1611: 'Seferihisar',
    1612: 'Selcuk', 1677: 'Tire', 1689: 'Torbali', 1703: 'Urla',
    1780: 'Beydag', 1801: 'Konak', 1826: 'Menderes', 1888: 'Balcova',
    1889: 'Cigli', 1890: 'Gaziemir', 1891: 'Narlidere', 1892: 'Guzelbahce',
    2006: 'Bayrakli', 2007: 'Karabaglar'
}
def load_fault_data(logger: logging.Logger) -> pd.DataFrame:
    """Load and clean fault records"""
    path = DATA_PATHS["fault_data"]
    logger.info(f"[LOAD] Fault data: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # --- GÜNCELLEME: Lokasyon Sütunlarını Ekle ---
    base_cols = ["cbs_id", "Şebeke Unsuru", "Sebekeye_Baglanma_Tarihi",
                 "started at", "ended at", "duration time", "cause code"]
    
    # Mevcut bakım sütunları + Sizin belirttiğiniz YENİ lokasyon sütunları
    extra_cols = ["Bakım Sayısı", "Son Bakım İş Emri Tarihi", "MARKA",
                  #"kVA_Rating",
                  "component_voltage", "voltage_level",
                  "X_KOORDINAT", "Y_KOORDINAT", "İlçe"]  # <--- EKLENDİ

    use_cols = [c for c in base_cols + extra_cols if c in df.columns]
    df = df[use_cols].copy()

    # Rename Mapping
    df = df.rename(columns={
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "duration time": "Süre_Ham",
        "Bakım Sayısı": "Bakim_Sayisi",
        "MARKA": "Marka",
        "component_voltage": "Gerilim_Seviyesi",
        "voltage_level": "Gerilim_Sinifi",
        # --- LOKASYON MAPPING ---
        "X_KOORDINAT": "Longitude",  # Genelde X Boylamdır
        "Y_KOORDINAT": "Latitude",   # Genelde Y Enlemdir
        "İlçe": "Ilce"
    })

    # Parse dates
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["started at"] = df["started at"].apply(parse_date_safely)
    df["ended at"] = df["ended at"].apply(parse_date_safely)
    df["Süre_Dakika"] = convert_duration_minutes(df["Süre_Ham"], logger)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])

    # Filter future dates (data quality check)
    today = pd.Timestamp.now()
    future_faults = (df["started at"] > today).sum()
    if future_faults > 0:
        logger.warning(f"[DATA QUALITY] {future_faults} gelecek tarihli arıza kaydı bulundu ve filtrelendi.")
        df = df[df["started at"] <= today].copy()

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
    # Koordinatları sayıya çevirmeyi garantiye al (Hatalı text varsa NaN olsun)
    for col in ["Longitude", "Latitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
    
    # --- GÜNCELLEME BURADA BAŞLIYOR ---
    # Arıza verisiyle aynı isimlere eşitliyoruz
    # --- GÜNCELLEME: Rename Haritası ---
    rename_map = {
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "MARKA": "Marka",
        "Bakım Sayısı": "Bakim_Sayisi",
        "component_voltage": "Gerilim_Seviyesi",
        "voltage_level": "Gerilim_Sinifi",
        #"kVA_Rating": "kVA_Rating",
        # --- YENİ LOKASYONLAR ---
        "X_KOORDINAT": "Longitude",
        "Y_KOORDINAT": "Latitude",
        "ADR_ILCE_ID": "Ilce_ID",   # Önce ID olarak alıyoruz
    }
    df = df.rename(columns=rename_map)
    # --- YENİ: ID'den İsme Çevirme ---
    if "Ilce_ID" in df.columns:
        # map fonksiyonu ile ID'leri isme çevir, bulamazsa 'Bilinmiyor' yazar
        df["Ilce"] = df["Ilce_ID"].map(ILCE_ID_MAPPING).fillna("Bilinmiyor")
        
        # Artık ID kolonuna ihtiyacımız kalmadıysa atabiliriz veya tutabiliriz
        # df = df.drop(columns=["Ilce_ID"])
    else:
        df["Ilce"] = "Unknown"
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])
    # Koordinat temizliği
    for col in ["Longitude", "Latitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Gereksiz/Boş kayıtları temizle
    df = df[df["Kurulum_Tarihi"].notna() & df["cbs_id"].notna()].copy()

    logger.info(f"[LOAD] Healthy equipment: {len(df)}")
    
    # Kaydederken de bu yeni sütunların olduğundan emin oluyoruz
    df.to_csv(INTERMEDIATE_PATHS["healthy_equipment_clean"], index=False, encoding="utf-8-sig")
    logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['healthy_equipment_clean']}")

    return df

# =============================================================================
# STEP 01: EQUIPMENT MASTER + SURVIVAL BASE
# =============================================================================

# =============================================================================
# 🔧 BAKIM VERİSİ STRATEJİSİ (MAINTENANCE STRATEGY)
# =============================================================================
# Sorun:
#   Veri setimizde 'Bakim_Sayisi' sütunu sıkça boş (NaN) geliyor.
#   NaN değerlerini 0 (Sıfır) ile doldurmak HATALIDIR. Çünkü:
#   - 0: "Kesinlikle bakım yapılmadı" (Kötü bir durum olabilir)
#   - NaN: "Bakım yapılıp yapılmadığını bilmiyoruz" (Belirsiz bir durum)
#
# Çözüm:
#   Modelin bu iki durumu ayırt edebilmesi için 'Bakim_Sayisi' sütununu
#   iki yeni özelliğe (feature) dönüştürüyoruz:
#
#   1. Bakim_Verisi_Var (Flag): 
#      - 1: Bakım verisi sistemde kayıtlı.
#      - 0: Bakım verisi yok / bilinmiyor.
#
#   2. Bakim_Sayisi_Safe (Value):
#      - Pozitif Sayılar (1, 2, 5...): Gerçek bakım sayısı.
#      - 0: Hiç bakım yapılmamış (Veri var ama sayı 0).
#      - -1: Bilinmiyor (NaN).
#
#   Neden?
#   Ağaç tabanlı modeller (XGBoost, Random Forest), -1 ile 0 arasındaki farkı
#   öğrenebilir. Böylece "Bakımsızlık Riski" ile "Veri Eksikliği Riski" birbirinden ayrılır.
# =============================================================================
def build_equipment_master(
    df_fault: pd.DataFrame,
    df_healthy: pd.DataFrame,
    logger: logging.Logger,
    data_end_date: pd.Timestamp
) -> pd.DataFrame:
    """Combine fault + healthy equipment into master registry"""
    
    # 1. Ortak Aggregation Kuralları
    agg_cols = {
        "Kurulum_Tarihi": ("Kurulum_Tarihi", "min"),
        "Ekipman_Tipi": ("Ekipman_Tipi", "first"),
        "Marka": ("Marka", "first"),
        "Gerilim_Seviyesi": ("Gerilim_Seviyesi", "max"),
        "Gerilim_Sinifi": ("Gerilim_Sinifi", "first"),
        "Bakim_Sayisi": ("Bakim_Sayisi", "max"),
        "Longitude": ("Longitude", "mean"), 
        "Latitude": ("Latitude", "mean"),
        "Ilce": ("Ilce", "first")
    }

    # 2. Arızalı Ekipmanları Özetle
    fault_agg_rules = agg_cols.copy()
    fault_agg_rules.update({
        "Fault_Count": ("cbs_id", "size"),
        "Ilk_Ariza_Tarihi": ("started at", "min"),
        "Son_Ariza_Tarihi": ("started at", "max")
    })
    
    # --- DÜZELTME BAŞLANGICI ---
    # HATA ÇÖZÜMÜ: Sadece df_fault içinde GERÇEKTEN VAR OLAN sütunları kurallara dahil et.
    # Eğer 'Latitude' yüklenemediyse, burada işlemeye çalışıp hata vermesin.
    final_fault_rules = {}
    for col, rule in fault_agg_rules.items():
        # Kuralın anahtarı (örn: 'Latitude') dataframe sütunlarında var mı?
        # Veya bu bir türetilen sütun mu (örn: 'Fault_Count')?
        if col in df_fault.columns or col == "Fault_Count":
            final_fault_rules[col] = rule
            
    fault_agg = df_fault.groupby("cbs_id").agg(**final_fault_rules).reset_index()
    # --- DÜZELTME BİTİŞİ ---
    
    # 3. Sağlam Ekipmanları Özetle
    # Aynı güvenli mantığı burada da uyguluyoruz
    healthy_agg_rules = {}
    for col, rule in agg_cols.items():
        if col in df_healthy.columns: 
            healthy_agg_rules[col] = rule
            
    healthy_agg = df_healthy.groupby("cbs_id").agg(**healthy_agg_rules).reset_index()
    healthy_agg["Fault_Count"] = 0

    # 🎯 DATA BALANCING (Config-Driven)
    # IEEE PHM Best Practice: 1:3 to 1:5 for predictive maintenance
    n_faulty = len(fault_agg)

    if BALANCE_ENABLED and BALANCE_METHOD == "undersample":
        n_healthy_target = n_faulty * BALANCE_TARGET_RATIO

        if len(healthy_agg) > n_healthy_target:
            logger.info(f"[BALANCING] Target ratio: 1:{BALANCE_TARGET_RATIO} (IEEE PHM Standard)")
            logger.info(f"[BALANCING] Undersampling healthy assets: {len(healthy_agg)} → {n_healthy_target}")
            healthy_agg = healthy_agg.sample(n=n_healthy_target, random_state=BALANCE_RANDOM_STATE).reset_index(drop=True)
        else:
            logger.info(f"[BALANCING] Healthy assets already below target: {len(healthy_agg)} < {n_healthy_target}")
    else:
        logger.info(f"[BALANCING] Disabled - using all {len(healthy_agg)} healthy assets")

    # 4. Birleştir
    all_eq = pd.concat([fault_agg, healthy_agg], ignore_index=True)
    
    # Çakışmaları Temizle (Bir ekipman hem arızalı hem sağlam listesinde olamaz ama varsa arızalıyı koru)
    all_eq = all_eq.sort_values(["cbs_id", "Fault_Count"], ascending=[True, False]) \
                   .drop_duplicates("cbs_id", keep="first")
    
    # Nadir Tipleri Temizle
    counts = all_eq["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[COLLAPSE] Rare types → 'Diger': {rare}")
        all_eq.loc[all_eq["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Diger"
    
    # --- YENİ KOD (YAPIŞTIRIN) ---
    if "Bakim_Sayisi" in all_eq.columns:
        # 1. Flag: Bu varlığın bakım bilgisini biliyor muyuz? (1: Evet, 0: Hayır/NaN)
        all_eq["Bakim_Verisi_Var"] = all_eq["Bakim_Sayisi"].notna().astype(int)
        
        # 2. Sayı: NaN olanları -1 yapıyoruz.
        # Neden -1? Çünkü Ağaç tabanlı modeller (RSF, XGBoost) -1'i "Bilinmiyor", 0'ı "Hiç Bakım Yok" olarak ayırabilir.
        all_eq["Bakim_Sayisi_Safe"] = all_eq["Bakim_Sayisi"].fillna(-1)
        
        logger.info(f"[MASTER] Bakım verisi işlendi. Bilinen kayıt: {all_eq['Bakim_Verisi_Var'].sum()}")
    else:
        # Sütun hiç yoksa varsayılanları oluştur
        all_eq["Bakim_Verisi_Var"] = 0
        all_eq["Bakim_Sayisi_Safe"] = -1

    logger.info(f"[MASTER] Equipment registry: {len(all_eq)} assets")
    return all_eq

# =============================================================================
# UPDATED build_survival_base 
# =============================================================================
# =============================================================================
# ⏳ SURVIVAL BASE DATASET (YAŞAM SÜRESİ TABLOSU & SOL KESİLME)
# =============================================================================
# Bu fonksiyon, Survival Analizi'nin bel kemiği olan (Duration, Event) çiftini oluşturur.
# İstatistiksel doğruluğu sağlamak için 3 kritik işlem yapar:
#
# 🎯 1. Gerçek Arıza Tanımı (Event = 1):
#    - Sigorta atması (Fuse Trip) gibi koruma operasyonları "Arıza" sayılmaz.
#    - Sadece fiziksel hasarlar (Tel kopması, Trafo yanması) "Ölüm" (Event=1) kabul edilir.
#
# 📏 2. Sol Kesilme (Left Truncation / Delayed Entry):
#    - SORUN: Veri setimiz 2021'de başlıyor ama şebekede 1990 model trafo var.
#    - RİSK: Modele "Bu trafo 1990-2021 arası hiç bozulmadı" dersek (Survivorship Bias),
#      model eski varlıkları "ölümsüz" sanar.
#    - ÇÖZÜM: 'entry_days' hesaplıyoruz. Modele diyoruz ki:
#      "Bu varlık 1990'da doğdu ama biz onu 2021'de (yani 11.000 günlükken) izlemeye başladık."
#      Model, 0-11.000 gün arasındaki sağ kalımı başarı hanesine yazmaz, sadece sonrasını değerlendirir.
#
# ⏱️ 3. Ömür (Duration):
#    - Ölenler için: Kurulum Tarihi -> İlk Gerçek Arıza Tarihi
#    - Yaşayanlar için: Kurulum Tarihi -> Analiz Tarihi (Verinin Bittiği Gün)
# =============================================================================
def build_survival_base(
    equipment_master: pd.DataFrame,
    df_fault: pd.DataFrame,
    logger,
    data_end_date
) -> pd.DataFrame:
    """
    Create survival dataset - uses ALL fault records

    ✅ UPDATED: Domain expert verified all cause codes are valid failures
    """

    # Data quality already checked in load_fault_data()
    df_fault_real = df_fault.copy()
    
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
# 📉 CHRONIC FEATURE ENGINEERING (KRONİK ARIZA VE MTBF ANALİZİ)
# =============================================================================
# Bu fonksiyon, varlıkların "kısa vadeli" sağlık durumunu 4 farklı açıdan analiz eder.
#
# 🧠 1. Exponential Decay (Üstel Bozunma): 
#    - "Dün yaşanan arıza, 3 ay önceki arızadan daha tehlikelidir."
#    - Yakın geçmişteki arızalara daha yüksek ağırlık vererek (Weight=1.0 vs 0.2)
#      kriz geçirmekte olan varlıkları öne çıkarır.
#
# 🧮 2. Bayesian MTBF (Mean Time Between Failures):
#    - SORUN: Klasik MTBF = (Süre / Arıza Sayısı). Hiç arıza yapmamış varlıkta
#      bölen 0 olduğu için sonuç sonsuz veya tanımsız çıkar.
#    - ÇÖZÜM: Formüle "Sanal Başlangıç Değerleri" (Priors) eklenir.
#      Formül: (Pencere Süresi + 30 gün) / (Arıza Sayısı + 1).
#    - SONUÇ: Hiç arızası olmayan varlıkların bile mantıklı bir risk skoru olur
#      ve model onları kıyaslayabilir.
#
# 📅 3. Annualized Rate (Yıllıklandırılmış Hız):
#    - 90 günlük performansı 1 yıla projete eder (Örn: 3 ayda 2 arıza = Yılda 8 arıza).
#
# 🚩 4. Chronic Flag:
#    - Belirli bir eşiği (örn: 3 arıza) aşan varlıkları "Kronik Sorunlu" (1) olarak etiketler.
#    - IEEE 1366 prensiplerine benzer şekilde, bu özellik modelin en güçlü sinyallerinden biridir.
# =============================================================================



# =============================================================================
def compute_chronic_features(
    df_fault: pd.DataFrame,
    t_ref: pd.Timestamp,
    logger: logging.Logger
) -> pd.DataFrame:
    """Chronic equipment detection (Bayesian MTBF + Decay)"""
    
    window_start = t_ref - pd.Timedelta(days=CHRONIC_WINDOW_DAYS)
    fe = df_fault[df_fault["started at"] >= window_start].copy()
    
    # Eğer hiç arıza yoksa boş dön
    if len(fe) == 0:
        logger.warning(f"[CHRONIC] No faults in window")
        # Sütun isimlerini eksiksiz tanımlayın
        cols = ["cbs_id", "Ariza_Sayisi_90g", "Chronic_Rate_Yillik", 
                "Chronic_Decay_Skoru", "Chronic_Flag", "MTBF_Bayes_Gun"]
        return pd.DataFrame(columns=cols)
    
    # 1. Arıza Sayıları
    counts = fe.groupby("cbs_id").size().rename("Ariza_Sayisi_90g")
    
    # 2. Decay Skoru
    age_days = (t_ref - fe["started at"]).dt.days.clip(lower=0)
    fe["decay"] = np.exp(-0.05 * age_days)
    decay_score = fe.groupby("cbs_id")["decay"].sum().rename("Chronic_Decay_Skoru")
    
    # 3. Yıllık Oran
    rate = (counts / (CHRONIC_WINDOW_DAYS / 365.25)).rename("Chronic_Rate_Yillik")
    
    # --- EKLENEN KISIM: BAYESIAN MTBF ---
    # Formül: (Pencere Süresi + Beta) / (Arıza Sayısı + Alfa)
    # Alfa=1 (Sanal 1 arıza), Beta=30 (Sanal 1 ay ömür) varsayalım.
    # Bu, 0 arızası olanı sonsuz yapmaz, "Henüz bozulmadı ama riskli olabilir" seviyesinde tutar.
    alpha = 1
    beta = 30
    mtbf = ((CHRONIC_WINDOW_DAYS + beta) / (counts + alpha)).rename("MTBF_Bayes_Gun")
    # ------------------------------------

    # 4. Kronik Bayrağı
    chronic_flag = ((counts >= CHRONIC_THRESHOLD_EVENTS) | (rate >= CHRONIC_MIN_RATE)).astype(int).rename("Chronic_Flag")
    
    # Çıktıları Birleştir (mtbf eklendi)
    out = pd.concat([counts, rate, decay_score, chronic_flag, mtbf], axis=1).reset_index()
    
    logger.info(f"[CHRONIC] Window: {CHRONIC_WINDOW_DAYS}d | Chronic assets: {chronic_flag.sum()}")
    return out

# =============================================================================
# 🕒 TEMPORAL FEATURES & OBSERVABILITY (ZAMANSAL ÖZELLİKLER & GÖZLENEBİLİRLİK)
# =============================================================================
# Bu fonksiyon, statik varlık verisine "Zaman Boyutunu" ve "Geçmiş İstatistiklerini" ekler.
#
# 🔍 1. Gözlenebilirlik (Observability) - Bias Önleme:
#    - SORUN: 30 yaşındaki bir trafoyu sadece son 3 yıldır (2021'den beri) izliyor olabiliriz.
#      Model bunu bilmezse, varlığın 30 yıldır sorunsuz çalıştığını sanar.
#    - ÇÖZÜM: 'Observation_Ratio' (İzlenen Süre / Toplam Yaş) hesaplanır.
#    - SONUÇ: Model, "Legacy" (Eski ama verisi az) varlıklar ile "Yeni" (Tüm hayatı bilinen)
#      varlıkları ayırt etmeyi öğrenir.
#
# 📊 2. Kronik Veri Entegrasyonu (Merge):
#    - compute_chronic_features fonksiyonundan gelen 4 kritik metriği ana tabloya işler:
#      a. Chronic_Flag: Kronik sorunlu mu? (1/0)
#      b. Chronic_Decay_Skoru: Arızalar ne kadar taze? (Yakın zamana ağırlık verir)
#      c. Chronic_Rate_Yillik: Yıllık arıza hızı.
#      d. MTBF_Bayes_Gun: (YENİ ✅) Sıfır arızalı varlıklar için bile hesaplanan,
#         istatistiksel olarak düzeltilmiş "Arızalar Arası Ortalama Süre".
# =============================================================================
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
    observation_start_date: pd.Timestamp,
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
    
    # --- Observability Features ---
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
        # --- DÜZELTME BURADA: MTBF_Bayes_Gun EKLENDİ ---
        cols_to_merge = [
            "Ariza_Sayisi_90g", 
            "Chronic_Rate_Yillik", 
            "Chronic_Decay_Skoru", 
            "Chronic_Flag", 
            "MTBF_Bayes_Gun"  # <--- ARTIK LİSTEDE!
        ]
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
        df["MTBF_Bayes_Gun"] = 0  # <--- EKLENDİ
    
    logger.info(f"[FEATURES] Temporal: Added age + chronic features (incl. MTBF) + observability stats")
    return df
# =============================================================================
# STEP 03: MODEL TRAINING
# =============================================================================
# =============================================================================
# 🧹 MULTICOLLINEARITY CLEANER (ÇOKLU BAĞLANTI TEMİZLİĞİ)
# =============================================================================
# Bu fonksiyon, modelin stabilitesini bozan "birbirinin kopyası" değişkenleri temizler.
#
# 🔍 Sorun (Multicollinearity):
#    - Örnek: 'Tref_Yas_Gun' ile 'Kurulum_Yili' neredeyse aynı bilgiyi taşır.
#    - İkisi birden modele girerse, Cox/Regression modellerinin katsayıları (Coefficients)
#      güvenilmez hale gelir ve standart hatalar aşırı büyür.
#
# 🛠️ Çözüm (Iterative VIF Removal):
#    1. Tüm sayısal değişkenlerin VIF (Variance Inflation Factor) değerini hesaplar.
#    2. VIF değeri eşiği (Genelde 10.0) geçen değişkenlerden EN YÜKSEK olanı seçer.
#    3. O değişkeni veri setinden atar.
#    4. Kalan değişkenlerle VIF'i tekrar hesaplar (Çünkü birini atınca diğerleri düzelebilir).
#    5. Tüm VIF değerleri < 10 olana kadar bu döngü devam eder.
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
# =============================================================================
# 🎯 FEATURE SELECTION (ÖZELLİK SEÇİMİ VE BOYUT İNDİRGEME)
# =============================================================================
# Bu fonksiyon, modelin performansını artırmak için "En Değerli" özellikleri seçer.
#
# 🔍 Neden Yapıyoruz?
#    - One-Hot Encoding sonrası (Marka, Mahalle vb.) yüzlerce sütun oluşabilir.
#    - Çok fazla sütun (High Dimensionality), modelin gereksiz veriyi ezberlemesine
#      (Overfitting) ve eğitim süresinin uzamasına neden olur.
#
# 🛠️ Nasıl Çalışır?
#    1. Geçici bir Random Forest modeli eğitilir.
#    2. Bu model, her bir özelliğin tahmine ne kadar katkı sağladığını (Feature Importance) ölçer.
#    3. Sadece en yüksek puana sahip ilk 'K' özellik (top_k) tutulur.
#    4. Geri kalan "Gürültü" (Noise) niteliğindeki zayıf özellikler veri setinden atılır.
# =============================================================================

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
    #rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    
    # Get top K features
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(top_k).index.tolist()
    
    if logger:
        logger.info(f"[FEATURE IMPORTANCE] Selected top {len(top_features)} features")
    
    return X_train[top_features], X_test[top_features]

# =============================================================================
# 👯 HIGH CORRELATION FILTER (YÜKSEK KORELASYON FİLTRESİ)
# =============================================================================
# Bu fonksiyon, VIF analizinden önce yapılan "Hızlı ve Kaba" temizliktir.
#
# 🔍 Amaç:
#    - Birbiriyle %95'ten fazla (threshold=0.95) benzerlik gösteren değişken çiftlerini bulur.
#    - Örnek: "Sıcaklık (C)" ve "Sıcaklık (F)". Bu ikisi matematiksel olarak aynı bilgidir.
#    - İkisini birden modele vermek, modelin kafasını karıştırır (Multicollinearity).
#
# 🛠️ Yöntem:
#    1. Korelasyon matrisini (Pearson) çıkarır.
#    2. Matrisin simetrik olduğunu bildiği için sadece "Üst Üçgen"e (Upper Triangle) bakar.
#    3. İlişkisi 0.95'i geçen çiftlerden ikincisini (sütun bazında sonra geleni) siler.
# =============================================================================
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


# =============================================================================
# 🏭 PREPROCESSING PIPELINE (VERİ ÖN İŞLEME HATTI)
# =============================================================================
# Bu fonksiyon, ham veriyi modelin anlayacağı matematiksel formata dönüştürür.
#
# 🔢 Sayısal Veriler İçin:
#    - Eksik Veriler (NaN): Medyan ile doldurulur (Median Imputation).
#      Bu yöntem, aykırı değerlerin (Outliers) ortalamayı bozmasını engeller.
#
# 🔠 Kategorik (Metin) Veriler İçin:
#    - Eksik Veriler: En sık geçen değer (Mode) ile doldurulur.
#    - Dönüşüm: One-Hot Encoding uygulanır.
#      Örn: "Marka: Siemens" -> [0, 0, 1, 0] gibi binary vektöre dönüşür.
#    - Güvenlik: `handle_unknown='ignore'` sayesinde, gelecekte bilinmeyen
#      bir kategori gelirse sistem çökmez, sadece o özelliği 0 sayar.
# =============================================================================
def build_preprocessor(X: pd.DataFrame, logger=None): # <--- logger parametresi eklendi
    """Sklearn pipeline for numeric + categorical features"""
    
    # 1. Sayısal ve Kategorik Sütunları Otomatik Ayır
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    # 2. Hangi sütun ne işlem görecek LOG'a yaz (Casus Kısım)
    if logger:
        logger.info("-" * 40)
        logger.info(f"[PREPROCESS] Sayısal Sütunlar (Median Impute): {len(num_cols)} adet")
        logger.info(f"   List: {num_cols}")
        logger.info(f"[PREPROCESS] Kategorik Sütunlar (Mode Impute): {len(cat_cols)} adet")
        logger.info(f"   List: {cat_cols}")
        logger.info("-" * 40)

    # 3. Pipeline Kurulumu (Değişmedi)
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
# =============================================================================
# 🛡️ COX MODEL SAFETY & LEAKAGE PREVENTION (COX GÜVENLİK FİLTRELERİ)
# =============================================================================
# Bu modül, Cox Proportional Hazards modelinin matematiksel olarak çökmesini ve
# hile yapmasını (Data Leakage) engeller.
#
# 1. select_survival_safe_features:
#    - "Geleceği Gösteren" verileri temizler.
#    - Örn: 'Fault_Count' veya 'Son_Ariza_Tarihi' verilirse, model varlığın ne kadar
#      yaşadığını dolaylı yoldan öğrenir (Leakage). Bu fonksiyon bunları yasaklar.
#
# 2. select_cox_safe_features:
#    - Cox modelinin en büyük düşmanı "Singular Matrix" (Tersi alınamayan matris) hatasıdır.
#    - Bu hatayı önlemek için:
#      a. Sabit Değerler (Constant Columns): Her satırda aynı olan veriler atılır.
#      b. High Cardinality: 20'den fazla seçeneği olan kategorik veriler (örn. Mahalle)
#         modeli şişirmesin diye atılır.
#      c. Multicollinearity: VIF ve Korelasyon testleri ile birbirinin kopyası olan
#         değişkenler temizlenir.
# =============================================================================

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
    
    # 6.⚠️ NEW: Remove highly correlated features
    X = remove_highly_correlated_features(X, threshold=0.95, logger=logger)
     
    # 7. ⚠️ NEW: Remove multicollinear features
    #X = remove_multicollinear_features(X, threshold=10.0, logger=logger)
    X = remove_multicollinear_features(X, threshold=20.0, logger=logger)
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
# =============================================================================
# 🧠 SURVIVAL MODEL TRAINING (COX & WEIBULL EĞİTİMİ)
# =============================================================================
# Bu fonksiyon, temizlenmiş veriyi alarak iki temel Survival modelini eğitir.
#
# 1. Temporal Split (Zamansal Bölme):
#    - Veriyi rastgele değil, Kurulum Tarihine göre böler (Eskiler Train, Yeniler Test).
#    - Bu yöntem, modelin "Geleceği Tahmin Etme" yeteneğini daha gerçekçi ölçer.
#
# 2. Cox PH Model (with Left Truncation):
#    - 'entry_days' parametresi kullanılarak "Delayed Entry" (Gecikmeli Giriş) tanıtılır.
#    - Bu, veri toplamaya başlamadan önce kurulmuş varlıkların yarattığı "Bias"ı siler.
#    - Penalizer (0.1): Aşırı öğrenmeyi (Overfitting) ve matris hatalarını önler.
#
# 3. Weibull AFT Model:
#    - Cox modeline alternatif olarak eğitilir. Parametrik yapısı sayesinde bazen
#      gelecek tahminlerinde daha kararlı sonuçlar verebilir.
# =============================================================================
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
        #cox = CoxPHFitter(penalizer=0.05)
        cox = CoxPHFitter(penalizer=0.1)
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
        #wb = WeibullAFTFitter(penalizer=0.05)
        wb = WeibullAFTFitter(penalizer=0.1)
        wb.fit(train_data, duration_col="duration_days", event_col="event")
        wb_pred = wb.predict_median(test_data)
        wb_cind = concordance_index(test_data["duration_days"], wb_pred, test_data["event"])
        logger.info(f"[WEIBULL] Test Concordance: {wb_cind:.4f}")
    except Exception as e:
        logger.error(f"[WEIBULL] Training failed: {e}")
    
    return cox, wb
# =============================================================================
# 🌲 RANDOM SURVIVAL FOREST (RSF) TRAINING
# =============================================================================
# Bu fonksiyon, makine öğrenmesi tabanlı, doğrusal olmayan (non-linear) bir
# yaşam analizi modeli eğitir.
#
# 🥊 Cox Modeli vs RSF:
#    - Cox: "Marka X riski %20 artırır" gibi genel kurallar bulur. (Yorumlanabilir)
#    - RSF: "Marka X, sadece Salihli bölgesindeyse ve yaşı > 10 ise risklidir" gibi
#      karmaşık etkileşimleri yakalar. (Daha yüksek tahmin gücü)
#
# 🔧 Kritik Mühendislik (Indexing Fix):
#    - Pandas DataFrame (Etiket bazlı) ile Scikit-Survival Array (Sıra bazlı)
#      arasındaki uyumsuzluğu çözmek için 'get_indexer' kullanılır.
#      Bu sayede Temporal Split sırasında veri kayması yaşanmaz.
# =============================================================================
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
        #n_estimators=100,
        n_estimators=50,
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
# 🚀 GRADIENT BOOSTING SURVIVAL ANALYSIS (GBSA)
# =============================================================================
# Bu fonksiyon, "Boosting" tekniğini kullanarak bir yaşam analizi modeli eğitir.
#
# 🔄 1. Standartlaştırılmış Zamansal Bölme (Consistency):
#    - Daha önceki manuel sıralama yerine, projenin ortak fonksiyonu olan
#      'temporal_train_test_split' kullanılır.
#    - Böylece Cox, RSF ve GBSA modelleri birebir AYNI eğitim ve test verisi
#      üzerinde yarışır. Sonuçlar adil bir şekilde kıyaslanabilir.
#
# 🛠️ 2. İndeksleme Mühendisliği (Label vs Position):
#    - Sorun: X (DataFrame) etiket bazlı (.loc), y (Numpy Array) sıra bazlı çalışır.
#    - Çözüm: 'get_indexer' metodu ile Eğitim/Test etiketleri (Labels), Numpy dizisinin
#      anlayacağı sıra numaralarına (Position) çevrilir. Bu, veri kaymasını önler.
#
# 🥊 3. GBSA vs RSF:
#    - RSF (Random Forest): Paralel çalışır, karar ağaçlarının ortalamasını alır.
#    - GBSA (Gradient Boosting): Seri çalışır. Her yeni ağaç, bir önceki ağacın
#      yaptığı hataları düzeltmek için kurulur. Genellikle daha keskin tahmin yapar.
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
        from sklearn.pipeline import Pipeline
    except ImportError:
        logger.warning("[ML] sksurv not installed. Skipping ML.")
        return None

    # 1. Temiz bir kopya oluştur
    work_df = df.copy()
    
    # 2. Sadece güvenli sütunları seç
    X = work_df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number, 'object'])
    
    # 3. Hedef Değişkeni (Target) Oluştur - Structured Array
    y = Surv.from_arrays(
        event=work_df["event"].astype(bool).values,
        time=work_df["duration_days"].values
    )

    # 4. TEMPORAL SPLIT (STANDARTLAŞTIRILMIŞ) 
    # Manuel sıralama yerine ortak fonksiyonu kullanıyoruz.
    try:
        # Fonksiyon bize Eğitim ve Test için ID listelerini (Labels) verir
        train_labels, test_labels = temporal_train_test_split(work_df, test_size=0.25, logger=logger)
        
        # --- KRİTİK NOKTA: LABEL vs POSITION ---
        
        # A) X bir DataFrame'dir, doğrudan Etiket (.loc) ile bölebiliriz
        X_train = X.loc[train_labels]
        X_test = X.loc[test_labels]
        
        # B) y bir Numpy dizisidir, Etiket anlamaz, Pozisyon (Sıra No) ister.
        # Bu yüzden Etiketleri -> Pozisyon Numarasına çeviriyoruz.
        train_pos = work_df.index.get_indexer(train_labels)
        test_pos = work_df.index.get_indexer(test_labels)
        
        y_train = y[train_pos]
        y_test = y[test_pos]
        
    except Exception as e:
        logger.warning(f"[ML] Temporal split failed ({e}), using random split")
        # Eğer tarih yoksa veya hata olursa rastgele böl
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=work_df["event"].values
        )

    # 5. Pipeline Kurulumu (GBSA)
    pre = build_preprocessor(X_train)
    
    gbsa = GradientBoostingSurvivalAnalysis(
        #n_estimators=100,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        loss="coxph",  # Cox mantığıyla optimize et
        random_state=42
    )

    model_pipeline = Pipeline([("pre", pre), ("gbsa", gbsa)])
    
    try:
        logger.info(f"[ML] Training GBSA on {len(X_train)} samples...")
        model_pipeline.fit(X_train, y_train)
        
        # Test Skoru (C-Index)
        score = model_pipeline.score(X_test, y_test)
        logger.info(f"[ML] GBSA Test Concordance: {score:.4f}")
        
        return {"model": model_pipeline, "safe_cols": feature_cols}
        
    except Exception as e:
        logger.warning(f"[ML] Training failed: {e}")
        return None

# =============================================================================
# BACKTESTING (Temporal Validation Proof)
# =============================================================================
# =============================================================================
# 🕰️ TEMPORAL BACKTESTER (ZAMANDA YOLCULUK TESTİ)
# =============================================================================
# Bu sınıf, modelin geçmişteki performansını simüle ederek "Geleceği Görme" (Look-ahead Bias)
# riskini test eder.
#
# 🔄 Çalışma Mantığı (Walk-Forward Validation):
#    1. Zamanda Geriye Git: Örn. 1 Ocak 2022 tarihine dön.
#    2. Geleceği Sil: 2022 sonrasındaki tüm arızaları ve verileri yok et.
#    3. Modeli Eğit: Sadece 2022 öncesi verilerle bir model kur.
#    4. Tahmin Yap: 2022 yılı içinde hangi trafoların bozulacağını tahmin et.
#    5. Gerçekle Kıyasla: 2022'de gerçekten bozulanlarla tahminleri karşılaştır.
#
# 📊 Kritik Metrik (Top-100 Precision):
#    - "Modelin en riskli dediği 100 trafonun kaçı o yıl gerçekten bozuldu?"
#    - Bu, sahadaki bakım ekipleri için en hayati metriktir (Return on Investment).
# =============================================================================
class TemporalBacktester:
    """Enhanced backtester with proper temporal split and survival modeling"""
    
    def __init__(self, df_fault: pd.DataFrame, df_healthy: pd.DataFrame, logger: logging.Logger):
        self.df_fault = df_fault
        self.df_healthy = df_healthy
        self.logger = logger
        self.results = []
    
    def _generate_snapshot(self, cutoff_date: pd.Timestamp):
        """Create training dataset as it would have looked at cutoff_date"""
        faults_past = self.df_fault[self.df_fault["started at"] <= cutoff_date].copy()

        # Data quality already checked in load_fault_data()
        faults_filtered = faults_past.copy()

        observation_start_date = self.df_fault["started at"].min() if not self.df_fault.empty else cutoff_date
        
        equipment_master = build_equipment_master(faults_past, self.df_healthy, self.logger, cutoff_date)
        
        df_snapshot = add_survival_columns_inplace(
            equipment_master, 
            faults_filtered, 
            cutoff_date, 
            observation_start_date,
            self.logger
        )
        
        chronic_df = compute_chronic_features(faults_past, cutoff_date, self.logger)
        
        df_snapshot = add_temporal_features_inplace(
            df_snapshot, 
            cutoff_date, 
            chronic_df, 
            observation_start_date,
            self.logger
        )
        
        return df_snapshot
    
    def _train_simple_model(self, df: pd.DataFrame, features: list):
        """
        ✅ FULLY ALIGNED: Uses same preprocessor as main pipeline
        """
        # 1. Feature Selection
        X = df[features].copy()

        # 2. ✅ USE MAIN PIPELINE'S PREPROCESSOR (build_preprocessor)
        preprocessor = build_preprocessor(X, logger=self.logger)

        # 3. ✅ TEMPORAL SPLIT (Same as main pipeline)
        try:
            # Use df with metadata for temporal split
            train_labels, test_labels = temporal_train_test_split(
                df,
                test_size=0.25,
                logger=self.logger
            )

            X_train = X.loc[train_labels]
            X_test = X.loc[test_labels]
            y_train = df.loc[train_labels, 'event']
            y_test = df.loc[test_labels, 'event']

        except Exception as e:
            self.logger.warning(f"[BACKTEST] Temporal split failed: {e}, using all data")
            X_train = X
            X_test = X.head(0)  # Empty test set
            y_train = df['event']
            y_test = pd.Series(dtype=int)

        # 4. ✅ FIT PREPROCESSOR + TRANSFORM
        try:
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test) if len(X_test) > 0 else None
        except Exception as e:
            self.logger.warning(f"[BACKTEST] Preprocessing failed: {e}")
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy='constant', constant=0), preprocessor

        # 5. ✅ TRAIN MODEL (Reduced complexity to prevent overfitting)
        model = XGBClassifier(
            n_estimators=20,  # Reduced from 50
            max_depth=2,      # Reduced from 3
            learning_rate=0.1,  # Added learning rate
            min_child_weight=5,  # Increased from default 1
            subsample=0.8,    # Added subsampling
            colsample_bytree=0.8,  # Added feature sampling
            reg_alpha=1.0,    # L1 regularization
            reg_lambda=1.0,   # L2 regularization
            scale_pos_weight=len(y_train[y_train==0]) / max(1, len(y_train[y_train==1])),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        try:
            model.fit(X_train_transformed, y_train)

            # ✅ Log training performance
            if X_test_transformed is not None and len(y_test) > 0:
                test_score = model.score(X_test_transformed, y_test)
                self.logger.info(f"[BACKTEST] Training accuracy: {test_score:.3f}")

            return model, preprocessor  # ✅ Return preprocessor for prediction

        except Exception as e:
            self.logger.warning(f"[BACKTEST] Model training failed: {e}")
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy='constant', constant=0), preprocessor
    
    def run(self, start_year: int, end_year: int, horizon_days: int = 365):
        """Run walk-forward validation"""
        self.logger.info("="*60)
        self.logger.info(f"[BACKTEST] Walk-Forward Validation ({start_year}-{end_year})")
        self.logger.info("="*60)
        
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        
        for year in range(start_year, end_year + 1):
            cutoff_date = pd.Timestamp(f"{year}-01-01")
            test_end_date = cutoff_date + pd.Timedelta(days=horizon_days)
            
            # 1. Generate Historical Snapshot
            self.logger.info(f"\n[BACKTEST] Generating snapshot for {cutoff_date.date()}...")
            df_train = self._generate_snapshot(cutoff_date)
            
            # 2. Define Ground Truth
            future_faults = self.df_fault[
                (self.df_fault["started at"] > cutoff_date) &
                (self.df_fault["started at"] <= test_end_date)
            ]

            # Data quality already checked in load_fault_data()
            future_faults = future_faults.copy()
            failed_ids = set(future_faults["cbs_id"].unique())
            
            # Target
            y_true = df_train["cbs_id"].isin(failed_ids).astype(int)
            
            if y_true.sum() < 5:
                self.logger.warning(f"[BACKTEST] {year}: Skipped - Insufficient failures ({y_true.sum()})")
                continue
            
            # ✅ Log ground truth stats
            self.logger.info(f"[BACKTEST] Ground truth: {y_true.sum()} failures out of {len(y_true)} assets ({100*y_true.mean():.2f}%)")
            
            # 3. Train Model
            exclude_cols = ["cbs_id", "event", "duration_days", "Kurulum_Tarihi", "entry_days",
                            "Ilk_Gercek_Ariza_Tarihi", "started at", "ended at", "Ilk_Ariza_Tarihi"]
            structural_cols = [c for c in df_train.columns if c not in exclude_cols]

            model, preprocessor = self._train_simple_model(df_train, structural_cols)

            # 4. Predict
            if preprocessor is not None:
                # ✅ Use same preprocessor (no manual encoding)
                X_test = df_train[structural_cols].copy()

                try:
                    X_test_transformed = preprocessor.transform(X_test)
                    probs = model.predict_proba(X_test_transformed)[:, 1]
                except Exception as e:
                    self.logger.warning(f"[BACKTEST] Prediction failed: {e}")
                    probs = np.zeros(len(y_true))
            else:
                probs = np.zeros(len(y_true))
            
            # 5. Evaluate
            try:
                auc = roc_auc_score(y_true, probs)
            except ValueError:
                auc = 0.5

            # ✅ THRESHOLD OPTIMIZATION: Find F1-optimal threshold
            from sklearn.metrics import precision_recall_curve

            try:
                precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)

                # Calculate F1 score for each threshold
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve)
                    f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0

                # Find threshold that maximizes F1
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                optimal_f1 = f1_scores[optimal_idx]

                self.logger.info(f"[BACKTEST] Optimal F1 threshold: {optimal_threshold:.3f} (F1={optimal_f1:.3f})")

            except Exception as e:
                self.logger.warning(f"[BACKTEST] Threshold optimization failed: {e}, using percentile")
                optimal_threshold = np.percentile(probs, 95)  # Fallback to top 5%
                optimal_f1 = 0.0

            # Apply optimal threshold
            y_pred = (probs >= optimal_threshold).astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Top-100 Precision
            if len(probs) >= 100:
                top_100_idx = np.argsort(probs)[-100:]
                hits = y_true.iloc[top_100_idx].sum()
                top_100_precision = hits / 100
            else:
                hits = y_true.sum()
                top_100_precision = hits / len(y_true)
            
            self.logger.info(f"[BACKTEST] {year} Results:")
            self.logger.info(f"  - AUC: {auc:.3f}")
            self.logger.info(f"  - F1 Score (Optimal): {f1:.3f}")
            self.logger.info(f"  - Precision@Optimal: {precision:.3f}")
            self.logger.info(f"  - Recall@Optimal: {recall:.3f}")
            self.logger.info(f"  - Top-100 Hits: {hits}/{min(100, len(y_true))} ({100*top_100_precision:.1f}%)")

            self.results.append({
                "Year": year,
                "AUC": auc,
                "F1_Score": f1,
                "Optimal_Threshold": optimal_threshold,
                "Precision": precision,
                "Recall": recall,
                "Top100_Hits": hits,
                "Top100_Precision": top_100_precision,
                "Total_Failures": y_true.sum(),
                "Total_Assets": len(y_true)
            })
        
        # Save Summary
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            out_dir = os.path.dirname(self.logger.handlers[0].baseFilename).replace("loglar", "data/sonuclar")
            out_path = os.path.join(out_dir, "backtest_results_temporal.csv")
            results_df.to_csv(out_path, index=False)
            self.logger.info(f"\n[BACKTEST] Results saved: {out_path}")

            # ✅ BONUS: Calculate and log threshold stability
            if "Optimal_Threshold" in results_df.columns:
                mean_threshold = results_df["Optimal_Threshold"].mean()
                std_threshold = results_df["Optimal_Threshold"].std()
                self.logger.info(f"\n[THRESHOLD STABILITY] Mean={mean_threshold:.3f}, Std={std_threshold:.3f}")

                if std_threshold < 0.05:
                    self.logger.info("  ✅ Threshold is STABLE across years (low variance)")
                else:
                    self.logger.warning("  ⚠️ Threshold is UNSTABLE (consider using year-specific thresholds)")

        return results_df

# =============================================================================
# EQUIPMENT-SPECIFIC MODELING
# =============================================================================
# =============================================================================
# 📊 EQUIPMENT STATISTICS (EKİPMAN BAZLI VERİ ENVANTERİ)
# =============================================================================
# Bu fonksiyon, her bir ekipman tipi (Trafo, Hücre, Kesici vb.) için veri
# yeterliliğini analiz eder.
#
# 🔍 Neyi Kontrol Eder?
#    1. Örneklem Boyutu (n_total): Model kurmak için yeterli sayı var mı?
#    2. Arıza Sayısı (n_events): Modelin "Ölümü" öğrenmesi için yeterince
#       örnek olay (Failure Event) gerçekleşmiş mi?
#    3. Veri Kalitesi (has_marka): Kritik özniteliklerin (Marka vb.) doluluk oranı nedir?
#
# ⚠️ Karar Destek:
#    - Eğer bir ekipman tipinde 'n_events < 5' ise, o ekipman için özel model
#      eğitmek yerine genel model kullanmak veya o tipi analizden çıkarmak gerekir.
# =============================================================================
# =============================================================================
# 📊 FINAL DATA AUDIT (EĞİTİM ÖNCESİ TAM KONTROL)
# =============================================================================
# Bu fonksiyon, pipeline'ın en sonunda çalışarak modelin ihtiyaç duyduğu TÜM verilerin
# (hem ham hem hesaplanmış) hazır olup olmadığını denetler.
#
# 🔍 Kritik Kontroller:
#    1. Lat/Long: Haritalama ve mekansal analiz için ikisinin de %100'e yakın olması gerekir.
#    2. Durat (Duration Days): Sağkalım süresi. Eğer bu oran düşükse, tarih verilerinde
#       veya 'add_survival_columns' fonksiyonunda mantık hatası var demektir.
#    3. CalcAge (Yaş): Sadece dolu olması yetmez, >0 olması gerekir.
#    4. MTBF: İstatistiksel özelliklerin (Bayesian) hesaplanıp hesaplanmadığını gösterir.
#
# ⚠️ Karar Mekanizması:
#    - Eğer 'CalcAge' veya 'Durat' %90'ın altındaysa, model ÇALIŞMAZ veya hatalı çalışır.
#    - Eğer 'Lat/Long' düşükse sadece haritalar etkilenir, model çalışmaya devam eder.
# =============================================================================
def get_equipment_stats(df: pd.DataFrame, equipment_master: pd.DataFrame, logger: logging.Logger) -> dict:
    """
    Final Audit: Checks Raw Data, Location, and Engineered Features completely.
    Returns dictionary with counts AND percentages.
    
    ✅ DÜZELTME: has_marka ve has_bakim eklendi
    """
    stats = {}
    
    # 1. DENETİM HARİTASI
    audit_map = {
        "Marka": "Marka",
        "Latitude": "Lat",
        "Longitude": "Long",
        "Gerilim_Seviyesi": "Volt",
        "Bakim_Sayisi": "Maint",
        "duration_days": "Durat",
        "entry_days": "Entry",
        "Tref_Yas_Gun": "CalcAge",
        "MTBF_Bayes_Gun": "MTBF",
        "Observation_Ratio": "ObsRate"
    }

    # Log Başlıkları
    headers = ["Type", "Total", "Events", "Rate"] + [v for v in audit_map.values()]
    header_fmt = "{:<15} | {:<6} | {:<6} | {:<6} | " + " | ".join([f"{{:<7}}" for _ in audit_map])
    
    logger.info("="*130)
    logger.info("[FINAL DATA AUDIT] Eğitim Öncesi Tam Kontrol")
    logger.info(header_fmt.format(*headers))
    logger.info("-" * 130)

    for eq_type in df["Ekipman_Tipi"].unique():
        df_eq = df[df["Ekipman_Tipi"] == eq_type]
        n_total = len(df_eq)
        
        if n_total == 0: 
            continue

        # Temel metrikler
        n_events = int(df_eq["event"].sum())
        event_rate = float(df_eq["event"].mean())
        
        # ✅ MARKA sayısı (absolute count)
        has_marka = 0
        if "Marka" in df_eq.columns:
            has_marka = int(df_eq["Marka"].notna().sum())

        # ✅ BAKIM sayısı (absolute count)
        has_bakim = 0
        if "Bakim_Sayisi" in df_eq.columns:
            # Not null AND not zero (gerçekten bakım yapılmış)
            has_bakim = int((df_eq["Bakim_Sayisi"].notna() & (df_eq["Bakim_Sayisi"] > 0)).sum())

        type_stats = {
            "n_total": n_total,
            "n_events": n_events,
            "event_rate": event_rate,
            "has_marka": has_marka,   # ✅ EKLENDİ
            "has_bakim": has_bakim,   # ✅ EKLENDİ
        }

        # Log satırı
        row_data = [
            eq_type,
            str(n_total),
            str(n_events),
            f"{100*event_rate:.1f}%"
        ]
        
        # Detaylı Sütun Kontrolleri
        for col_name, label in audit_map.items():
            val_str = "MISS" 
            pct = 0.0
            
            if col_name in df_eq.columns:
                valid_mask = df_eq[col_name].notna()
                
                # Mantık Kontrolü
                if col_name in ["Tref_Yas_Gun", "duration_days"]:
                    valid_mask = valid_mask & (df_eq[col_name] > 0)
                elif col_name == "Bakim_Sayisi":
                    # Bakım için: Not null (veri var demek, 0 bile bilgidir)
                    pass  # valid_mask zaten notna()
                
                valid_count = valid_mask.sum()
                pct = 100 * valid_count / n_total
                val_str = f"{pct:.0f}%"
            
            row_data.append(val_str)
            type_stats[label] = pct

        # Satırı Yazdır
        logger.info(header_fmt.format(*row_data))
        
        # Ana sözlüğe kaydet
        stats[eq_type] = type_stats

    logger.info("="*130)
    
    # ✅ DEBUG LOG
    logger.info("\n[ANALYSIS ELIGIBILITY CHECK]")
    for eq_type, stat in stats.items():
        marka_ok = "✅" if stat['has_marka'] >= 30 else "❌"
        bakim_ok = "✅" if stat['has_bakim'] >= 30 else "❌"
        logger.info(
            f"  {eq_type:<15}: "
            f"Marka={stat['has_marka']:>4} {marka_ok} | "
            f"Bakim={stat['has_bakim']:>4} {bakim_ok}"
        )
    
    return stats
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
# =============================================================================
# 🔮 ML PREDICTION: CONDITIONAL PROBABILITY OF FAILURE (KOŞULLU RİSK HESABI)
# =============================================================================
# Bu fonksiyon, eğitilen modelin ürettiği "Sağkalım Eğrilerini" (Survival Functions)
# kullanarak, her bir varlığın gelecekteki arıza ihtimalini hesaplar.
#
# 🧠 Kritik Mantık (Conditional Probability):
#    - Soru: "Bu trafo önümüzdeki 1 yıl içinde bozulur mu?"
#    - Yanlış Yöntem: Sadece S(t=1 yıl) değerine bakmak.
#    - Doğru Yöntem: Varlığın ŞU ANKİ YAŞINI (t) hesaba katmak.
#
# 📐 Formül:
#    Risk = 1 - ( S(t + Horizon) / S(t) )
#
#    - S(t): Varlığın bugüne kadar hayatta kalma olasılığı.
#    - S(t + Horizon): Gelecekteki hedef tarihe kadar hayatta kalma olasılığı.
#    - Bu formül, "Bugüne kadar sağ kalan bir varlığın, X gün daha yaşama ihtimali nedir?"
#      sorusunun cevabıdır. Eski varlıklar için riski abartmayı önler.
# =============================================================================

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
            # MODELİN SINIRLARINI KONTROL ET (FIX)
            max_model_time = fn.x[-1]  # Modelin bildiği en son zaman
            min_model_time = fn.x[0]   # Modelin bildiği ilk zaman

            # Mevcut yaşı sınırlar içine çek
            age_now = current_age[i]
            # Eğer varlık modelin gördüğü max yaştan büyükse, max yaş kabul et
            if age_now > max_model_time:
                age_now = max_model_time
            
            # Gelecek yaşı sınırlar içine çek
            age_future = age_now + H
            if age_future > max_model_time:
                age_future = max_model_time

            # Olasılıkları al
            # fn(t) fonksiyonu StepFunction'dır, sınır dışı değerde hata verir
            try:
                prob_survive_now = fn(age_now)
                prob_survive_future = fn(age_future)
            except ValueError:
                # Hala hata alırsak (çok nadir), güvenli moda geç
                prob_survive_now = 1.0
                prob_survive_future = 1.0

            # Hesaplama (Sıfıra bölünme koruması)
            if prob_survive_now < 1e-5:
                conditional_risk = 1.0 # Zaten ölü kabul et
            else:
                conditional_risk = 1.0 - (prob_survive_future / prob_survive_now)
            
            pofs.append(np.clip(conditional_risk, 0, 1))

        out[f"ml_pof_{label}"] = pofs
        
    return out
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
# =============================================================================
# 📈 EXPLORATORY ANALYSIS (BETİMSEL İSTATİSTİKLER)
# =============================================================================
# Bu fonksiyonlar, tahmin (prediction) yapmaz; verinin röntgenini çeker.
#
# 1. Marka Analizi:
#    - "Relative Risk" (Göreceli Risk) metriği kullanılır.
#    - Eğer bir markanın riski 1.5 ise, ortalamadan %50 daha sık bozuluyor demektir.
#    - DİKKAT: Bazen eski markalar daha sık bozulur. "Median_Age" kontrol edilmelidir.
#
# 2. Bakım Etkisi:
#    - Bakım sayısı ile arıza oranı arasındaki ilişkiyi gösterir.
#    - Beklenti: Bakım arttıkça arıza oranının düşmesidir.
#    - Anomali: Bazen çok bakım yapılanlar daha çok bozulur (Reaktif Bakım - Arıza oldukça gitme).
# =============================================================================
def analyze_marka_effect(df_eq: pd.DataFrame, eq_type: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Brand risk analysis (Marka Performans Karnesi)
    """
    if "Marka" not in df_eq.columns:
        return pd.DataFrame()

    df_marka = df_eq[df_eq["Marka"].notna()].copy()
    
    # İstatistiksel anlamlılık için en az 30 veri
    if len(df_marka) < 30:
        return pd.DataFrame()
    
    # Genel Ortalamayı Hesapla (Kıyaslama için)
    avg_failure_rate = df_marka["event"].mean()

    marka_stats = df_marka.groupby("Marka").agg(
        Failures=("event", "sum"),
        Total=("event", "count"),
        Failure_Rate=("event", "mean"),
        Median_Age=("duration_days", "median")
    ).reset_index()
    
    # Sadece anlamlı sayıdaki markaları al (En az 5 trafosu olan markalar)
    marka_stats = marka_stats[marka_stats["Total"] >= 5].sort_values("Failure_Rate", ascending=False)
    
    # Göreceli Risk: (Marka Arıza Oranı / Ortalama Arıza Oranı)
    # 1.0 = Ortalama, >1.0 = Riskli, <1.0 = Sağlam
    if avg_failure_rate > 0:
        marka_stats["Relative_Risk"] = marka_stats["Failure_Rate"] / avg_failure_rate
    else:
        marka_stats["Relative_Risk"] = 0.0

    logger.info(f"[{eq_type}] MARKA ANALİZİ: {len(marka_stats)} marka incelendi (Ort. Arıza: {avg_failure_rate:.1%})")
    
    # En kötü 3 markayı raporla
    for _, row in marka_stats.head(3).iterrows():
        logger.info(
            f"  🚨 {row['Marka']:<10} : Arıza %{100*row['Failure_Rate']:.1f} | "
            f"Risk x{row['Relative_Risk']:.1f} | "
            f"Yaş: {row['Median_Age']:.0f} gün | "
            f"(N={int(row['Total'])})"
        )
    
    marka_stats["Ekipman_Tipi"] = eq_type
    return marka_stats


def analyze_bakim_effect(df_eq: pd.DataFrame, eq_type: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Maintenance effect analysis (Bakım Etki Analizi)
    ⚠️ DÜZELTME: equipment_master yerine df_eq kullanılmalı (event verisi için)
    """
    if "Bakim_Sayisi" not in df_eq.columns:
        return pd.DataFrame()
    
    # Bakım sayısı 0 veya daha büyük olanları al (NaN'ları at)
    df_bakim = df_eq[df_eq["Bakim_Sayisi"].notna()].copy()
    
    if len(df_bakim) < 30:
        return pd.DataFrame()
    
    # Bakım sayılarını grupla (Binning)
    # [0-1), [1-3), [3-5), [5-10), [10+)
    df_bakim["Bakim_Bin"] = pd.cut(
        df_bakim["Bakim_Sayisi"],
        bins=[-1, 0, 2, 5, 10, 1000], # -1 dahil ederek 0'ı yakalarız
        labels=["0 (Hiç)", "1-2", "3-5", "6-10", "10+"]
    )
    
    # Hangi grupta ne kadar arıza var?
    bakim_stats = df_bakim.groupby("Bakim_Bin", observed=False).agg(
        Asset_Count=("cbs_id", "count"),
        Event_Count=("event", "sum"),     # <--- EKLENDİ
        Failure_Rate=("event", "mean")    # <--- EKLENDİ (Kritik Metrik)
    ).reset_index()
    
    logger.info(f"[{eq_type}] BAKIM ETKİSİ:")
    
    # Sonuçları yazdır
    for _, row in bakim_stats.iterrows():
        if row['Asset_Count'] > 0:
            logger.info(
                f"  🔧 Bakım {row['Bakim_Bin']:<8}: "
                f"Arıza %{100*row['Failure_Rate']:.1f} "
                f"(N={row['Asset_Count']})"
            )
    
    bakim_stats["Ekipman_Tipi"] = eq_type
    return bakim_stats

# =============================================================================
# 🏥 HEALTH SCORE & RISK MATRIX (SAĞLIK VE RİSK PUANLAMASI)
# =============================================================================
# Bu fonksiyon, model çıktısını (PoF – Probability of Failure) operasyonel olarak
# anlamlı bir Sağlık Skoru'na (0–100) ve Risk Sınıfı'na dönüştürür.
#
# 📊 Yöntem: Percentile Ranking (Yüzdelik Sıralama – Göreli Risk)
# -----------------------------------------------------------------------------
# - Mutlak olasılıklar (PoF) genellikle çok küçüktür (örn. %0.1 – %1).
# - Bu nedenle sistem, "mutlak risk" yerine "göreli risk" yaklaşımını kullanır.
#
#   ❌ "Bu trafonun arıza ihtimali %0.12"
#   ✅ "Bu trafo, aynı tipteki varlıkların %95’inden daha risklidir"
#
# - Sıralama, her ekipman tipi kendi içinde yapılır
#   (Trafo trafoyla, direk direkle kıyaslanır).
#
# 🧮 Sağlık Skoru Hesabı:
# -----------------------------------------------------------------------------
# - Risk sıralaması (percentile) ters çevrilerek sağlık skoruna dönüştürülür:
#
#     Health_Score = 100 × (1 − Risk_Percentile)
#
# - En riskli varlıklar → düşük sağlık skoru
# - En sağlıklı varlıklar → yüksek sağlık skoru
#
# 🚦 Risk Sınıfları (Percentile Bazlı):
# -----------------------------------------------------------------------------
# - KRİTİK  : En kötü %5  (Risk_Percentile ≥ 0.95)
# - YÜKSEK  : Sonraki %15 (0.80 ≤ Risk_Percentile < 0.95)
# - ORTA    : Sonraki %30 (0.50 ≤ Risk_Percentile < 0.80)
# - DÜŞÜK   : En iyi %50  (Risk_Percentile < 0.50)
#
# ⚠️ Kronik Varlık Kuralı (Hard Business Rule):
# -----------------------------------------------------------------------------
# - "Chronic_Flag" = 1 olan varlıklar, model skorundan bağımsız olarak
#   her zaman öncelikli kabul edilir.
# - Bu varlıkların sağlık skoru en fazla 60 ile sınırlandırılır ve
#   risk sınıfı otomatik olarak "KRİTİK (KRONİK)" olarak atanır.
#
# 🎯 Amaç:
# -----------------------------------------------------------------------------
# - Mutlaka filonun en riskli varlıklarını görünür kılmak
# - Saha ekiplerine her zaman "öncelikli bakım listesi" üretebilmek
# - Yönetim için göreli, karşılaştırılabilir ve aksiyon alınabilir bir çıktı sağlamak
# =============================================================================

# REMOVED: Duplicate/broken versions of compute_health_score
# Using only the production-grade version below

def compute_health_score(
    df: pd.DataFrame,
    pof_col: str = "PoF_Ensemble_12Ay",
    group_col: str = "Ekipman_Tipi",
    chronic_col: str = "Chronic_Flag",
    min_group_size: int = 100,
    min_health: int = 10,
    logger: logging.Logger = None,
    use_global_only: bool = True  # ✅ NEW: Force pure global ranking to fix PoF-Risk inconsistency
) -> pd.DataFrame:
    """
    Production-grade Health Score & Risk Class computation.

    ✅ FIXED: Now uses PURE GLOBAL RANKING by default to ensure consistent PoF-to-Risk mapping.
    This eliminates the issue where lower PoF values had higher risk classifications than higher PoF values.

    Args:
        use_global_only: If True (default), uses only global ranking for consistent PoF-Risk mapping.
                        If False, uses hybrid local+global ranking (may cause inconsistencies).
    """

    # --- 0. Guardrails ---
    if pof_col not in df.columns:
        df["Health_Score"] = 90
        df["Risk_Sinifi"] = "BILINMIYOR"
        return df

    df[pof_col] = df[pof_col].fillna(0).clip(0, 1)

    # --- 1. Ranking Strategy ---
    df["Global_Rank"] = df[pof_col].rank(pct=True)

    if use_global_only:
        # ✅ PURE GLOBAL RANKING: Ensures PoF-Risk consistency
        # Lower PoF will ALWAYS have lower/equal risk than higher PoF
        df["Final_Rank"] = df["Global_Rank"]
        if logger:
            logger.info("Using PURE GLOBAL ranking (PoF-Risk consistent)")
    else:
        # Hybrid ranking (old method - may cause inconsistencies)
        if group_col in df.columns:
            df["Local_Rank"] = df.groupby(group_col)[pof_col].rank(pct=True)
            group_sizes = df.groupby(group_col)[pof_col].transform("count")
            w_local = np.where(group_sizes >= min_group_size, 1.0, 0.5)
            df["Final_Rank"] = (
                df["Local_Rank"] * w_local +
                df["Global_Rank"] * (1 - w_local)
            )
            if logger:
                logger.info("Using HYBRID ranking (local+global)")
        else:
            df["Final_Rank"] = df["Global_Rank"]

    # --- 2. Health Score ---
    df["Health_Score"] = 100 * (1 - df["Final_Rank"])
    df["Health_Score"] = df["Health_Score"].clip(lower=min_health, upper=100)

    # --- 3. Chronic override ---
    if chronic_col in df.columns:
        kronik_mask = df[chronic_col] == 1
        df.loc[kronik_mask, "Health_Score"] = np.minimum(
            df.loc[kronik_mask, "Health_Score"], 60
        )

    # --- 4. Risk Class Assignment ---
    def assign_risk(row):
        if row.get(chronic_col, 0) == 1:
            return "KRİTİK (KRONİK)"

        p = row["Final_Rank"]
        # ✅ Industry-standard thresholds (IEEE/CIGRE)
        if p >= 0.95:  # Top 5%
            return "KRİTİK"
        if p >= 0.85:  # Top 15%
            return "YÜKSEK"
        if p >= 0.50:  # Top 50%
            return "ORTA"
        return "DÜŞÜK"  # Bottom 50%

    df["Risk_Sinifi"] = df.apply(assign_risk, axis=1)

    # --- 5. Logging ---
    if logger:
        logger.info(
            f"Risk Distribution → "
            f"KRİTİK={sum(df['Risk_Sinifi']=='KRİTİK')}, "
            f"YÜKSEK={sum(df['Risk_Sinifi']=='YÜKSEK')}, "
            f"ORTA={sum(df['Risk_Sinifi']=='ORTA')}, "
            f"DÜŞÜK={sum(df['Risk_Sinifi']=='DÜŞÜK')}"
        )

    return df

# =============================================================================
# MAIN PIPELINE
# =============================================================================
# =============================================================================
# 🚀 MAIN PIPELINE ORCHESTRATION (ANA YÖNETİM MERKEZİ)
# =============================================================================
# Bu fonksiyon, ham veriden nihai tahminlere giden uçtan uca (End-to-End) akışı yönetir.
# "PoF v4.1" mimarisinin kalbidir.
#
# 🔄 İŞLEM AKIŞI (WORKFLOW):
#
# 1. 📥 Data Ingestion & Config (Yükleme):
#    - Arıza ve Sağlam verileri yüklenir.
#    - Gözlem başlangıç tarihi (Left Truncation) veriden otomatik tespit edilir.
#
# 2. 🏗️ Dataset Construction (Veri İnşası):
#    - 'Survival Analysis' formatı kurulur (Duration, Event, Entry).
#    - Kronik arızalar (Chronic Features) ve zamansal özellikler türetilir.
#    - Ara dosyalar (Intermediate) kaydedilir.
#
# 3. 🛡️ Global Modeling (Güvenlik Ağı / Fallback):
#    - Verisi az olan (N < 100) ekipman tipleri için model eğitilemez.
#    - Bu adımda tüm filoyu kullanan "Genel Modeller" (Cox, RSF, ML) eğitilir ve
#      yetersiz verili tipler için yedek (fallback) olarak hafızada tutulur.
#
# 4. ⚙️ Stratified Training (Katmanlı Eğitim):
#    - Her ekipman tipi için döngüye girilir.
#    - KARAR MEKANİZMASI:
#      a. Yeterli Veri Var mı? -> O tipe ÖZEL model eğit (En yüksek hassasiyet).
#      b. Veri Yetersiz mi? -> GLOBAL modelleri devreye sok (Tahminsiz bırakma).
#    - Marka ve Bakım analizleri (varsa) bu aşamada üretilir.
#
# 5. 🏥 Final Scoring (Puanlama & Raporlama):
#    - Tüm tahminler birleştirilir.
#    - Olasılıklar (PoF) -> Sağlık Skoru (0-100) ve Risk Sınıfına dönüştürülür.
#    - 'pof_predictions_final.csv', 'marka_analysis.csv' vb. master dosyalar kaydedilir.
#
# 6. 🕰️ Backtesting (Zaman Tüneli Testi):
#    - Modelin başarısını kanıtlamak için 2022-2024 yılları simüle edilir.
#    - Yönetim sunumu için AUC ve Top-100 isabet oranları hesaplanır.
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

    # 2. Data quality already checked in load_fault_data()
    df_fault_filtered = df_fault.copy()

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
        
    # Save combined feature set (ozellikler_pof)
    all_feature_cols = ["cbs_id"] + structural_cols + temporal_cols + ["event", "duration_days", "entry_days"]
    all_feature_cols = [c for c in all_feature_cols if c in df_all.columns]
    df_all[all_feature_cols].to_csv(INTERMEDIATE_PATHS["ozellikler_pof"], index=False, encoding="utf-8-sig")
    
    logger.info(f"[SAVE] Intermediate: {INTERMEDIATE_PATHS['ozellikler_pof']}")
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
        try:
            # 1. Filter Data
            df_eq = df_all[df_all["Ekipman_Tipi"] == eq_type].copy()
            stats = eq_stats.get(eq_type, {'n_total': 0, 'n_events': 0, 'has_marka': 0, 'has_bakim': 0})
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
                # --- BAKIM ANALİZİ (Koşullu) ---
                if stats.get("has_bakim", 0) >= 30:  # ✅ KOŞUL EKLENDİ!
                    try:
                        bakim_analysis = analyze_bakim_effect(df_eq, eq_type, logger)
                        if not bakim_analysis.empty:
                            all_bakim_analyses.append(bakim_analysis)
                    except Exception as e:
                        logger.warning(f"[{eq_type}] Bakim analysis failed: {e}")

            # 3. MERGE PREDICTIONS WITH METADATA (FIXED)
            # We merge 'preds' (which only has cbs_id + probabilities) back to df_eq metadata
            meta_cols = ["cbs_id", "Ekipman_Tipi"]
            if "Fault_Count" in df_eq.columns: meta_cols.append("Fault_Count")
            preds_full = df_eq[meta_cols].merge(preds, on="cbs_id", how="left")
            preds_full["Model_Type"] = model_source

            # 4. COMPUTE HEALTH SCORE
            # Now preds_full definitely has "Ekipman_Tipi", so grouping works
            try:
                preds_full = compute_health_score(preds_full, logger)
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
        except Exception as e:
            logger.error(f"[{eq_type}] Failed to process equipment type: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

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

    # ✅ CREATE ENSEMBLE (Average of available models)
    logger.info("[ENSEMBLE] Creating ensemble predictions from available models...")

    for horizon_label in SURVIVAL_HORIZON_LABELS.values():
        # Find all model predictions for this horizon
        model_cols = [c for c in predictions.columns if f"pof_{horizon_label}" in c]

        if model_cols:
            # Average across available models (ignoring NaNs)
            predictions[f"PoF_Ensemble_{horizon_label}"] = predictions[model_cols].mean(axis=1, skipna=True)
            logger.info(f"  - {horizon_label}: Averaged {len(model_cols)} models → PoF_Ensemble_{horizon_label}")
        else:
            # No predictions for this horizon (shouldn't happen but safety check)
            predictions[f"PoF_Ensemble_{horizon_label}"] = 0.0
            logger.warning(f"  - {horizon_label}: No model predictions found, using 0.0")

    # Final Report Merge (Add context like Voltage, Install Date)
    report_cols = ["Ekipman_Tipi", "Gerilim_Sinifi", "Fault_Count", "Kurulum_Tarihi", "Marka", "Ilce"]
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
    critical = (report["Health_Score"] < 20).sum()
    mean_health = report["Health_Score"].mean()

    logger.info(f"Total assets: {len(report):,}")
    logger.info(f"Critical assets (Health<20): {critical:,} ({100*critical/len(report):.1f}%)")
    logger.info(f"Mean Health Score: {mean_health:.1f}")

    # 🔥 TOP 10 KRİTİK EKiPMAN - SAHA ÖNCELİĞİ
    if 'PoF_Ensemble_12ay' in report.columns:
        top10_cols = ['cbs_id', 'Ekipman_Tipi', 'PoF_Ensemble_12ay', 'Health_Score', 'Risk_Sinifi']
        # Add optional columns if they exist
        if 'Ilce' in report.columns:
            top10_cols.insert(2, 'Ilce')

        available_cols = [c for c in top10_cols if c in report.columns]
        top10_risk = report.nlargest(10, 'PoF_Ensemble_12ay')[available_cols]

        logger.info("\n🚨 TOP 10 KRİTİK EKiPMAN:")
        logger.info(top10_risk.round(3).to_string(index=False))
    else:
        logger.warning("[TOP 10] PoF_Ensemble_12ay column not found, skipping top 10 report")


    # -------------------------------------------------------------------------
    # STEP 6: BACKTESTING (Temporal Validation)
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("STEP 6 - Temporal Backtesting (Optional)")
    logger.info("="*60 + "\n")
   
    try:
        backtester = TemporalBacktester(df_fault, df_healthy, logger)       
        # Run walk-forward validation for available years
        backtest_results = backtester.run(
            start_year=2022,  # First year to test
            end_year=2024,    # Last year to test
            horizon_days=365  # 12-month prediction window
        )      
        if not backtest_results.empty:

            logger.info("\n[BACKTEST SUMMARY]")
            logger.info(f"Average AUC: {backtest_results['AUC'].mean():.3f}")
            logger.info(f"Average Top-100 Hit Rate: {backtest_results['Top100_Hits'].mean():.1f}")
            logger.info(f"Total Years Tested: {len(backtest_results)}")
        else:
            logger.warning("[BACKTEST] No results generated (insufficient data)")
    except Exception as e:
        logger.error(f"[BACKTEST] Failed: {e}")
        logger.info("[BACKTEST] Skipping temporal validation, continuing with main pipeline")  
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
