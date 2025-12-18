# -*- coding: utf-8 -*-
"""
PoF3 - Ultimate Reporting Engine (v5.0)
=============================================================================
COMPLETE DASHBOARD GENERATOR
Features:
  1. Operational Dashboard (Trends & Seasonality)
  2. Health Score Deep Dive (Distributions)
  3. Risk Drivers Analysis (Proxy for SHAP - Correlation) [NEW]
  4. Interactive Risk Maps (Folium HTML)
  5. Survival Curves (Kaplan-Meier)
  6. Executive PowerPoint Presentation (FIXED)
  7. Final Excel Action Lists
  
Author: PoF3 Team
Date: December 2025
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- OPTIONAL DEPENDENCIES ---
try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    print("⚠️ 'python-pptx' not found. PPTX generation skipped.")

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "sonuclar")
INPUT_DIR = os.path.join(BASE_DIR, "data", "girdiler")

# Files
PRED_FILE = os.path.join(OUTPUT_DIR, "pof_predictions_final.csv")
MASTER_FILE = os.path.join(OUTPUT_DIR, "equipment_master.csv")
FULL_DATA_FILE = os.path.join(OUTPUT_DIR, "model_input_data_full.csv") # For Correlations
SURVIVAL_FILE = os.path.join(OUTPUT_DIR, "survival_base.csv")
RAW_FAULT_FILE = os.path.join(INPUT_DIR, "ariza_final.xlsx")

VISUAL_DIR = os.path.join(BASE_DIR, "gorseller")
ACTION_DIR = os.path.join(OUTPUT_DIR, "aksiyon_listeleri")

# Styling
plt.style.use('ggplot')
sns.set_palette("husl")

# --- LOGGER ---
def setup_logger():
    log_dir = os.path.join(BASE_DIR, "loglar")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("PoF_Report")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    fh = logging.FileHandler(os.path.join(log_dir, "reporting_engine_v5.log"), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

# --- DATA PREP ---
def load_predictions(logger):
    if not os.path.exists(PRED_FILE):
        logger.error(f"❌ Input file missing: {PRED_FILE}")
        return None
    df = pd.read_csv(PRED_FILE)
    
    # Self-Heal: Risk Class
    if 'Risk_Sinifi' not in df.columns:
        risk_score = 100 - df.get('Health_Score', 100)
        q95, q80 = risk_score.quantile(0.95), risk_score.quantile(0.80)
        df['Risk_Sinifi'] = risk_score.apply(lambda x: 'KRİTİK' if x>=q95 else 'YÜKSEK' if x>=q80 else 'ORTA' if x>=50 else 'DÜŞÜK')

    # Merge Context
    if os.path.exists(MASTER_FILE):
        master = pd.read_csv(MASTER_FILE)
        df['cbs_id'] = df['cbs_id'].astype(str)
        master['cbs_id'] = master['cbs_id'].astype(str)
        cols = [c for c in ['Latitude', 'Longitude', 'Ilce'] if c in master.columns and c not in df.columns]
        if cols: df = df.merge(master[['cbs_id'] + cols], on='cbs_id', how='left')
            
    return df

# --- MODULE 1: OPERATIONAL DASHBOARD ---
def generate_eda_dashboard(logger):
    if not os.path.exists(RAW_FAULT_FILE):
        return

    logger.info("📊 Generating Operational Dashboard...")
    try:
        df = pd.read_excel(RAW_FAULT_FILE, usecols=['started at', 'cause code', 'Şebeke Unsuru'])
        df['started at'] = pd.to_datetime(df['started at'], errors='coerce')
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Monthly Trend
        ax1 = plt.subplot(2, 3, 1)
        df.groupby(df['started at'].dt.to_period('M')).size().plot(kind='line', marker='o', color='steelblue', ax=ax1)
        ax1.set_title('Aylık Arıza Trendi')
        
        # 2. Equipment Dist
        ax2 = plt.subplot(2, 3, 2)
        df['Şebeke Unsuru'].value_counts().head(10).plot(kind='barh', color='coral', ax=ax2)
        ax2.set_title('En Çok Arızalanan Ekipmanlar')
        ax2.invert_yaxis()

        # 3. Cause Code
        ax3 = plt.subplot(2, 3, 3)
        is_fuse = df['cause code'].str.contains('Sigorta', case=False, na=False).sum()
        ax3.pie([len(df)-is_fuse, is_fuse], labels=['Gerçek Arıza', 'Sigorta Atığı'], autopct='%1.1f%%', colors=['#d62728', '#2ca02c'])
        ax3.set_title('Arıza Tipi')

        # 4. Weekly
        ax4 = plt.subplot(2, 3, 4)
        df['Week'] = df['started at'].dt.isocalendar().week
        df.groupby('Week').size().plot(kind='area', color='lightcoral', alpha=0.6, ax=ax4)
        ax4.set_title('Haftalık Yoğunluk')

        # 5. Hourly
        ax5 = plt.subplot(2, 3, 5)
        df.groupby(df['started at'].dt.hour).size().plot(kind='bar', color='purple', ax=ax5)
        ax5.set_title('Saatlik Dağılım')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUAL_DIR, "00_operasyonel_pano.png"), dpi=300)
        plt.close()
        logger.info("   > Saved: 00_operasyonel_pano.png")
    except: pass

# --- MODULE 2: HEALTH SCORE DEEP DIVE ---
def generate_health_deep_dive(df, logger):
    logger.info("❤️ Generating Health Score Deep Dive...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Dist
    axes[0, 0].hist(df['Health_Score'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 0].axvline(df['Health_Score'].mean(), color='red', linestyle='--')
    axes[0, 0].set_title('Sağlık Skoru Dağılımı')
    
    # 2. By Type
    df.groupby('Ekipman_Tipi')['Health_Score'].mean().sort_values().plot(kind='barh', color='coral', ax=axes[0, 1])
    axes[0, 1].set_title('Tip Bazlı Ort. Sağlık')
    
    # 3. Pie
    df['Risk_Sinifi'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1, 0])
    axes[1, 0].set_ylabel('')
    
    # 4. Scatter
    if 'PoF_Ensemble_12Ay' in df.columns:
        sns.scatterplot(data=df, x='Health_Score', y='PoF_Ensemble_12Ay', hue='Risk_Sinifi', ax=axes[1, 1], alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "02_saglik_detay_analizi.png"), dpi=300)
    plt.close()
    logger.info("   > Saved: 02_saglik_detay_analizi.png")

# --- MODULE 3: RISK DRIVERS (SHAP PROXY) ---
def generate_risk_drivers(logger):
    if not os.path.exists(FULL_DATA_FILE):
        logger.warning("⚠️ Full model data missing. Risk Drivers skipped.")
        return

    logger.info("🔍 Generating Risk Drivers (Correlation Analysis)...")
    try:
        df_full = pd.read_csv(FULL_DATA_FILE)
        
        # Determine Target (Invert Health Score or use Event)
        target_col = 'Risk_Index'
        if 'Health_Score' in df_full.columns:
            df_full[target_col] = 100 - df_full['Health_Score']
        elif 'event' in df_full.columns:
            df_full[target_col] = df_full['event']
        else:
            return

        # Select Numeric Features
        numeric_df = df_full.select_dtypes(include=[np.number])
        if target_col not in numeric_df.columns: return

        # Calculate Correlation
        corr = numeric_df.corrwith(numeric_df[target_col]).drop(target_col).sort_values(ascending=False)
        
        # Plot Top 15 Positive & Negative Correlations
        top_corr = pd.concat([corr.head(10), corr.tail(5)])
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 0 else 'green' for x in top_corr.values]
        top_corr.plot(kind='barh', color=colors)
        plt.title('Risk Arttırıcı/Azaltıcı Faktörler (Korelasyon Analizi)', fontsize=14)
        plt.xlabel('Korelasyon Katsayısı (Risk ile İlişki)')
        plt.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        
        plt.savefig(os.path.join(VISUAL_DIR, "04_risk_faktorleri.png"), dpi=300)
        plt.close()
        logger.info("   > Saved: 04_risk_faktorleri.png")
        
    except Exception as e:
        logger.error(f"   Drivers Failed: {e}")

# --- MODULE 4: INTERACTIVE MAP ---
def generate_interactive_map(df, logger):
    if not HAS_FOLIUM: return
    if 'Latitude' not in df.columns: return

    logger.info("🗺️ Generating Interactive Map...")
    map_data = df.dropna(subset=['Latitude', 'Longitude'])
    map_data = map_data[(map_data['Latitude'] != 0) & (map_data['Longitude'] != 0)]
    if map_data.empty: return

    m = folium.Map(location=[map_data['Latitude'].mean(), map_data['Longitude'].mean()], zoom_start=11, tiles='CartoDB dark_matter')
    colors = {'KRİTİK': 'red', 'YÜKSEK': 'orange', 'ORTA': 'yellow', 'DÜŞÜK': 'green'}
    
    for _, row in map_data[map_data['Risk_Sinifi'].isin(['KRİTİK', 'YÜKSEK'])].iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5, color=colors.get(row['Risk_Sinifi']), fill=True,
            popup=f"{row['Ekipman_Tipi']}: {row['Health_Score']:.1f}"
        ).add_to(m)
    
    HeatMap(map_data[['Latitude', 'Longitude', 'Risk_Score']].values.tolist(), radius=15).add_to(m)
    m.save(os.path.join(VISUAL_DIR, "01_interaktif_risk_haritasi.html"))
    logger.info("   > Saved: 01_interaktif_risk_haritasi.html")

# --- MODULE 5: SURVIVAL CURVES ---
def plot_survival_curves(logger):
    if not HAS_LIFELINES or not os.path.exists(SURVIVAL_FILE): return
    logger.info("📉 Generating Survival Curves...")
    
    df_surv = pd.read_csv(SURVIVAL_FILE)
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10,6))
    
    for etype in df_surv['Ekipman_Tipi'].value_counts().head(4).index:
        sub = df_surv[df_surv['Ekipman_Tipi']==etype]
        if len(sub)>10:
            kmf.fit(sub['duration_days'], sub['event'], label=etype)
            kmf.plot_survival_function(ci_show=False)
            
    plt.title("Varlık Ömür Eğrileri"); plt.savefig(os.path.join(VISUAL_DIR, "03_sagkalim_egrileri.png")); plt.close()
    logger.info("   > Saved: 03_sagkalim_egrileri.png")

# --- MODULE 6: PPTX REPORT (FIXED) ---
def generate_pptx_report(df, logger):
    if not HAS_PPTX: return
    logger.info("📽️ Generating PowerPoint...")
    prs = Presentation()
    
    # Slide 1: Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "PoF3 Varlık Sağlık Analizi"
    slide.placeholders[1].text = f"Yönetici Özeti\n{datetime.now().strftime('%d.%m.%Y')}"

    # Slide 2: Metrics
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Özet Metrikler"
    slide.placeholders[1].text = f"Toplam Varlık: {len(df):,}\nKritik Varlık: {len(df[df['Risk_Sinifi']=='KRİTİK']):,}\nOrtalama Sağlık: {df['Health_Score'].mean():.1f}"

    # Helper
    def add_img_slide(img, title):
        path = os.path.join(VISUAL_DIR, img)
        if os.path.exists(path):
            s = prs.slides.add_slide(prs.slide_layouts[5])
            s.shapes.title.text = title
            s.shapes.add_picture(path, Inches(0.5), Inches(1.5), width=Inches(9))

    add_img_slide("00_operasyonel_pano.png", "Operasyonel Durum")
    add_img_slide("02_saglik_detay_analizi.png", "Sağlık Analizi")
    add_img_slide("03_sagkalim_egrileri.png", "Ömür Eğrileri")
    add_img_slide("04_risk_faktorleri.png", "Risk Faktörleri (Drivers)")

    prs.save(os.path.join(OUTPUT_DIR, "PoF3_Yonetici_Sunumu_v5.pptx"))
    logger.info("   > Saved: PoF3_Yonetici_Sunumu_v5.pptx")

# --- MAIN ---
def main():
    logger = setup_logger()
    logger.info("🚀 STARTING Reporting Engine v5.0")
    for d in [VISUAL_DIR, ACTION_DIR]: os.makedirs(d, exist_ok=True)
    
    df = load_predictions(logger)
    if df is None: return
    
    generate_eda_dashboard(logger)
    generate_health_deep_dive(df, logger)
    generate_risk_drivers(logger) # NEW
    generate_interactive_map(df, logger)
    plot_survival_curves(logger)
    
    # Reports
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "PoF3_Yonetici_Raporu_v5.xlsx")) as writer:
        df.to_excel(writer, sheet_name="TUM_VERI", index=False)
        df[df['Risk_Sinifi']=='KRİTİK'].head(1000).to_excel(writer, sheet_name="ACIL_AKSIYON", index=False)
    logger.info("   > Excel Saved.")
    
    generate_pptx_report(df, logger) # FIXED CALL
    logger.info("✅ ALL TASKS COMPLETE.")

if __name__ == "__main__":
    main()