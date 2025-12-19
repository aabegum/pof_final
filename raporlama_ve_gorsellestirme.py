# -*- coding: utf-8 -*-
"""
05_raporlama_ve_gorsellestirme.py (PoF3 - Ultimate Reporting Engine v3.3)
FIXES:
1. Auto-calculates 'PoF_Ensemble_12Ay' if missing.
2. Robust merging logic for Case Studies.
3. Maps 'Risk_Sinifi' to 'Risk_Class' automatically.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# PPTX Library Check
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    print("Warning: 'python-pptx' not installed. Skipping PPTX generation.")

# --- KONFİGÜRASYON VE YOLLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "sonuclar")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "ara_ciktilar")
LOG_DIR = os.path.join(BASE_DIR, "loglar")

# Klasörlerin varlığından emin ol
for d in [OUTPUT_DIR, INTERMEDIATE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# Alt Klasörler
ACTION_DIR = os.path.join(OUTPUT_DIR, "aksiyon_listeleri")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "gorseller")
REPORT_DIR = OUTPUT_DIR

for d in [ACTION_DIR, VISUAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Görsel Ayarları
plt.style.use('ggplot')
sns.set_palette("husl")

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("Raporlama")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"raporlama_{ts}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

# ------------------------------------------------------------------------------
# HELPER: ENSURE POF COLUMN EXISTS
# ------------------------------------------------------------------------------
def ensure_pof_column(df, logger):
    """
    'PoF_Ensemble_12Ay' kolonunu kontrol eder, yoksa hesaplar.
    """
    target = 'PoF_Ensemble_12Ay'
    
    if target in df.columns:
        return df
    
    logger.warning(f"  [FIX] '{target}' eksik. Bileşenlerden hesaplanmaya çalışılıyor...")
    
    # Olası PoF kolonlarını bul
    candidates = [c for c in df.columns if ('_pof_12' in c.lower()) or ('_12ay' in c.lower() and 'pof' in c.lower())]
    candidates = [c for c in candidates if c != target]
    
    if candidates:
        logger.info(f"  [FIX] Bulunan bileşenler: {candidates}")
        df[target] = df[candidates].mean(axis=1)
        logger.info(f"  [FIX] '{target}' kolonu {len(candidates)} modelin ortalaması ile oluşturuldu.")
    else:
        logger.warning("  [FAIL] 12 aylık PoF bileşeni bulunamadı. Dummy (0.0) oluşturuluyor.")
        df[target] = 0.0
        
    return df

# ------------------------------------------------------------------------------
# PHASE 1: ACTION PLANNING
# ------------------------------------------------------------------------------
def generate_action_lists(df, logger):
    logger.info("="*60)
    logger.info("[PHASE 1] Aksiyon Listeleri Oluşturuluyor...")
    
    # Kolon eşleştirme (Risk_Class yoksa Risk_Sinifi kullan)
    risk_col = 'Risk_Class' if 'Risk_Class' in df.columns else 'Risk_Sinifi'
    
    # 1. ACİL MÜDAHALE (Kritik + Kronik)
    if 'Kronik_Flag' in df.columns:
        crit_chronic = df[
            (df[risk_col].isin(['Critical', 'KRİTİK'])) & 
            (df['Kronik_Flag'] == 1)
        ].copy()
    else:
        crit_chronic = pd.DataFrame()
    
    if not crit_chronic.empty:
        path = os.path.join(ACTION_DIR, "01_acil_mudahale_listesi.csv")
        crit_chronic.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [ACİL] Kronik & Kritik: {len(crit_chronic)} varlık")

    # 2. YÜKSEK RİSKLİ TRAFOLAR (CAPEX)
    trafos = df[
        (df['Ekipman_Tipi'].str.contains('Trafo', na=False)) & 
        (df[risk_col].isin(['Critical', 'High', 'KRİTİK', 'YÜKSEK']))
    ].copy()
    
    if not trafos.empty:
        path = os.path.join(ACTION_DIR, "02_yuksek_riskli_trafolar_capex.csv")
        trafos.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [CAPEX] Yüksek Riskli Trafolar: {len(trafos)} varlık")

    # 3. İŞLETME KONTROL (Yüksek Olasılık ama Düşük Etki olabilir)
    if 'PoF_Ensemble_12Ay' in df.columns:
        inspection = df[
            (df['PoF_Ensemble_12Ay'] > 0.10) & 
            (df[risk_col].isin(['Low', 'Medium', 'DÜŞÜK', 'ORTA']))
        ].copy()
        
        if not inspection.empty:
            path = os.path.join(ACTION_DIR, "03_bakim_rotasi_kontrol.csv")
            inspection.sort_values('PoF_Ensemble_12Ay', ascending=False).to_csv(path, index=False, encoding='utf-8-sig')
            logger.info(f"  > [OPEX] Yüksek Olasılık/Düşük Risk Sınıfı: {len(inspection)} varlık")

    return crit_chronic

# ------------------------------------------------------------------------------
# PHASE 2: VISUALIZATION
# ------------------------------------------------------------------------------
def plot_single_chart(df, col_x, col_y, plot_type, title, filename, logger, **kwargs):
    width = kwargs.pop('width', 10)
    height = kwargs.pop('height', 6)
    
    plt.figure(figsize=(width, height))
    
    try:
        if plot_type == 'scatter':
            sns.scatterplot(data=df, x=col_x, y=col_y, **kwargs)
        elif plot_type == 'hist':
            sns.histplot(df[col_x], kde=True, **kwargs)
        elif plot_type == 'bar':
            sns.barplot(x=col_x, y=col_y, data=df, **kwargs)

        plt.title(title, fontsize=14)
        plt.tight_layout()
        path = os.path.join(VISUAL_DIR, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  > Kaydedildi: {filename}")
        return path
    except Exception as e:
        logger.error(f"  [ERROR] Grafik çizilemedi {filename}: {str(e)}")
        plt.close()
        return None

def generate_visuals(df, logger):
    logger.info("="*60)
    logger.info("[PHASE 2] Görsel Panolar Oluşturuluyor...")
    charts = {}
    
    # Risk Kolonunu Belirle
    risk_col = 'Risk_Class' if 'Risk_Class' in df.columns else 'Risk_Sinifi'

    # 1. SAĞLIK SKORU DAĞILIMI
    if 'Health_Score' in df.columns:
        path = plot_single_chart(df, 'Health_Score', None, 'hist', 
                                 'Varlık Sağlık Skoru Dağılımı', "02_saglik_skoru_dagilimi.png", logger,
                                 bins=30, color='teal', edgecolor='black')
        charts['health_dist'] = path

    # 2. KRONİK ANALİZİ
    if 'Chronic_Flag' in df.columns:
        counts = df['Chronic_Flag'].value_counts()
        plt.figure(figsize=(8, 6))
        counts.plot(kind='bar', color=['green', 'red'], edgecolor='black')
        plt.title('Kronik Varlık Dağılımı (0=Normal, 1=Kronik)', fontsize=14)
        plt.xticks(rotation=0)
        path = os.path.join(VISUAL_DIR, "03_kronik_dagilimi.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['chronic_dist'] = path

    # 3. COĞRAFİ HARİTA (Varsa)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        gdf = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()
        if not gdf.empty:
                palette_map = {
                    'Critical': 'red', 'High': 'orange', 'Medium': 'gold', 'Low': 'green',
                    'KRİTİK': 'red', 'YÜKSEK': 'orange', 'ORTA': 'gold', 'DÜŞÜK': 'green'
                }
                # Bilinmeyenleri gri yap
                for label in gdf[risk_col].unique():
                    if label not in palette_map: palette_map[label] = 'gray'

                path = plot_single_chart(gdf, 'Longitude', 'Latitude', 'scatter', 
                                        'Coğrafi Risk Haritası', "04_cografi_risk_haritasi.png", logger,
                                        hue=risk_col, height=10, width=10,
                                        palette=palette_map, s=30, alpha=0.8)
                charts['geo_map'] = path

    return charts

def validate_base_rates(df, logger):
    logger.info("="*60)
    logger.info("[VALIDATION] Model Kalibrasyon Kontrolü (Sektör Ortalamaları ile)...")
    
    # Sektör beklentileri (yıllık arıza oranı)
    INDUSTRY_RANGES = {
        'Trafo': (0.005, 0.05), 'Kesici': (0.01, 0.08), 'Ayırıcı': (0.02, 0.12),
        'Sigorta': (0.10, 0.40), 'Hat': (0.005, 0.15), 'Direk': (0.001, 0.03)
    }
    
    if 'PoF_Ensemble_12Ay' not in df.columns:
        logger.warning("  [SKIP] PoF kolonu yok. Validasyon yapılamıyor.")
        return

    stats = df.groupby('Ekipman_Tipi')['PoF_Ensemble_12Ay'].mean().reset_index()
    stats.columns = ['Type', 'Predicted_Rate']
    
    for _, row in stats.iterrows():
        etype = row['Type']
        pred = row['Predicted_Rate']
        # Eşleşen anahtar kelime bul
        matched_key = next((k for k in INDUSTRY_RANGES if k in str(etype)), None)
        
        if matched_key:
            low, high = INDUSTRY_RANGES[matched_key]
            status = "✅ OK"
            if pred < low: status = "📉 DÜŞÜK"
            if pred > high: status = "🚨 YÜKSEK"
            logger.info(f"  > {str(etype).ljust(20)}: {pred:.1%} (Hedef: {low:.0%} - {high:.0%}) -> {status}")

def plot_aggregate_risk_by_type(df, logger):
    if 'PoF_Ensemble_12Ay' not in df.columns:
        return None

    # Agregasyon
    agg_df = df.groupby('Ekipman_Tipi').agg(
        Mean_PoF_1Y=('PoF_Ensemble_12Ay', 'mean'),
        Count=('cbs_id', 'count')
    ).reset_index()
    
    # Eğer Chronic_Flag varsa onu da ekle
    if 'Chronic_Flag' in df.columns:
        chronic_agg = df.groupby('Ekipman_Tipi')['Chronic_Flag'].mean().reset_index()
        agg_df = agg_df.merge(chronic_agg, on='Ekipman_Tipi')
        
    agg_df = agg_df[agg_df['Count'] >= 30].sort_values('Mean_PoF_1Y', ascending=False).head(10)

    if agg_df.empty: return None

    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Ekipman_Tipi', y='Mean_PoF_1Y', data=agg_df, ax=ax1, color='darkred', alpha=0.7)
    ax1.set_ylabel('Ortalama PoF (1 Yıl)', color='darkred', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.set_xlabel('Ekipman Tipi', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    if 'Chronic_Flag' in agg_df.columns:
        ax2 = ax1.twinx()
        sns.lineplot(x='Ekipman_Tipi', y='Chronic_Flag', data=agg_df, ax=ax2, color='darkgreen', marker='o', linewidth=3)
        ax2.set_ylabel('Ortalama Kronik Oranı', color='darkgreen', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='darkgreen')
    
    plt.title('Ekipman Tipine Göre Risk Yoğunluğu (Top 10)', fontsize=14)
    path = os.path.join(VISUAL_DIR, "08_aggregate_risk_by_type.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  > Kaydedildi: 08_aggregate_risk_by_type.png")
    return path

# ------------------------------------------------------------------------------
# PHASE 3: EXCEL REPORTING
# ------------------------------------------------------------------------------
def create_excel_report(df, crit_chronic, case_studies, logger): 
    logger.info("="*60)
    logger.info("[PHASE 3] Excel Raporu Oluşturuluyor...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(REPORT_DIR, f"PoF3_Analiz_Raporu_Final.xlsx")
    
    risk_col = 'Risk_Class' if 'Risk_Class' in df.columns else 'Risk_Sinifi'
    
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        total = len(df)
        crit_count = (df[risk_col].isin(['Critical', 'KRİTİK'])).sum() if risk_col in df.columns else 0
        avg_health = df['Health_Score'].mean() if 'Health_Score' in df.columns else 0
        
        summary = pd.DataFrame({
            'KPI': ['Toplam Varlık', 'Kritik Riskli', 'Kronik ve Kritik', 'Ortalama Sağlık', 'Rapor Tarihi'],
            'Değer': [total, crit_count, len(crit_chronic), f"{avg_health:.1f}", timestamp]
        })
        summary.to_excel(writer, sheet_name='Yonetici_Ozeti', index=False)
        
        if not crit_chronic.empty:
            crit_chronic.to_excel(writer, sheet_name='Acil_Mudahale', index=False)

        if not case_studies.empty:
            case_studies.to_excel(writer, sheet_name='Vaka_Analizi_CaseStudy', index=False)
            
        # Top 1000 Riskli
        sort_col = 'PoF_Ensemble_12Ay' if 'PoF_Ensemble_12Ay' in df.columns else df.columns[0]
        df.sort_values(sort_col, ascending=False).head(1000).to_excel(writer, sheet_name='Risk_Master_Top1000', index=False)
            
    logger.info(f"  > Kaydedildi: {os.path.basename(out_path)}")
    
def generate_case_studies(df_risk, logger):
    logger.info("[PHASE 1.5] Vaka Analizleri (Case Studies) Oluşturuluyor...")
    
    # Arıza olayları dosyasını bul (intermediate_paths)
    events_path = os.path.join(INTERMEDIATE_DIR, "fault_events_clean.csv")
    
    if not os.path.exists(events_path):
        # Eğer ara çıktı yoksa, ana girdi dosyasını kullanmayı dene (Fallback)
        raw_path = os.path.join(BASE_DIR, "data", "girdiler", "ariza_final.xlsx")
        if os.path.exists(raw_path):
            events = pd.read_excel(raw_path)
            # Kolon isimlerini uyarla
            if 'started at' in events.columns: events['Ariza_Baslangic_Zamani'] = events['started at']
            if 'cbs_id' not in events.columns and 'Ekipman Kodu' in events.columns: events['cbs_id'] = events['Ekipman Kodu']
        else:
            return pd.DataFrame()
    else:
        events = pd.read_csv(events_path)

    # Tarih parse et
    if 'Ariza_Baslangic_Zamani' in events.columns:
        events['Ariza_Baslangic_Zamani'] = pd.to_datetime(events['Ariza_Baslangic_Zamani'], errors='coerce')
    else:
        return pd.DataFrame()

    events['cbs_id'] = events['cbs_id'].astype(str).str.lower().str.strip()
    
    # Son 6 aydaki arızaları al
    analysis_date = events['Ariza_Baslangic_Zamani'].max()
    if pd.isna(analysis_date): return pd.DataFrame()
    
    cutoff_date = analysis_date - timedelta(days=180)
    recent_faults = events[events['Ariza_Baslangic_Zamani'] >= cutoff_date].copy()
    
    if recent_faults.empty:
        return pd.DataFrame()

    # Risk verisiyle birleştir
    df_risk = ensure_pof_column(df_risk, logger)
    
    risk_col = 'Risk_Class' if 'Risk_Class' in df_risk.columns else 'Risk_Sinifi'
    cols_to_merge = ['cbs_id', risk_col, 'PoF_Ensemble_12Ay']
    
    # Varsa ekle
    for c in ['Health_Score', 'Ekipman_Tipi', 'Ilce']:
        if c in df_risk.columns: cols_to_merge.append(c)

    case_df = recent_faults.merge(
        df_risk[cols_to_merge], 
        on='cbs_id', 
        how='left'
    )
    
    # Değerlendirme
    def judge_prediction(row):
        if risk_col not in row or pd.isna(row[risk_col]): return "Bilinmeyen Varlık"
        r = row[risk_col]
        if r in ['Critical', 'High', 'KRİTİK', 'YÜKSEK']:
            return "BAŞARILI (Öngörüldü)"
        elif r in ['Medium', 'ORTA']:
            return "KISMİ (İzleme)"
        else:
            return "KAÇIRILDI (Düşük Risk)"

    case_df['Model_Karari'] = case_df.apply(judge_prediction, axis=1)
    
    # En yüksek başarılı ve en kötü kaçırılanları seç
    successes = case_df[case_df['Model_Karari'] == "BAŞARILI (Öngörüldü)"].head(10)
    misses = case_df[case_df['Model_Karari'] == "KAÇIRILDI (Düşük Risk)"].head(5)
    
    final_cases = pd.concat([successes, misses])
    
    logger.info(f"  > {len(final_cases)} adet vaka analizi oluşturuldu.")
    return final_cases

# ------------------------------------------------------------------------------
# PHASE 4: POWERPOINT PRESENTATION
# ------------------------------------------------------------------------------
def create_pptx_presentation(df, charts, logger):
    if not HAS_PPTX:
        return

    logger.info("="*60)
    logger.info("[PHASE 4] PowerPoint Sunumu Oluşturuluyor...")
    
    prs = Presentation()
    timestamp = datetime.now().strftime("%d %B %Y")
    
    # Slide 1: Başlık
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "PoF3 Risk ve Sağlık Analizi"
    slide.placeholders[1].text = f"Yönetici Özeti Raporu\n{timestamp}"
    
    # Slide 2: Özet
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Genel Durum Özeti"
    
    total = len(df)
    risk_col = 'Risk_Class' if 'Risk_Class' in df.columns else 'Risk_Sinifi'
    crit = (df[risk_col].isin(['Critical', 'KRİTİK'])).sum() if risk_col in df.columns else 0
    chronic = (df['Chronic_Flag'] == 1).sum() if 'Chronic_Flag' in df.columns else 0
    
    content = f"""
    Toplam Varlık Sayısı: {total:,}
    Kritik Riskli Varlıklar: {crit:,}
    Kronik Sorunlu Varlıklar: {chronic:,}
    
    Veri Seti: Arıza Bakım Yönetim Sistemi
    Analiz Tarihi: {timestamp}
    """
    slide.placeholders[1].text = content

    # Grafikler
    chart_slides = {
        'health_dist': "Varlık Sağlık Dağılımı",
        'chronic_dist': "Kronik Varlık Analizi",
        'aggregate_risk': "Ekipman Tipine Göre Risk",
        'geo_map': "Coğrafi Risk Haritası"
    }
    
    for key, title in chart_slides.items():
        if key in charts and os.path.exists(charts[key]):
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            slide.shapes.title.text = title
            slide.shapes.add_picture(charts[key], Inches(1), Inches(1.5), height=Inches(5.5))
            
    out_path = os.path.join(OUTPUT_DIR, "PoF3_Yonetici_Sunumu_Final.pptx")
    prs.save(out_path)
    logger.info(f"  > Kaydedildi: {os.path.basename(out_path)}")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    logger = setup_logger()
    logger.info("🚀 PoF3 Raporlama Motoru Başlatılıyor...")
    
    # 1. Dosya Kontrolü ve Yükleme
    risk_path = os.path.join(OUTPUT_DIR, "pof_predictions_final.csv")
    
    if not os.path.exists(risk_path):
        # Fallback: Eski isimle dene
        alt_path = os.path.join(OUTPUT_DIR, "risk_equipment_master.csv")
        if os.path.exists(alt_path):
            risk_path = alt_path
        else:
            logger.error(f"[FATAL] Sonuç dosyası bulunamadı: {risk_path}")
            logger.error("Lütfen önce 'pof.py' (Step 04) çalıştırın.")
            return
        
    df = pd.read_csv(risk_path)
    
    # 2. Kolon Eşleştirme (Risk_Sinifi -> Risk_Class)
    if 'Risk_Sinifi' in df.columns and 'Risk_Class' not in df.columns:
        logger.info("[MAPPING] 'Risk_Sinifi' -> 'Risk_Class' eşleştirmesi yapılıyor.")
        df['Risk_Class'] = df['Risk_Sinifi']
    elif 'Risk_Class' not in df.columns:
        logger.warning("[WARN] Risk kolonu yok. Varsayılan 'Low' atanıyor.")
        df['Risk_Class'] = 'Low'

    # 3. PoF Kolonunu Garantiye Al
    df = ensure_pof_column(df, logger)
    
    # 4. Master Veri ile Zenginleştirme (Lokasyon vb.)
    master_path = os.path.join(INTERMEDIATE_DIR, "equipment_master.csv")
    if os.path.exists(master_path):
        meta = pd.read_csv(master_path)
        # ID normalizasyonu
        meta['cbs_id'] = meta['cbs_id'].astype(str).str.lower().str.strip()
        df['cbs_id'] = df['cbs_id'].astype(str).str.lower().str.strip()
        
        desired = ['Latitude', 'Longitude', 'Musteri_Sayisi', 'Ilce', 'Sehir', 'Mahalle', 'Ekipman_Tipi']
        add = [c for c in desired if c in meta.columns and c not in df.columns]
        
        if add:
            logger.info(f"[MERGE] Ek bağlam kolonları ekleniyor: {add}")
            df = df.merge(meta[['cbs_id'] + add], on='cbs_id', how='left')
    
    # Eksik metin verilerini doldur
    for col in ['Ilce', 'Sehir', 'Ekipman_Tipi']:
        if col not in df.columns: df[col] = 'Unknown'
        else: df[col] = df[col].fillna('Unknown')

    logger.info(f"[LOAD] Raporlanacak Varlık Sayısı: {len(df):,}")
    validate_base_rates(df, logger)
    
    # 5. Raporları Üret
    crit_chronic = generate_action_lists(df, logger)
    case_studies = generate_case_studies(df, logger)
    charts = generate_visuals(df, logger)
    
    agg_path = plot_aggregate_risk_by_type(df, logger)
    if agg_path: charts['aggregate_risk'] = agg_path 
    
    create_excel_report(df, crit_chronic, case_studies, logger)
    create_pptx_presentation(df, charts, logger)
    
    logger.info("")
    logger.info("[SUCCESS] Raporlama ve Görselleştirme Tamamlandı.")

if __name__ == "__main__":
    main()