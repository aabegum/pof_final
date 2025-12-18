# -*- coding: utf-8 -*-
"""
PoF3 - Detaylı EDA ve Operasyonel Analiz Modülü
================================================
Modelden bağımsız olarak, sadece HAM ARIZA VERİSİNİ analiz eder.
Amaç: Şebekenin operasyonel karakteristiğini ve veri kalitesini anlamak.

Çıktılar:
1. Arıza Isı Haritası (Saat vs Gün) -> Ekip planlaması için.
2. Mevsimsellik Analizi (Yaz/Kış Yükü).
3. Pareto Analizi (Hangi %20'lik ekipman %80 sorunu yaratıyor?).
4. Müdahale Süresi Analizi (Ekipler ne kadar hızlı?).
5. Arıza Neden Dağılımı.

Yazar: PoF3 Team
Tarih: Aralık 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "girdiler", "ariza_final.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "gorseller", "eda")

# Görsel Ayarları
plt.style.use('ggplot')
sns.set_palette("tab10")
plt.rcParams['font.family'] = 'DejaVu Sans'

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print(f"[LOAD] Veri yükleniyor: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("❌ Dosya bulunamadı!")
        return None
    
    # Sadece gerekli sütunları oku
    df = pd.read_excel(INPUT_FILE)
    
    # Tarih formatlama (Hassas nokta)
    df['started at'] = pd.to_datetime(df['started at'], errors='coerce')
    
    # Tarih özelliklerini türet
    df['Yıl'] = df['started at'].dt.year
    df['Ay'] = df['started at'].dt.month
    df['Hafta'] = df['started at'].dt.isocalendar().week
    df['Saat'] = df['started at'].dt.hour
    df['Gun_Ismi'] = df['started at'].dt.day_name()
    
    print(f"✅ Toplam Kayıt: {len(df):,}")
    return df

# --- ANALİZ 1: ISI HARİTASI (Operasyonel Yoğunluk) ---
def plot_heatmap(df):
    print("[1/5] Isı Haritası çiziliyor...")
    plt.figure(figsize=(12, 6))
    
    # Günleri sırala
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_tr = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
    
    heatmap_data = df.groupby(['Gun_Ismi', 'Saat']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(days_order)
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=0.5, annot=False)
    
    plt.title('Arıza Yoğunluk Isı Haritası (Gün vs Saat)', fontsize=14)
    plt.xlabel('Saat (00:00 - 23:00)')
    plt.ylabel('Gün')
    plt.yticks(ticks=np.arange(7)+0.5, labels=days_tr, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_operasyonel_isi_haritasi.png"), dpi=300)
    plt.close()

# --- ANALİZ 2: MEVSİMSELLİK (Trend) ---
def plot_seasonality(df):
    print("[2/5] Mevsimsellik analizi yapılıyor...")
    plt.figure(figsize=(14, 6))
    
    # Aylık Trend
    monthly = df.groupby(df['started at'].dt.to_period('M')).size()
    monthly.index = monthly.index.astype(str)
    
    monthly.plot(kind='line', marker='o', color='steelblue', linewidth=2)
    
    plt.title('Aylık Arıza Trendi (Mevsimsellik Kontrolü)', fontsize=14)
    plt.xlabel('Dönem')
    plt.ylabel('Arıza Sayısı')
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    
    # Zirve noktaları işaretle
    max_val = monthly.max()
    max_date = monthly.idxmax()
    plt.annotate(f'Zirve: {max_date}', xy=(list(monthly.index).index(max_date), max_val), 
                 xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_mevsimsellik_trendi.png"), dpi=300)
    plt.close()

# --- ANALİZ 3: PARETO (Hangi Ekipman?) ---
def plot_pareto(df):
    print("[3/5] Pareto analizi yapılıyor...")
    plt.figure(figsize=(12, 8))
    
    # Veriyi hazırla
    col = 'Şebeke Unsuru'
    if col not in df.columns: return

    counts = df[col].value_counts()
    cumulative = counts.cumsum() / counts.sum() * 100
    
    # Bar Chart (Sol Eksen)
    ax1 = plt.gca()
    counts.head(15).plot(kind='bar', color='coral', ax=ax1)
    ax1.set_ylabel('Arıza Sayısı', color='coral')
    ax1.tick_params(axis='y', labelcolor='coral')
    
    # Line Chart (Sağ Eksen - Kümülatif %)
    ax2 = ax1.twinx()
    ax2.plot(cumulative.head(15).values, color='blue', marker='D', ms=5)
    ax2.set_ylabel('Kümülatif %', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color='grey', linestyle='--', label='%80 Eşiği')
    
    plt.title('Pareto Analizi: Arızaların %80\'i hangi ekipmanlardan geliyor?', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_pareto_analizi.png"), dpi=300)
    plt.close()

# --- ANALİZ 4: SİGORTA vs GERÇEK ARIZA ---
def plot_cause_breakdown(df):
    print("[4/5] Neden analizi yapılıyor...")
    plt.figure(figsize=(10, 6))
    
    if 'cause code' not in df.columns: return
    
    # Kategorize Et
    df['Tip'] = df['cause code'].apply(lambda x: 'Sigorta Atığı (Geçici)' if 'Sigorta' in str(x) else 'Donanım Arızası (Kalıcı)')
    
    counts = df['Tip'].value_counts()
    
    # Pasta Grafik
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', 
            colors=['#2ca02c', '#d62728'], startangle=90, explode=(0.05, 0))
    
    plt.title('Arıza Karakteristiği: Kalıcı vs Geçici', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_ariza_tipi_dagilimi.png"), dpi=300)
    plt.close()

# --- ANALİZ 5: MÜDAHALE SÜRELERİ (Duration) ---
def plot_durations(df):
    print("[5/5] Süre analizi yapılıyor...")
    # duration sütununu bul veya hesapla
    
    cols = [c for c in df.columns if 'süre' in c.lower() or 'duration' in c.lower()]
    target_col = None
    
    # Eğer hazır sütun yoksa hesapla
    if not cols and 'ended at' in df.columns:
        df['ended at'] = pd.to_datetime(df['ended at'], errors='coerce')
        df['Duration_Min'] = (df['ended at'] - df['started at']).dt.total_seconds() / 60
        target_col = 'Duration_Min'
    elif cols:
        target_col = cols[0]
        
    if not target_col: return

    plt.figure(figsize=(12, 6))
    
    # Aykırı değerleri temizle (Max 24 saat = 1440 dk)
    clean_data = df[df[target_col] < 1440][target_col]
    
    sns.histplot(clean_data, bins=50, kde=True, color='purple')
    
    med = clean_data.median()
    plt.axvline(med, color='red', linestyle='--', label=f'Medyan: {med:.0f} dk')
    
    plt.title('Arıza Müdahale/Kesinti Süresi Dağılımı', fontsize=14)
    plt.xlabel('Dakika')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_mudahale_suresi.png"), dpi=300)
    plt.close()

# --- MAIN ---
def main():
    ensure_dirs()
    df = load_data()
    
    if df is None: return
    
    plot_heatmap(df)
    plot_seasonality(df)
    plot_pareto(df)
    plot_cause_breakdown(df)
    plot_durations(df)
    
    print("\n✅ EDA TAMAMLANDI.")
    print(f"📂 Görseller: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()