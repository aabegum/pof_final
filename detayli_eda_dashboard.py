# -*- coding: utf-8 -*-
"""
PoF3 - Detaylı EDA ve Operasyonel Analiz Modülü
================================================
GÜNCELLENMİŞ VERSİYON:
1. Data Cleaning (Filtreleme) kaldırıldı (Ana pipeline'da yapıldığı için).
2. Tarih formatı uyarısı (dayfirst=True) düzeltildi.
3. Grafik çökme hatası (explode length mismatch) giderildi.
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
# Ham veriyi analiz ediyoruz
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
    
    df = pd.read_excel(INPUT_FILE)
    
    # --- DÜZELTME 1: Tarih Formatı Uyarısı ---
    # dayfirst=True ekleyerek uyarıyı susturuyoruz
    df['started at'] = pd.to_datetime(df['started at'], errors='coerce', dayfirst=True)
    
    # --- TEMİZLİK YOK (İSTEĞİNİZ ÜZERİNE KALDIRILDI) ---
    # Ham veri olduğu gibi analiz edilecek.
    print(f"📊 Ham Veri Analiz Ediliyor: {len(df):,} kayıt")

    # Tarih özelliklerini türet
    df['Yıl'] = df['started at'].dt.year
    df['Ay'] = df['started at'].dt.month
    df['Hafta'] = df['started at'].dt.isocalendar().week
    df['Saat'] = df['started at'].dt.hour
    df['Gun_Ismi'] = df['started at'].dt.day_name()
    
    return df

# --- ANALİZ 1: ISI HARİTASI ---
def plot_heatmap(df):
    print("[1/5] Isı Haritası çiziliyor...")
    if df.empty: return
    plt.figure(figsize=(12, 6))
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_tr = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
    
    heatmap_data = df.groupby(['Gun_Ismi', 'Saat']).size().unstack(fill_value=0)
    # Veride olmayan günleri de kapsayacak şekilde reindex
    heatmap_data = heatmap_data.reindex(days_order)
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=0.5, annot=False)
    
    plt.title('Arıza Yoğunluk Haritası (Ham Veri)', fontsize=14)
    plt.xlabel('Saat')
    plt.ylabel('Gün')
    plt.yticks(ticks=np.arange(7)+0.5, labels=days_tr, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_operasyonel_isi_haritasi.png"), dpi=300)
    plt.close()

# --- ANALİZ 2: MEVSİMSELLİK ---
def plot_seasonality(df):
    print("[2/5] Mevsimsellik analizi yapılıyor...")
    if df.empty: return
    plt.figure(figsize=(14, 6))
    
    monthly = df.groupby(df['started at'].dt.to_period('M')).size()
    monthly.index = monthly.index.astype(str)
    
    monthly.plot(kind='line', marker='o', color='steelblue', linewidth=2)
    
    plt.title('Aylık Arıza Trendi', fontsize=14)
    plt.xlabel('Dönem')
    plt.ylabel('Arıza Sayısı')
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_mevsimsellik_trendi.png"), dpi=300)
    plt.close()

# --- ANALİZ 3: PARETO ---
def plot_pareto(df):
    print("[3/5] Pareto analizi yapılıyor...")
    if df.empty: return
    plt.figure(figsize=(12, 8))
    
    col = 'Şebeke Unsuru'
    if col not in df.columns: return

    counts = df[col].value_counts()
    cumulative = counts.cumsum() / counts.sum() * 100
    
    ax1 = plt.gca()
    counts.head(15).plot(kind='bar', color='coral', ax=ax1)
    ax1.set_ylabel('Arıza Sayısı', color='coral')
    
    ax2 = ax1.twinx()
    ax2.plot(cumulative.head(15).values, color='blue', marker='D', ms=5)
    ax2.set_ylabel('Kümülatif %', color='blue')
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color='grey', linestyle='--', label='%80 Eşiği')
    
    plt.title('Pareto Analizi: En Çok Arızalanan Ekipmanlar', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_pareto_analizi.png"), dpi=300)
    plt.close()

# --- ANALİZ 4: ARIZA TİPİ (DÜZELTİLDİ) ---
def plot_cause_breakdown(df):
    print("[4/5] Neden analizi yapılıyor...")
    if df.empty: return
    plt.figure(figsize=(10, 6))
    
    col = 'cause code' if 'cause code' in df.columns else 'Ariza_Nedeni'
    if col not in df.columns: return
    
    # Ham veride "Sigorta" vs "Diğer" ayrımı yapıyoruz
    df['Tip'] = df[col].astype(str).apply(lambda x: 'Sigorta Atığı (Geçici)' if 'Sigorta' in x or 'SİGORTA' in x else 'Donanım Arızası (Kalıcı)')
    
    counts = df['Tip'].value_counts()
    
    # --- DÜZELTME 2: Dinamik Explode ---
    # Eğer tek kategori varsa explode patlamamalı
    explode = [0.05] * len(counts) 
    
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', 
            colors=['#2ca02c', '#d62728'], startangle=90, explode=explode)
    
    plt.title('Arıza Karakteristiği (Ham Veri)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_ariza_tipi_dagilimi.png"), dpi=300)
    plt.close()

# --- ANALİZ 5: MÜDAHALE SÜRELERİ ---
def plot_durations(df):
    print("[5/5] Süre analizi yapılıyor...")
    if df.empty: return
    
    cols = [c for c in df.columns if 'süre' in c.lower() or 'duration' in c.lower()]
    target_col = None
    
    if not cols and 'ended at' in df.columns:
        df['ended at'] = pd.to_datetime(df['ended at'], errors='coerce')
        df['Duration_Min'] = (df['ended at'] - df['started at']).dt.total_seconds() / 60
        target_col = 'Duration_Min'
    elif cols:
        target_col = cols[0]
        
    if not target_col: return

    plt.figure(figsize=(12, 6))
    # 24 saatten uzun sürenleri (outlier) görselden çıkaralım ki histogram bozulmasın
    clean_data = df[df[target_col] < 1440][target_col] 
    
    sns.histplot(clean_data, bins=50, kde=True, color='purple')
    if not clean_data.empty:
        plt.axvline(clean_data.median(), color='red', linestyle='--', label=f'Medyan: {clean_data.median():.0f} dk')
    
    plt.title('Arıza Müdahale Süresi Dağılımı', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_mudahale_suresi.png"), dpi=300)
    plt.close()

# --- MAIN ---
def main():
    ensure_dirs()
    df = load_data()
    
    if df is None: return
    
    try:
        plot_heatmap(df)
        plot_seasonality(df)
        plot_pareto(df)
        plot_cause_breakdown(df)
        plot_durations(df)
        
        print("\n✅ EDA TAMAMLANDI.")
        print(f"📂 Görseller: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()