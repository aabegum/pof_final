import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime

# =============================================================================
# 1. AYARLAR
# =============================================================================
st.set_page_config(
    page_title="Varlık Analitiği & Operasyon Paneli (v7.0)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Renk Paleti
COLORS = {
    'primary': '#0052cc', 'secondary': '#172B4D',
    'danger': '#FF5630', 'warning': '#FFAB00', 'success': '#36B37E',
    'info': '#00B8D9', 'urban': '#6366f1', 'rural': '#10b981'
}

# Dosya Yolları
BASE_DIR = "data"
INPUT_DIR = os.path.join(BASE_DIR, "girdiler")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "ara_ciktilar")
OUTPUT_DIR = os.path.join(BASE_DIR, "sonuclar")

# Sütun Eşleştirme
COLUMN_MAP = {
    "Tref_Yas_Gun": "Yas_Gun",
    "Fault_Count": "Toplam_Ariza",
    "started at": "Ariza_Baslangic_Zamani",
    # Koordinat
    "X_KOORDINAT": "Boylam", "Y_KOORDINAT": "Enlem",
    "Longitude": "Boylam", "Latitude": "Enlem",
    "x_koordinat": "Boylam", "y_koordinat": "Enlem",
    # Bakım
    "Bakım Sayısı": "Bakim_Sayisi", "Bakim_Sayisi": "Bakim_Sayisi",
    "Son Bakım İş Emri Tarihi": "Son_Bakim_Tarihi",
    "Son Bakımdan İtibaren Geçen Gün Sayısı": "Son_Bakim_Gecen_Gun",
    "İlk Bakım İş Emri Tarihi": "Ilk_Bakim_Tarihi",
}

CUSTOMER_COLS = [
    "urban mv+suburban mv", "urban lv+suburban lv", 
    "urban mv", "urban lv", "suburban mv", "suburban lv", 
    "rural mv", "rural lv", "total customer count"
]

# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
# =============================================================================

def safe_get(row, key, default=0):
    val = row.get(key, default)
    if isinstance(val, pd.Series): return val.iloc[0]
    return val

def clean_coordinates(df, lat_col='Enlem', lon_col='Boylam'):
    """Koordinatları temizler (Virgül -> Nokta dönüşümü ve Numeric zorlama)"""
    for col in [lat_col, lon_col]:
        if col in df.columns:
            # String'e çevir, virgülleri nokta yap
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            # Sayıya çevir, hataları NaN yap
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 0'ları NaN yap
            df[col] = df[col].replace(0, np.nan)
    return df

def calculate_chronic_flags(df_main, df_faults):
    """90/180/365 Günlük Kronik Hesaplama"""
    if df_faults is None or df_faults.empty or 'Ariza_Baslangic_Zamani' not in df_faults.columns:
        return df_main
    
    analysis_date = df_faults['Ariza_Baslangic_Zamani'].max()
    chronic_res = pd.DataFrame({'cbs_id': df_main['cbs_id'].unique()})
    
    for days, thresh in [(90, 2), (180, 3), (365, 4)]:
        start = analysis_date - pd.Timedelta(days=days)
        mask = df_faults['Ariza_Baslangic_Zamani'] >= start
        counts = df_faults[mask].groupby('cbs_id').size().reset_index(name=f'Kronik_{days}g_Sayi')
        counts[f'Kronik_{days}g'] = (counts[f'Kronik_{days}g_Sayi'] >= thresh).astype(int)
        chronic_res = chronic_res.merge(counts, on='cbs_id', how='left').fillna(0)
    
    cols_to_drop = [c for c in chronic_res.columns if c in df_main.columns and c != 'cbs_id']
    if cols_to_drop: df_main = df_main.drop(columns=cols_to_drop)
        
    return df_main.merge(chronic_res, on='cbs_id', how='left')

def get_region_type(row):
    try:
        total = safe_get(row, 'total customer count')
        if pd.isna(total) or total == 0:
            urbans = ['MERKEZ', 'SALİHLİ', 'ALAŞEHİR', 'TURGUTLU', 'AKHİSAR', 'YUNUSEMRE', 'ŞEHZADELER']
            ilce = str(safe_get(row, 'Ilce', '')).upper()
            return 'Kentsel' if any(u in ilce for u in urbans) else 'Kırsal'
        
        u_load = safe_get(row, 'urban mv') + safe_get(row, 'urban lv') + safe_get(row, 'suburban mv') + safe_get(row, 'suburban lv')
        if safe_get(row, 'urban mv+suburban mv') > 0:
            u_load = safe_get(row, 'urban mv+suburban mv') + safe_get(row, 'urban lv+suburban lv')
            
        return 'Kentsel' if (u_load / total) > 0.6 else 'Kırsal'
    except: return "Bilinmiyor"

def extract_voltage_class(eq_type, row):
    try:
        if safe_get(row, 'urban mv') > 0 or safe_get(row, 'rural mv') > 0: return 'OG (Orta Gerilim)'
        name = str(eq_type).upper()
        if any(x in name for x in ['OG', '34.5', 'TRAFO', 'HÜCRE']): return 'OG (Orta Gerilim)'
        if any(x in name for x in ['AG', 'PANO', 'BOX', '0.4']): return 'AG (Alçak Gerilim)'
        return 'Diğer'
    except: return 'Diğer'

# =============================================================================
# 3. VERİ YÜKLEME (Fallback Mekanizmalı)
# =============================================================================
@st.cache_data
def load_and_process_data():
    data = {}
    
    # --- 1. ANA VERİ ---
    path = os.path.join(INTERMEDIATE_DIR, "model_input_data_full.csv")
    if not os.path.exists(path):
        path = os.path.join(INTERMEDIATE_DIR, "ozellikler_pof.csv")
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Eksik Verileri Tamamla (Koordinat/Bakım)
        needed = ["X_KOORDINAT", "Y_KOORDINAT", "Bakım Sayısı", "Son Bakımdan İtibaren Geçen Gün Sayısı", "total customer count"]
        missing = [c for c in needed if c not in df.columns and COLUMN_MAP.get(c) not in df.columns]
        
        if missing:
            raw_dfs = []
            for f in ["ariza_final.xlsx", "saglam_final.xlsx"]:
                p = os.path.join(INPUT_DIR, f)
                if os.path.exists(p):
                    try:
                        tmp = pd.read_excel(p)
                        if 'ID' in tmp.columns: tmp.rename(columns={'ID':'cbs_id'}, inplace=True)
                        cols = [c for c in missing if c in tmp.columns]
                        if cols: raw_dfs.append(tmp[['cbs_id'] + cols])
                    except: pass
            if raw_dfs:
                full_raw = pd.concat(raw_dfs).drop_duplicates('cbs_id')
                df = df.merge(full_raw, on='cbs_id', how='left')

        df.rename(columns=COLUMN_MAP, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]

        # Tarih & Format
        date_cols = [c for c in df.columns if 'Tarih' in c or 'Zaman' in c]
        for col in date_cols: df[col] = pd.to_datetime(df[col], errors='coerce')
        
        for c in CUSTOMER_COLS: 
            if c in df.columns: df[c] = df[c].fillna(0)
            
        if 'Yas_Gun' in df.columns: df['Yas_Yil'] = df['Yas_Gun'] / 365.25
        elif 'ekipman_yasi_gun' in df.columns: df['Yas_Yil'] = df['ekipman_yasi_gun'] / 365.25
        
        # Bakım Durumu
        if 'Bakim_Sayisi' in df.columns:
            df['Bakim_Durumu'] = df['Bakim_Sayisi'].apply(lambda x: 'Bakımlı' if x>0 else ('Hiç Bakılmadı' if x==0 else 'Veri Yok'))
        else: df['Bakim_Durumu'] = 'Veri Yok'
        
        # Özellikler
        df['Bolge_Tipi'] = df.apply(get_region_type, axis=1)
        df['Gerilim_Seviyesi'] = df.apply(lambda row: extract_voltage_class(row.get('Ekipman_Tipi'), row), axis=1)
        
        # --- KRİTİK: KOORDİNAT TEMİZLİĞİ ---
        df = clean_coordinates(df)
        
        data['features'] = df
    else:
        st.error("Veri dosyası bulunamadı.")
        st.stop()

    # --- 2. ARIZA VERİSİ ---
    path_fault = os.path.join(INTERMEDIATE_DIR, "fault_events_clean.csv")
    if os.path.exists(path_fault):
        df_f = pd.read_csv(path_fault)
        if 'started at' in df_f.columns:
            df_f['Ariza_Baslangic_Zamani'] = pd.to_datetime(df_f['started at'], errors='coerce')
            df_f['Mevsim'] = df_f['Ariza_Baslangic_Zamani'].dt.month.map({12:'Kış', 1:'Kış', 2:'Kış', 3:'İlkbahar', 4:'İlkbahar', 5:'İlkbahar', 6:'Yaz', 7:'Yaz', 8:'Yaz', 9:'Sonbahar', 10:'Sonbahar', 11:'Sonbahar'})
        data['faults'] = df_f
        data['features'] = calculate_chronic_flags(data['features'], df_f)

    return data

# --- BAŞLATMA ---
try:
    all_data = load_and_process_data()
    df = all_data['features']
    df_faults = all_data.get('faults')
except Exception as e:
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🎛️ Filtreler")

# İlçe
districts = sorted(df['Ilce'].dropna().unique().tolist()) if 'Ilce' in df.columns else []
sel_dist = st.sidebar.multiselect("📍 İlçe", districts)

# Ekipman Tipi
types = sorted(df['Ekipman_Tipi'].dropna().unique().tolist()) if 'Ekipman_Tipi' in df.columns else []
all_types = st.sidebar.checkbox("✅ Tüm Ekipman Tiplerini Seç", value=True)
sel_types = types if all_types else st.sidebar.multiselect("⚙️ Ekipman Tipi", types)

# Bakım Durumu
maint_opts = ['Tümü', 'Bakımlı', 'Hiç Bakılmadı', 'Veri Yok']
sel_maint = st.sidebar.selectbox("🔧 Bakım Durumu", maint_opts)

# Filtrele
mask = pd.Series([True]*len(df))
if sel_dist: mask &= df['Ilce'].isin(sel_dist)
if sel_types: mask &= df['Ekipman_Tipi'].isin(sel_types)
if sel_maint != 'Tümü': mask &= (df['Bakim_Durumu'] == sel_maint)
df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning("Seçilen filtreye uygun kayıt yok.")
    st.stop()

# --- ANA EKRAN ---
st.title("⚡ Varlık Yönetimi ve Operasyon Paneli")
st.markdown(f"**Analiz Kapsamı:** {len(df_filtered):,} Varlık | **Mod:** EDA & Operasyonel İzleme")

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Toplam Varlık", f"{len(df_filtered):,}")
if 'Yas_Yil' in df_filtered.columns: k1.metric("Ortalama Yaş", f"{df_filtered['Yas_Yil'].mean():.1f} Yıl")
if 'total customer count' in df_filtered.columns: k2.metric("Etkilenen Müşteri", f"{int(df_filtered['total customer count'].sum()):,}")
if 'Kronik_90g' in df_filtered.columns: k3.metric("Son 90 Gün Kronik", f"{int(df_filtered['Kronik_90g'].sum())}", delta="Riskli", delta_color="inverse")

# --- SEKMELER (V2 YAPISI + V6 ÖZELLİKLERİ) ---
tabs = st.tabs([
    "📈 Genel Bakış", 
    "⚠️ Veri Kalitesi Karnesi", 
    "⚡ Arıza Karakteristiği", 
    "⏳ Yaşam Analizi (EDA)", 
    "🔄 Tekrarlayan Sorunlar"
])

# =============================================================================
# TAB 1: GENEL BAKIŞ & HARİTA
# =============================================================================
with tabs[0]:
    # 1. HARİTA BÖLÜMÜ
    st.subheader("🌍 Coğrafi Dağılım")
    
    # Harita için geçerli veriyi hazırla
    if 'Enlem' in df_filtered.columns and 'Boylam' in df_filtered.columns:
        valid_map = df_filtered[df_filtered['Enlem'].notna() & df_filtered['Boylam'].notna()]
        
        if not valid_map.empty:
            df_view = valid_map.sample(min(len(valid_map), 3000))
            
            fig_map = px.scatter_mapbox(
                df_view, lat="Enlem", lon="Boylam", color="Bolge_Tipi",
                size="total customer count" if 'total customer count' in df_view.columns else None,
                hover_name="cbs_id", hover_data=["Ekipman_Tipi", "Ilce", "Bakim_Durumu"],
                zoom=8, height=500, title=f"Varlık Haritası ({len(valid_map):,} nokta)",
                color_discrete_map={'Kentsel': COLORS['urban'], 'Kırsal': COLORS['rural']}
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("⚠️ Harita çizilemiyor: Koordinatlar (X,Y) eksik veya format hatalı.")
            
    st.divider()
    
    # 2. EKİPMAN & BAKIM ÖZETİ
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Ekipman Tipi Dağılımı")
        fig_type = px.pie(df_filtered, names='Ekipman_Tipi', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_type, use_container_width=True)
    
    with c2:
        st.subheader("Bakım Durumu")
        if 'Bakim_Durumu' in df_filtered.columns:
            fig_maint = px.pie(df_filtered, names='Bakim_Durumu', hole=0.4, 
                               color='Bakim_Durumu', color_discrete_map={'Bakımlı': COLORS['success'], 'Hiç Bakılmadı': COLORS['danger']})
            st.plotly_chart(fig_maint, use_container_width=True)

# =============================================================================
# TAB 2: VERİ KALİTESİ
# =============================================================================
with tabs[1]:
    st.subheader("🔍 Veri Sağlığı Karnesi")
    
    missing = df_filtered.isnull().sum().reset_index()
    missing.columns = ['Kolon', 'Eksik']
    missing['Oran'] = (missing['Eksik'] / len(df_filtered)) * 100
    missing = missing[missing['Eksik'] > 0].sort_values('Oran', ascending=False)
    
    if not missing.empty:
        fig_miss = px.bar(missing, x='Oran', y='Kolon', orientation='h', title="Eksik Veri (%)", color='Oran', color_continuous_scale='Reds')
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("✅ Veri seti eksiksiz!")
        
    c1, c2, c3 = st.columns(3)
    # Kritik Kontroller
    no_coord = df_filtered['Enlem'].isna().sum() if 'Enlem' in df_filtered.columns else len(df_filtered)
    if no_coord > 0: c1.warning(f"⚠️ {no_coord} varlıkta koordinat yok.")
    else: c1.success("✅ Koordinatlar Tam.")
        
    no_cust = (df_filtered.get('total customer count', 0) == 0).sum()
    if no_cust > 0: c2.warning(f"⚠️ {no_cust} varlıkta müşteri verisi 0.")
    else: c2.success("✅ Müşteri Verisi Tam.")
    
    no_maint = (df_filtered['Bakim_Durumu'] == 'Veri Yok').sum()
    if no_maint > 0: c3.warning(f"⚠️ {no_maint} varlıkta bakım verisi yok.")
    else: c3.success("✅ Bakım Verisi Tam.")

# =============================================================================
# TAB 3: ARIZA KARAKTERİSTİĞİ
# =============================================================================
with tabs[2]:
    st.subheader("⚡ Arıza İstatistikleri")
    if df_faults is not None:
        rel = df_faults[df_faults['cbs_id'].isin(df_filtered['cbs_id'])]
        if not rel.empty and 'Ariza_Baslangic_Zamani' in rel.columns:
            c1, c2 = st.columns([2, 1])
            with c1:
                rel['Ay'] = rel['Ariza_Baslangic_Zamani'].dt.to_period('M').astype(str)
                trend = rel.groupby('Ay').size().reset_index(name='Adet')
                st.plotly_chart(px.line(trend, x='Ay', y='Adet', title="Aylık Arıza Trendi", markers=True), use_container_width=True)
            with c2:
                if 'Mevsim' in rel.columns:
                    season = rel['Mevsim'].value_counts().reset_index()
                    season.columns = ['Mevsim', 'Adet']
                    st.plotly_chart(px.pie(season, values='Adet', names='Mevsim', title="Mevsimsel Dağılım", hole=0.4), use_container_width=True)
        else: st.info("Arıza verisi yok.")

# =============================================================================
# TAB 4: YAŞAM ANALİZİ
# =============================================================================
with tabs[3]:
    st.subheader("⏳ Kaplan-Meier Yaşam Eğrisi")
    st.markdown("Varlıkların yaşa bağlı hayatta kalma olasılığı (İstatistiksel Baseline).")
    
    if 'duration_days' in df_filtered.columns and 'event' in df_filtered.columns:
        try:
            from lifelines import KaplanMeierFitter
            kmf = KaplanMeierFitter()
            sample = df_filtered.sample(min(len(df_filtered), 5000))
            kmf.fit(sample['duration_days'], event_observed=sample['event'])
            srv = kmf.survival_function_.reset_index()
            srv.columns = ['Gun', 'Olasilik']
            st.plotly_chart(px.line(srv, x='Gun', y='Olasilik', title="Sağkalım Olasılığı", template="plotly_white"), use_container_width=True)
        except: st.info("Survival analizi kütüphanesi eksik veya veri yetersiz.")
    else: st.warning("Yaşam analizi verileri (duration_days, event) bulunamadı.")

# =============================================================================
# TAB 5: TEKRARLAYAN SORUNLAR
# =============================================================================
with tabs[4]:
    st.subheader("🔄 Kronik Varlık Analizi (Çoklu Pencere)")
    
    if 'Kronik_365g' in df_filtered.columns:
        c1, c2, c3 = st.columns(3)
        c1.metric("Son 90 Gün (>2 Arıza)", int(df_filtered['Kronik_90g'].sum()), help="Acil")
        c2.metric("Son 180 Gün (>3 Arıza)", int(df_filtered['Kronik_180g'].sum()))
        c3.metric("Son 1 Yıl (>4 Arıza)", int(df_filtered['Kronik_365g'].sum()), help="Yatırım")
        
        st.write("#### 📋 Kronik Varlık Listesi")
        chronic = df_filtered[df_filtered['Kronik_365g'] == 1].copy()
        if not chronic.empty:
            cols = ['cbs_id', 'Ekipman_Tipi', 'Ilce', 'Kronik_365g_Sayi', 'Toplam_Ariza']
            st.dataframe(chronic[[c for c in cols if c in chronic.columns]].sort_values('Kronik_365g_Sayi', ascending=False), use_container_width=True)
        else: st.success("Kronik varlık yok.")
    else: st.info("Kronik analiz hesaplanamadı.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Varlık Paneli v7.0 | {datetime.now().strftime('%d.%m.%Y')}")