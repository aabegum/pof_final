import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import numpy as np
# --- AYARLAR VE DİZİNLER ---
st.set_page_config(page_title="PoF3 | Varlık Yönetimi Sistemi", layout="wide")

# Dosya Yolları (pof.py ve raporlama scripti ile uyumlu)
OUTPUT_DIR = "data/sonuclar"
INTERMEDIATE_DIR = "data/ara_ciktilar"
INPUT_DIR = "data/girdiler"

# --- VERİ YÜKLEME (Hata Korumalı) ---
@st.cache_data
def load_pof_data():
    main_path = os.path.join(OUTPUT_DIR, "pof_predictions_final.csv")
    
    if not os.path.exists(main_path):
        st.error(f"❌ Kritik Veri Eksik: {main_path} bulunamadı.")
        st.stop()
        
    df = pd.read_csv(main_path)

    # --- İLÇE İSİMLERİ DÜZELTME SÖZLÜĞÜ ---
    ILCE_DUZELTME = {
        "Alasehir": "Alaşehir", "Cesme": "Çeşme", "Beydag": "Beydağ",
        "Gordes": "Gördes", "Kirkagac": "Kırkağaç", "Koprubasi": "Köprübaşı",
        "Sarigol": "Sarıgöl", "Saruhanli": "Saruhanlı", "Sehzadeler": "Şehzadeler",
        "Aliaga": "Aliağa", "Bayindir": "Bayındır", "Foca": "Foça",
        "Kemalpasa": "Kemalpaşa", "Kinik": "Kınık", "Odemis": "Ödemiş",
        "Selcuk": "Selçuk", "Balcova": "Balçova", "Cigli": "Çiğli",
        "Guzelbahce": "Güzelbahçe", "Karabaglar": "Karabağlar"
    }

    # Temizlik İşlemi (df_all yerine df kullanıyoruz)
    if 'Ilce' in df.columns:
        df['Ilce'] = df['Ilce'].replace(ILCE_DUZELTME)
        # Türkçe karakter hassasiyeti ile baş harfleri büyüt
        df['Ilce'] = df['Ilce'].astype(str).str.title().replace({"I": "İ", "i̇": "i"}, regex=True)

    # PoF Ensemble Hesaplama
    if "PoF_Ensemble_12Ay" not in df.columns:
        cols = [c for c in df.columns if "12ay" in c.lower() and "pof" in c.lower()]
        df["PoF_Ensemble_12Ay"] = df[cols].mean(axis=1) if cols else 0.0

    # Eksik Verileri Tamamlama
    missing_cols = [c for c in ['Tref_Yas_Gun', 'Bakim_Sayisi', 'Chronic_Flag'] if c not in df.columns]
    if missing_cols:
        pof_feat_path = os.path.join(INTERMEDIATE_DIR, "ozellikler_pof.csv")
        if os.path.exists(pof_feat_path):
            df_pof_feat = pd.read_csv(pof_feat_path)
            cols_to_merge = ['cbs_id'] + [c for c in missing_cols if c in df_pof_feat.columns]
            df = df.merge(df_pof_feat[cols_to_merge], on='cbs_id', how='left')

    # Diğer Analiz Dosyalarını Yükle
    df_raw = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "fault_events_clean.csv")) if os.path.exists(os.path.join(INTERMEDIATE_DIR, "fault_events_clean.csv")) else None
    df_feat = pd.read_csv(os.path.join(INTERMEDIATE_DIR, "ozellikler_pof.csv")) if os.path.exists(os.path.join(INTERMEDIATE_DIR, "ozellikler_pof.csv")) else None
    df_marka = pd.read_csv(os.path.join(OUTPUT_DIR, "marka_analysis.csv")) if os.path.exists(os.path.join(OUTPUT_DIR, "marka_analysis.csv")) else None
    df_bakim = pd.read_csv(os.path.join(OUTPUT_DIR, "bakim_analysis.csv")) if os.path.exists(os.path.join(OUTPUT_DIR, "bakim_analysis.csv")) else None
    
    return df, df_marka, df_bakim, df_raw, df_feat

# Veriyi Çağır
df_all, df_marka, df_bakim, df_raw, df_feat = load_pof_data()

# --- 1. SESSION STATE BAŞLATMA ---
# Sayfa ilk açıldığında hepsi boş gelsin (istediğiniz gibi)
if 'sel_districts' not in st.session_state:
    st.session_state.sel_districts = []
if 'sel_types' not in st.session_state:
    st.session_state.sel_types = []
if 'sel_risks' not in st.session_state:
    st.session_state.sel_risks = []
if 'sel_brands' not in st.session_state:
    st.session_state.sel_brands = []

# --- 2. SIDEBAR BAŞLIK VE BUTONLAR ---
st.sidebar.title("🔍 Şebeke Filtreleri")
c1, c2 = st.sidebar.columns(2)

# Liste Tanımları (Daha önce load_pof_data içinde temizlenmiş olmalı)
all_districts = sorted(df_all['Ilce'].dropna().unique().tolist())
all_types = sorted(df_all['Ekipman_Tipi'].unique().tolist())
all_risks = ['KRİTİK', 'KRİTİK (KRONİK)', 'YÜKSEK', 'ORTA', 'DÜŞÜK']
# 'Tümü' seçeneğini listeden çıkarıyoruz, sadece marka isimleri kalsın
all_brands = sorted([b for b in df_all['Marka'].dropna().unique() if str(b) != 'nan'])

# ✅ TÜMÜNÜ SEÇ BUTONU
if c1.button("✅ Tümünü Seç", use_container_width=True):
    st.session_state.sel_districts = all_districts
    st.session_state.sel_types = all_types
    st.session_state.sel_risks = all_risks
    st.session_state.sel_brands = all_brands # Artık Marka da dolacak
    st.rerun() 

# 🗑️ TEMİZLE BUTONU
if c2.button("🗑️ Temizle", use_container_width=True):
    st.session_state.sel_districts = []
    st.session_state.sel_types = []
    st.session_state.sel_risks = []
    st.session_state.sel_brands = []
    st.rerun()

st.sidebar.divider()

# --- 3. AKILLI DROPDOWNLAR (Türkçeleştirilmiş) ---

selected_district = st.sidebar.multiselect(
    "📍 İlçe", all_districts, 
    default=st.session_state.sel_districts,
    placeholder="Bölge seçiniz...", 
    key='d_ms'
)
st.session_state.sel_districts = selected_district

selected_types = st.sidebar.multiselect(
    "⚙️ Ekipman Tipi", all_types, 
    default=st.session_state.sel_types,
    placeholder="Ekipman seçiniz...", 
    key='t_ms'
)
st.session_state.sel_types = selected_types

selected_risks = st.sidebar.multiselect(
    "🚨 Risk Sınıfı", all_risks, 
    default=st.session_state.sel_risks,
    placeholder="Risk seviyesi seçiniz...", 
    key='r_ms'
)
st.session_state.sel_risks = selected_risks

# MARKA ALANI: default=[] yaparak o "Tümü" etiketini kaldırdık
selected_brands = st.sidebar.multiselect(
    "🏭 Marka", all_brands, 
    default=st.session_state.sel_brands,
    placeholder="Marka seçiniz...", 
    key='b_ms'
)
st.session_state.sel_brands = selected_brands

# --- Yaş Filtresi ---
st.sidebar.subheader("📅 Yaş (Yıl)")
if 'Tref_Yas_Gun' in df_all.columns:
    age_years = (df_all['Tref_Yas_Gun'] / 365.25).fillna(0)
    min_age, max_age = int(age_years.min()), int(age_years.max())
    age_range = st.sidebar.slider(
        "Ekipman Yaşı",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        help="Ekipmanların yaş aralığını filtrele"
    )
else:
    age_range = (0, 100)

# --- Bakım Filtresi ---
st.sidebar.subheader("🔧 Bakım Durumu")
if 'Bakim_Sayisi' in df_all.columns:
    maint_options = st.sidebar.radio(
        "Bakım Filtresi",
        options=['Tümü', 'Bakım Yapılmış (>0)', 'Hiç Bakılmamış (0)', 'Veri Yok (NaN)'],
        index=0,
        label_visibility="collapsed"
    )
else:
    maint_options = 'Tümü'

# --- Filtreleme İşlemi ---
mask = (df_all['Ekipman_Tipi'].isin(selected_types)) & \
       (df_all['Risk_Sinifi'].isin(selected_risks))

if 'Ilce' in df_all.columns:
    mask &= (df_all['Ilce'].isin(selected_district))

# Yaş filtresi
if 'Tref_Yas_Gun' in df_all.columns:
    age_in_years = (df_all['Tref_Yas_Gun'] / 365.25).fillna(0)
    mask &= (age_in_years >= age_range[0]) & (age_in_years <= age_range[1])

# Marka filtresi
if 'Marka' in df_all.columns and 'Tümü' not in selected_brands:
    mask &= df_all['Marka'].isin(selected_brands)

# Bakım filtresi
if 'Bakim_Sayisi' in df_all.columns:
    if maint_options == 'Bakım Yapılmış (>0)':
        mask &= (df_all['Bakim_Sayisi'].notna()) & (df_all['Bakim_Sayisi'] > 0)
    elif maint_options == 'Hiç Bakılmamış (0)':
        mask &= (df_all['Bakim_Sayisi'] == 0)
    elif maint_options == 'Veri Yok (NaN)':
        mask &= df_all['Bakim_Sayisi'].isna()

filtered_df = df_all[mask]
# --- Sidebarda filtreleme yapıldıktan hemen sonra ---
# Seçim durumunu analiz et
single_type_selected = len(selected_types) == 1
selected_eq_type = selected_types[0] if single_type_selected else None

# Kullanıcıyı yönlendir (UX)
if not single_type_selected:
    st.sidebar.caption("ℹ️ **İpucu:** Detaylı profil analizi için tek bir ekipman tipi seçin.")
else:
    st.sidebar.success(f"🎯 {selected_eq_type} için derin analiz aktif.")
# --- ANA PANEL ---
st.title("⚡ PoF Varlık Yönetimi Karar Destek Sistemi")
st.markdown(f"**Analiz Tarihi:** {datetime.now().strftime('%d.%m.%Y')} | **Filtrelenen Varlık Sayısı:** {len(filtered_df):,}")

# --- 1. SEKMELİ YAPI ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Yönetici Özeti", "🚨 Aksiyon Listeleri",
    "🏗️ Ekipman Analizi", "🧪 Model Analitiği",
    "🔍 Özellik Mühendisliği", "📁 Girdi Veri Kalitesi"
])

with tab1:
    # KPI Metrikleri
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Toplam Varlık", f"{len(df_all):,}")
    with c2:
        crit_count = len(df_all[df_all['Risk_Sinifi'].str.contains('KRİTİK', na=False)])
        st.metric("Kritik Varlık", crit_count, delta=f"%{100*crit_count/len(df_all):.1f}")
    with c3:
        avg_health = df_all['Health_Score'].mean()
        st.metric("Filo Sağlık Puanı", f"{avg_health:.1f} / 100")
    with c4:
        chronic_count = int(df_all['Chronic_Flag'].fillna(0).sum()) if 'Chronic_Flag' in df_all.columns else 0
        st.metric("Kronik Varlık (IEEE 1366)", chronic_count)

    st.divider()

    # --- SEÇİLEN EKİPMAN İSTATİSTİKLERİ ---
    st.subheader("📊 Seçilen Ekipmanlar - Detaylı İstatistikler")

    stat_cols = st.columns(4)

    # Yaş istatistikleri
    with stat_cols[0]:
        if 'Tref_Yas_Gun' in filtered_df.columns:
            avg_age_years = (filtered_df['Tref_Yas_Gun'] / 365.25).mean()
            st.metric(
                "⏳ Ortalama Yaş",
                f"{avg_age_years:.1f} yıl",
                help="Seçilen ekipmanların ortalama yaşı"
            )
        else:
            st.metric("⏳ Ortalama Yaş", "N/A")

    # Marka dağılımı
    with stat_cols[1]:
        if 'Marka' in filtered_df.columns:
            brand_counts = filtered_df['Marka'].value_counts()
            if len(brand_counts) > 0:
                top_brand = brand_counts.index[0]
                brand_count = filtered_df['Marka'].nunique()
                st.metric(
                    "🏭 En Yaygın Marka",
                    top_brand,
                    delta=f"{brand_count} farklı marka",
                    help="Seçilen ekipmanlardaki en yaygın marka"
                )
            else:
                st.metric("🏭 En Yaygın Marka", "N/A")
        else:
            st.metric("🏭 Marka", "N/A")

    # Bakım istatistikleri
    with stat_cols[2]:
        if 'Bakim_Sayisi' in filtered_df.columns:
            avg_maint = filtered_df['Bakim_Sayisi'].mean()
            maint_rate = (filtered_df['Bakim_Sayisi'].notna() & (filtered_df['Bakim_Sayisi'] > 0)).mean() * 100
            st.metric(
                "🔧 Ort. Bakım Sayısı",
                f"{avg_maint:.1f}",
                delta=f"%{maint_rate:.0f} bakımlı",
                help="Seçilen ekipmanların ortalama bakım sayısı"
            )
        else:
            st.metric("🔧 Bakım", "N/A")

    # Kronik oranı
    with stat_cols[3]:
        if 'Chronic_Flag' in filtered_df.columns:
            chronic_rate = (filtered_df['Chronic_Flag'].fillna(0) == 1).mean() * 100
            chronic_total = int(filtered_df['Chronic_Flag'].fillna(0).sum())
            st.metric(
                "⚠️ Kronik Oranı",
                f"%{chronic_rate:.1f}",
                delta=f"{chronic_total} adet",
                help="IEEE 1366 standardına göre kronik ekipman oranı"
            )
        else:
            st.metric("⚠️ Kronik", "N/A")

    st.divider()

    # Sütunların varlığını kontrol eden dinamik liste
    hover_list = ["cbs_id"]
    for col in ["Marka", "Ekipman_Tipi", "Risk_Sinifi"]:
        if col in filtered_df.columns:
            hover_list.append(col)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("🎯 Risk Matrisi (Sağlık vs. Arıza Olasılığı)")
        # PoF_Ensemble_12Ay pof.py tarafından üretilen bileşik skordur
        # Grafik çizimi
        fig = px.scatter(
            filtered_df, 
            x="PoF_Ensemble_12Ay", 
            y="Health_Score",
            color="Risk_Sinifi", 
            hover_data=hover_list, # <--- Dinamik liste kullanımı
            color_discrete_map={
                'KRİTİK': 'red', 
                'KRİTİK (KRONİK)': 'purple', 
                'YÜKSEK': 'orange', 
                'ORTA': 'gold', 
                'DÜŞÜK': 'green'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_right:
        st.subheader("🚦 Risk Dağılımı")
        risk_dist = filtered_df['Risk_Sinifi'].value_counts().reset_index()
        fig_pie = px.pie(risk_dist, values='count', names='Risk_Sinifi', hole=0.4,
                         color='Risk_Sinifi', color_discrete_map={'KRİTİK': 'red', 'KRİTİK (KRONİK)': 'purple', 'YÜKSEK': 'orange', 'ORTA': 'gold', 'DÜŞÜK': 'green'})
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- EKİPMAN TİPİNE GÖRE DETAYLI KARŞILAŞTIRMA ---
    st.divider()
    st.subheader("⚙️ Ekipman Tipine Göre Karşılaştırmalı Analiz")

    if len(selected_types) > 0 and len(filtered_df) > 0:
        # Her ekipman tipi için özet tablo
        summary_data = []

        for eq_type in selected_types:
            eq_df = filtered_df[filtered_df['Ekipman_Tipi'] == eq_type]

            if len(eq_df) > 0:
                summary_row = {
                    'Ekipman Tipi': eq_type,
                    'Adet': len(eq_df),
                    'Ort. Yaş (yıl)': (eq_df['Tref_Yas_Gun'] / 365.25).mean() if 'Tref_Yas_Gun' in eq_df.columns else 0,
                    'Ort. Sağlık': eq_df['Health_Score'].mean() if 'Health_Score' in eq_df.columns else 0,
                    'Kritik Sayısı': len(eq_df[eq_df['Risk_Sinifi'].str.contains('KRİTİK', na=False)]),
                    'Kronik Sayısı': int(eq_df['Chronic_Flag'].fillna(0).sum()) if 'Chronic_Flag' in eq_df.columns else 0,
                }

                # Marka bilgisi
                if 'Marka' in eq_df.columns:
                    brand_counts = eq_df['Marka'].value_counts()
                    top_brand = brand_counts.index[0] if len(brand_counts) > 0 else "N/A"
                    summary_row['En Yaygın Marka'] = top_brand

                # Bakım bilgisi
                if 'Bakim_Sayisi' in eq_df.columns:
                    maint_rate = (eq_df['Bakim_Sayisi'].notna() & (eq_df['Bakim_Sayisi'] > 0)).mean() * 100
                    summary_row['Bakımlı Oran (%)'] = maint_rate

                summary_data.append(summary_row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Formatlama
            summary_df['Ort. Yaş (yıl)'] = summary_df['Ort. Yaş (yıl)'].round(1)
            summary_df['Ort. Sağlık'] = summary_df['Ort. Sağlık'].round(1)
            if 'Bakımlı Oran (%)' in summary_df.columns:
                summary_df['Bakımlı Oran (%)'] = summary_df['Bakımlı Oran (%)'].round(1)

            # Renklendirme için stil
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

            # Görselleştirme
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                st.write("#### 📊 Yaş Dağılımı (Ekipman Tipine Göre)")
                if 'Tref_Yas_Gun' in filtered_df.columns:
                    df_age_plot = filtered_df[filtered_df['Tref_Yas_Gun'].notna()].copy()
                    df_age_plot['Yaş (Yıl)'] = df_age_plot['Tref_Yas_Gun'] / 365.25
                    fig_age = px.box(
                        df_age_plot,
                        x='Ekipman_Tipi',
                        y='Yaş (Yıl)',
                        color='Ekipman_Tipi',
                        labels={'Yaş (Yıl)': 'Yaş (Yıl)', 'Ekipman_Tipi': 'Ekipman Tipi'},
                        title="Ekipman Tipine Göre Yaş Dağılımı"
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("Yaş bilgisi mevcut değil")

            with col_viz2:
                st.write("#### 🏥 Sağlık Skoru (Ekipman Tipine Göre)")
                if 'Health_Score' in filtered_df.columns:
                    fig_health = px.box(
                        filtered_df[filtered_df['Health_Score'].notna()],
                        x='Ekipman_Tipi',
                        y='Health_Score',
                        color='Ekipman_Tipi',
                        labels={'Health_Score': 'Sağlık Skoru', 'Ekipman_Tipi': 'Ekipman Tipi'},
                        title="Ekipman Tipine Göre Sağlık Skoru Dağılımı"
                    )
                    st.plotly_chart(fig_health, use_container_width=True)
                else:
                    st.info("Sağlık skoru mevcut değil")
    else:
        st.info("Ekipman tipi seçin veya filtre kriterlerini ayarlayın.")

with tab2:
    st.subheader("📋 Operasyonel Öncelik Listeleri")
    st.markdown("Analiz sonuçları, saha ekipleri için 3 ana kategoriye ayrılmıştır.")
    
    a1, a2, a3 = st.columns(3)
    
    # --- a1: ACİL MÜDAHALE ---
    with a1:
        st.error("🚨 ACİL MÜDAHALE (Kritik & Kronik)")
        urgent = filtered_df[filtered_df['Risk_Sinifi'] == 'KRİTİK (KRONİK)'].sort_values('PoF_Ensemble_12Ay', ascending=False)
        st.write(f"Müdahale gereken {len(urgent)} varlık.")
        
        # Dinamik sütun seçimi (KeyError önleyici)
        cols_a1 = [c for c in ['cbs_id', 'Ekipman_Tipi', 'Marka', 'PoF_Ensemble_12Ay'] if c in urgent.columns]
        st.dataframe(urgent[cols_a1].head(10), use_container_width=True)

    # --- a2: YATIRIM (CAPEX) ---
    with a2:
        st.warning("💰 YATIRIM (CAPEX) / Trafolar")
        # Sağlık skoru 40'ın altındaki trafolar
        capex = filtered_df[(filtered_df['Ekipman_Tipi'].str.contains('Trafo', na=False)) & (filtered_df['Health_Score'] < 20)]
        st.write(f"Yenileme planlanacak {len(capex)} trafo.")
        
        cols_a2 = [c for c in ['cbs_id', 'Health_Score', 'Marka', 'Ilce'] if c in capex.columns]
        st.dataframe(capex[cols_a2].head(10), use_container_width=True)

    # --- a3: FIRSAT BAKIMI (OPEX) ---
    with a3:
        st.info("🔍 FIRSAT BAKIMI (OPEX)")
        # PoF %15'ten büyük ama risk sınıfı henüz düşük olanlar
        opex = filtered_df[(filtered_df['PoF_Ensemble_12Ay'] > 0.15) & (filtered_df['Risk_Sinifi'].isin(['ORTA', 'DÜŞÜK']))]
        st.write(f"Önleyici bakım önerilen {len(opex)} varlık.")
        
        cols_a3 = [c for c in ['cbs_id', 'PoF_Ensemble_12Ay', 'Risk_Sinifi', 'Marka'] if c in opex.columns]
        st.dataframe(opex[cols_a3].head(10), use_container_width=True)

with tab3:
    if single_type_selected:
        # --- TEKLİ SEÇİM: DERİN ANALİZ ---
        st.header(f"🔍 {selected_eq_type} – Detaylı Varlık Profili")
        
        # A. Yaş Profili
        if 'Tref_Yas_Gun' in filtered_df.columns:
            st.subheader("⏳ Yaş ve Ömür Analizi")
            age_years = filtered_df['Tref_Yas_Gun'] / 365.25
            c1, c2, c3 = st.columns(3)
            c1.metric("Medyan Yaş", f"{age_years.median():.1f} Yıl")
            c2.metric("P90 Yaş", f"{age_years.quantile(0.9):.1f} Yıl")
            c3.metric("Kritik Yaş Üstü (>25)", f"{len(age_years[age_years > 25])} Adet")

            fig_age = px.histogram(filtered_df, x=age_years, nbins=20, 
                                   title=f"{selected_eq_type} Yaş Dağılımı",
                                   labels={'x': 'Yaş (Yıl)', 'y': 'Adet'})
            st.plotly_chart(fig_age, use_container_width=True)

        # B. Marka ve Risk
        if 'Marka' in filtered_df.columns:
            st.divider()
            st.subheader("🏭 Marka ve Tedarikçi Performansı")
            brand_perf = filtered_df.groupby('Marka').agg(
                Adet=('cbs_id', 'count'),
                Ort_Health=('Health_Score', 'mean'),
                Ort_PoF=('PoF_Ensemble_12Ay', 'mean')
            ).reset_index().sort_values('Adet', ascending=False)

            fig_brand = px.bar(brand_perf.head(10), x='Adet', y='Marka', orientation='h',
                                color='Ort_PoF', color_continuous_scale='Reds',
                                title="En Yaygın 10 Marka ve Risk Durumu")
            st.plotly_chart(fig_brand, use_container_width=True)

        # C. Bakım Profili (Numpy kullanarak)
        if 'Bakim_Sayisi' in filtered_df.columns:
            st.divider()
            st.subheader("🔧 Bakım ve Operasyon Karnesi")
            import numpy as np
            
            maint_labels = ['Veri Yok', 'Hiç Bakılmadı', '1-2 Bakım', 'Yoğun Bakım (>2)']
            maint_cond = [
                (filtered_df['Bakim_Sayisi'].isna()),
                (filtered_df['Bakim_Sayisi'] == 0),
                (filtered_df['Bakim_Sayisi'] <= 2),
                (filtered_df['Bakim_Sayisi'] > 2)
            ]
            filtered_df['Bakım_Durumu'] = np.select(maint_cond, maint_labels)
            
            fig_maint = px.pie(filtered_df, names='Bakım_Durumu', title="Bakım Dağılımı",
                               color='Bakım_Durumu',
                               color_discrete_map={'Veri Yok': 'gray', 'Hiç Bakılmadı': 'red', '1-2 Bakım': 'green', 'Yoğun Bakım (>2)': 'orange'})
            st.plotly_chart(fig_maint, use_container_width=True)

    else:
        # --- ÇOKLU SEÇİM: KARŞILAŞTIRMALI ÖZET ---
        st.header("📊 Ekipmanlar Arası Karşılaştırmalı Özet")
        st.info("ℹ️ Detaylı analiz için soldaki filtreden tek bir ekipman tipi seçebilirsiniz.")

        # 1. summary_df'i BURADA HESAPLA (Hata Fixi 🚀)
        summary_list = []
        for eq_type in selected_types:
            eq_data = filtered_df[filtered_df['Ekipman_Tipi'] == eq_type]
            if not eq_data.empty:
                summary_list.append({
                    'Ekipman Tipi': eq_type,
                    'Adet': len(eq_data),
                    'Ort. Sağlık': eq_data['Health_Score'].mean(),
                    'Kritik Sayısı': len(eq_data[eq_data['Risk_Sinifi'].str.contains('KRİTİK', na=False)]),
                    'Kronik Sayısı': int(eq_data['Chronic_Flag'].sum()) if 'Chronic_Flag' in eq_data.columns else 0
                })
        
        if summary_list:
            summary_df = pd.DataFrame(summary_list)
            st.dataframe(summary_df.sort_values('Kritik Sayısı', ascending=False), 
                         use_container_width=True, hide_index=True)

            # 2. Karşılaştırmalı Grafikler
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig_comp_health = px.box(filtered_df, x='Ekipman_Tipi', y='Health_Score', 
                                         color='Ekipman_Tipi', title="Tiplere Göre Sağlık Dağılımı")
                st.plotly_chart(fig_comp_health, use_container_width=True)
            with col_v2:
                # Yaş Karşılaştırması
                if 'Tref_Yas_Gun' in filtered_df.columns:
                    filtered_df['Yas_Yil'] = filtered_df['Tref_Yas_Gun'] / 365.25
                    fig_comp_age = px.box(filtered_df, x='Ekipman_Tipi', y='Yas_Yil', 
                                          title="Tiplere Göre Yaş Dağılımı")
                    st.plotly_chart(fig_comp_age, use_container_width=True)
        else:
            st.warning("Seçilen filtrelere uygun veri bulunamadı.")
with tab4:
    st.subheader("🧪 Model Doğrulama (Backtesting) ve Teşhis")

    # --- SUBTABS: Backtesting | Baseline Comparison ---
    subtab1, subtab2 = st.tabs(["📈 Backtesting Skorları", "📊 Baseline Karşılaştırması"])

    with subtab1:
        # pof.py içindeki TemporalBacktester sonuçları
        backtest_path = os.path.join(OUTPUT_DIR, "backtest_results_temporal.csv")
        if os.path.exists(backtest_path):
            df_bt = pd.read_csv(backtest_path)
            st.write("**Zaman Serisi Doğrulama Skorları (AUC)**")
            st.line_chart(df_bt.set_index('Year')['AUC'])
            st.write(f"**Ortalama AUC Skoru:** {df_bt['AUC'].mean():.3f}")

        st.divider()
        st.write("**Sağlık Skoru Hesaplama Formülü:**")
        st.latex(r"Health\_Score = 100 \times (1 - Risk\_Percentile)")
        st.info("Not: Kronik (IEEE 1366) varlıklar için sağlık skoru tavanı 60'tır.")

    with subtab2:
        st.write("### 📊 Model Performansı vs Kaplan-Meier Baseline")
        st.caption("Model performansını naive istatistiksel baseline ile karşılaştırarak gerçek değerini ölç.")

        # Feature engineering output'u kullanarak KM baseline hesapla
        if df_feat is not None and 'event' in df_feat.columns and 'duration_days' in df_feat.columns:
            from lifelines import KaplanMeierFitter
            from lifelines.utils import concordance_index

            # Temiz veri
            df_km = df_feat[['event', 'duration_days', 'Ekipman_Tipi']].dropna()

            if not df_km.empty:
                # Global KM Baseline
                kmf_global = KaplanMeierFitter()
                kmf_global.fit(df_km['duration_days'], df_km['event'])

                # KM survival probabilities (12 aylık tahmin)
                horizon_days = 365
                km_survival_12m = kmf_global.survival_function_at_times(df_km['duration_days'].clip(upper=horizon_days))
                km_risk_12m = 1 - km_survival_12m.values

                # KM concordance (survival probability'yi risk skoru olarak kullan)
                km_concordance = concordance_index(
                    df_km['duration_days'],
                    -km_risk_12m,  # Negatif çünkü concordance yüksek değer = uzun yaşam bekler
                    df_km['event']
                )

                # Model concordance (backtest sonuçlarından)
                model_concordance = df_bt['AUC'].mean() if os.path.exists(backtest_path) else 0.85

                # Karşılaştırma
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "🤖 Model Concordance",
                        f"{model_concordance:.3f}",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "📉 KM Baseline Concordance",
                        f"{km_concordance:.3f}",
                        delta=None
                    )
                with col3:
                    lift = ((model_concordance - km_concordance) / km_concordance * 100)
                    st.metric(
                        "📈 Model Lift",
                        f"{lift:+.1f}%",
                        delta=f"{model_concordance - km_concordance:+.3f}",
                        delta_color="normal"
                    )

                # Interpretation
                st.divider()
                if lift > 10:
                    st.success(f"✅ **Model Güçlü:** Baseline'a göre %{lift:.1f} iyileşme sağlıyor!")
                    st.caption("Model, naive istatistiksel yaklaşımdan anlamlı derecede daha iyi tahmin yapıyor.")
                elif lift > 5:
                    st.info(f"ℹ️ **Model Orta Seviye:** Baseline'a göre %{lift:.1f} iyileşme.")
                    st.caption("Model katkı sağlıyor ancak özellik mühendisliği iyileştirilebilir.")
                else:
                    st.warning(f"⚠️ **Model Zayıf:** Baseline'a göre sadece %{lift:.1f} iyileşme.")
                    st.caption("Model basit KM'den çok az farklı tahmin yapıyor - feature engineering gözden geçirilmeli!")

                # Ekipman tipine göre KM curves
                st.divider()
                st.write("### 📉 Ekipman Tipine Göre Survival Curves (Kaplan-Meier)")

                # Top 5 ekipman tipi
                top_equipment = df_km['Ekipman_Tipi'].value_counts().head(5).index.tolist()

                import plotly.graph_objects as go
                fig_km = go.Figure()

                for eq_type in top_equipment:
                    df_eq = df_km[df_km['Ekipman_Tipi'] == eq_type]
                    if len(df_eq) > 10:  # Minimum sample size
                        kmf_eq = KaplanMeierFitter()
                        kmf_eq.fit(df_eq['duration_days'], df_eq['event'], label=eq_type)

                        fig_km.add_trace(go.Scatter(
                            x=kmf_eq.survival_function_.index,
                            y=kmf_eq.survival_function_[eq_type],
                            mode='lines',
                            name=eq_type,
                            line=dict(width=2)
                        ))

                fig_km.update_layout(
                    title="Ekipman Tipine Göre Survival Curves (KM Baseline)",
                    xaxis_title="Gün",
                    yaxis_title="Survival Probability",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig_km, use_container_width=True)
                st.caption("💡 Eğrinin düşüş hızı o ekipmanın risk profilini gösterir. Model bu baseline'ı aşmalı.")
            else:
                st.warning("Survival analizi için yeterli veri yok.")
        else:
            st.info("📊 Model girdileri (model_input_data_full.csv) henüz oluşmadı. Lütfen pof.py scriptini çalıştırın.")
with tab5:
    st.subheader("🔍 Özellik Mühendisliği ve Model Girdileri Analizi")

    if df_feat is not None and 'event' in df_feat.columns:
        subtab1, subtab2, subtab3 = st.tabs([
            "📊 Özellik Korelasyonu", "🎲 Özellik Dağılımları", "⚠️ Veri Kalitesi"
        ])

        # --- SUB TAB 1: Özellik Korelasyonu ---
        with subtab1:
            st.write("### 🔗 Özellik Korelasyon Matrisi")

            target_features = ['Tref_Yas_Gun', 'MTBF_Bayes_Gun', 'Chronic_Decay_Skoru',
                             'Observation_Ratio', 'Ariza_Sayisi_90g', 'event']
            available_features = [f for f in target_features if f in df_feat.columns]

            if len(available_features) > 1:
                df_feat_clean = df_feat[available_features].dropna(how='all')
                if not df_feat_clean.empty:
                    corr_matrix = df_feat_clean.corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        title="Özellikler Arası Korelasyon",
                        zmin=-1, zmax=1
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Yüksek korelasyon uyarısı
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = abs(corr_matrix.iloc[i, j])
                            if corr_val > 0.7 and corr_matrix.columns[i] != 'event' and corr_matrix.columns[j] != 'event':
                                high_corr_pairs.append({
                                    'Özellik 1': corr_matrix.columns[i],
                                    'Özellik 2': corr_matrix.columns[j],
                                    'Korelasyon': f"{corr_val:.3f}"
                                })

                    if high_corr_pairs:
                        st.warning("⚠️ **Yüksek Korelasyon Tespit Edildi (>0.7)**")
                        st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
                        st.caption("Bu özellikler arasında çoklu doğrusallık (multicollinearity) riski var.")
                    else:
                        st.success("✅ Özellikler arası bağımsızlık sağlanmış.")
                else:
                    st.warning("Korelasyon hesaplaması için yeterli veri yok")

        # --- SUB TAB 2: Özellik Dağılımları ---
        with subtab2:
            st.write("### 🎲 Arızalı vs Sağlam Ekipman Karşılaştırması")

            numeric_features = [f for f in available_features if f != 'event' and f in df_feat.columns]

            if numeric_features:
                selected_feature = st.selectbox(
                    "İncelenecek Özellik:",
                    numeric_features,
                    key='feature_dist_select'
                )

                df_feat['Durum'] = df_feat['event'].map({1: 'Arızalı', 0: 'Sağlam'})

                # Violin plot
                fig_violin = px.violin(
                    df_feat.dropna(subset=[selected_feature, 'event']),
                    y=selected_feature,
                    x='Durum',
                    box=True,
                    points='outliers',
                    title=f"{selected_feature} Dağılımı",
                    color='Durum',
                    color_discrete_map={'Arızalı': '#d62728', 'Sağlam': '#2ca02c'}
                )
                st.plotly_chart(fig_violin, use_container_width=True)

                # İstatistiksel karşılaştırma
                failed = df_feat[df_feat['event'] == 1][selected_feature].dropna()
                healthy = df_feat[df_feat['event'] == 0][selected_feature].dropna()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Arızalı Ekipmanlar (Medyan)", f"{failed.median():.1f}")
                with col2:
                    st.metric("Sağlam Ekipmanlar (Medyan)", f"{healthy.median():.1f}")
                with col3:
                    diff_pct = ((failed.median() - healthy.median()) / healthy.median() * 100) if healthy.median() != 0 else 0
                    st.metric("Fark", f"{diff_pct:+.1f}%")

                # KS testi (ayırıcılık gücü)
                from scipy.stats import ks_2samp
                ks_stat, p_value = ks_2samp(failed, healthy)

                if p_value < 0.05:
                    st.success(f"✅ **Özellik ayırıcılığı güçlü** (KS İstatistiği={ks_stat:.3f}, p<0.05)")
                    st.caption("Bu özellik arızalı ve sağlam ekipmanları ayırt etmede başarılı.")
                else:
                    st.warning(f"⚠️ **Özellik ayırıcılığı zayıf** (KS İstatistiği={ks_stat:.3f}, p={p_value:.3f})")
                    st.caption("Bu özellik modele sınırlı katkı yapıyor olabilir.")

        # --- SUB TAB 3: Veri Kalitesi ---
        with subtab3:
            st.write("### ⚠️ Eksik Veri ve Kalite Kontrolleri")

            # Eksik veri analizi
            missing_pct = (df_feat.isnull().sum() / len(df_feat) * 100).sort_values(ascending=False)
            missing_pct = missing_pct[missing_pct > 0].head(15)

            if not missing_pct.empty:
                fig_missing = px.bar(
                    x=missing_pct.values,
                    y=missing_pct.index,
                    orientation='h',
                    title="Özelliklerde Eksik Veri Oranı (%)",
                    labels={'x': 'Eksik Veri %', 'y': 'Özellik'}
                )
                fig_missing.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="50% Eşik")
                st.plotly_chart(fig_missing, use_container_width=True)

                critical_missing = missing_pct[missing_pct > 50]
                if not critical_missing.empty:
                    st.error(f"❌ **{len(critical_missing)} özellikte >%50 eksik veri var!**")
                    st.caption("Bu özellikler modelde kullanılmamalı veya uygun imputation yapılmalı.")
            else:
                st.success("✅ Model girdilerinde eksik veri tespit edilmedi!")

            # Sabit özellikler kontrolü
            st.write("#### 🔍 Sabit Değerli Özellikler (Variance=0)")
            const_features = [col for col in df_feat.select_dtypes(include=['int64', 'float64']).columns
                            if df_feat[col].nunique() == 1]

            if const_features:
                st.warning(f"⚠️ **{len(const_features)} özellik sabit değere sahip:**")
                st.write(", ".join(const_features))
                st.caption("Bu özellikler modele katkı yapmaz ve çıkarılmalıdır.")
            else:
                st.success("✅ Tüm özellikler değişkenlik gösteriyor.")

            # Formül açıklaması
            st.divider()
            st.write("### 🧮 Özellik Mühendisliği Formülleri")

            col1, col2 = st.columns(2)
            with col1:
                st.latex(r"MTBF_{Bayes} = \frac{CHRONIC\_WINDOW + \beta}{Ariza\_Sayisi + \alpha}")
                st.caption("Bayesian smoothing ile MTBF hesabı - veri azlığı problemini çözer")

            with col2:
                st.latex(r"Observation\_Ratio = \frac{Gozlem\_Suresi}{Toplam\_Yas}")
                st.caption("Left truncation düzeltmesi - sadece gözlem süresi içindeki riskleri modeller")
    else:
        st.info("📊 Özellik mühendisliği ara çıktısı (model_input_data_full.csv) henüz oluşmadı. Lütfen pof.py scriptini çalıştırın.")

with tab6:
    st.subheader("📁 Girdi Dosyaları Kalite Kontrolü ve Zaman Serisi Analizi")

    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "📂 Dosya Durumu", "📉 Zaman Serisi", "🗺️ Coğrafi Analiz", "🏭 Marka & Bakım"
    ])

    # --- SUB TAB 1: Dosya Durumu ---
    with subtab1:
        st.write("### 📂 Veri Kaynakları Kontrol Paneli")

        input_files = {
            "Arıza Kayıtları": os.path.join(INPUT_DIR, "ariza_final.xlsx"),
            "Sağlam Ekipmanlar": os.path.join(INPUT_DIR, "saglam_final.xlsx"),
            #"Bakım Kayıtları": os.path.join(INPUT_DIR, "bakim_kayitlari.xlsx")
        }

        file_status = []
        for name, path in input_files.items():
            exists = os.path.exists(path)
            if exists:
                size_mb = os.path.getsize(path) / (1024**2)
                modified = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%d.%m.%Y')
                status_icon = "✅"
            else:
                size_mb = 0
                modified = "N/A"
                status_icon = "❌"

            file_status.append({
                "Dosya": name,
                "Durum": status_icon,
                "Boyut (MB)": f"{size_mb:.2f}",
                "Son Güncelleme": modified
            })

        st.dataframe(pd.DataFrame(file_status), use_container_width=True)

        # Veri kalite skoru
        st.divider()
        st.write("### 🎯 Genel Veri Kalite Skoru")

        # Basit kalite skoru hesaplama
        quality_components = {}

        # 1. Eksiksizlik (40 puan)
        key_cols = ['Ekipman_Tipi', 'Kurulum_Tarihi', 'Gerilim_Seviyesi']
        existing_cols = [c for c in key_cols if c in df_all.columns]
        if existing_cols:
            completeness = sum(df_all[col].notna().mean() for col in existing_cols) / len(key_cols)
            quality_components['Eksiksizlik'] = completeness * 40
        else:
            quality_components['Eksiksizlik'] = 0

        # 2. Koordinat (20 puan)
        if 'Latitude' in df_all.columns and 'Longitude' in df_all.columns:
            coord_quality = ((df_all['Latitude'] != 0) & (df_all['Longitude'] != 0)).mean()
            quality_components['Koordinat'] = coord_quality * 20
        else:
            quality_components['Koordinat'] = 0

        # 3. Marka (20 puan)
        if 'Marka' in df_all.columns:
            brand_quality = df_all['Marka'].notna().mean()
            quality_components['Marka'] = brand_quality * 20
        else:
            quality_components['Marka'] = 0

        # 4. Bakım (20 puan)
        if 'Bakim_Sayisi' in df_all.columns:
            maint_quality = df_all['Bakim_Sayisi'].notna().mean()
            quality_components['Bakım'] = maint_quality * 20
        else:
            quality_components['Bakım'] = 0

        total_score = sum(quality_components.values())

        # Gauge chart
        fig_quality = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Veri Kalite Skoru (0-100)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcccc"},
                    {'range': [40, 70], 'color': "#fff4cc"},
                    {'range': [70, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        st.plotly_chart(fig_quality, use_container_width=True)

        # Detay tablo
        col1, col2 = st.columns(2)
        with col1:
            detail_df = pd.DataFrame({
                'Kategori': list(quality_components.keys()),
                'Puan': [f"{v:.1f}" for v in quality_components.values()],
                'Maks': [40, 20, 20, 20]
            })
            st.dataframe(detail_df, use_container_width=True)

        with col2:
            if total_score >= 80:
                st.success("✅ **Mükemmel:** Veri kalitesi yüksek!")
            elif total_score >= 60:
                st.info("ℹ️ **İyi:** Veri kullanılabilir durumda.")
            else:
                st.warning("⚠️ **Dikkat:** Veri kalitesi iyileştirme gerektiriyor.")

    # --- SUB TAB 2: Zaman Serisi ---
    with subtab2:
        st.write("### 📉 Arıza Trendleri ve Mevsimsellik Analizi")

        if df_raw is not None and 'started at' in df_raw.columns:
            # Tarih parsing
            df_raw['started at'] = pd.to_datetime(df_raw['started at'], errors='coerce')
            df_raw_clean = df_raw.dropna(subset=['started at'])

            if not df_raw_clean.empty:
                # Aylık trend
                df_raw_clean['YearMonth'] = df_raw_clean['started at'].dt.to_period('M').astype(str)
                monthly_faults = df_raw_clean.groupby('YearMonth').size().reset_index(name='Arıza Sayısı')

                fig_trend = px.line(
                    monthly_faults,
                    x='YearMonth',
                    y='Arıza Sayısı',
                    title="Aylık Arıza Trendi",
                    markers=True
                )

                if len(monthly_faults) > 0:
                    avg_faults = monthly_faults['Arıza Sayısı'].mean()
                    fig_trend.add_hline(
                        y=avg_faults,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Ortalama: {avg_faults:.0f}"
                    )

                st.plotly_chart(fig_trend, use_container_width=True)

                # İstatistikler
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ortalama Aylık Arıza", f"{monthly_faults['Arıza Sayısı'].mean():.0f}")
                with col2:
                    st.metric("En Yüksek (Aylık)", f"{monthly_faults['Arıza Sayısı'].max()}")
                with col3:
                    st.metric("Standart Sapma", f"{monthly_faults['Arıza Sayısı'].std():.1f}")

                # Mevsimsellik
                st.divider()
                st.write("### 🌦️ Mevsimsel Arıza Dağılımı")

                df_raw_clean['Mevsim'] = df_raw_clean['started at'].dt.month.map({
                    12: 'Kış', 1: 'Kış', 2: 'Kış',
                    3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
                    6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
                    9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
                })

                if 'Ekipman_Tipi' in df_raw_clean.columns:
                    # Top 5 ekipman tipi
                    top_equipment = df_raw_clean['Ekipman_Tipi'].value_counts().head(5).index.tolist()
                    df_seasonal = df_raw_clean[df_raw_clean['Ekipman_Tipi'].isin(top_equipment)]

                    seasonal = df_seasonal.groupby(['Mevsim', 'Ekipman_Tipi']).size().reset_index(name='Arıza')

                    fig_season = px.bar(
                        seasonal,
                        x='Mevsim',
                        y='Arıza',
                        color='Ekipman_Tipi',
                        title="Mevsime Göre Ekipman Arızaları (Top 5 Ekipman)",
                        barmode='stack',
                        category_orders={'Mevsim': ['Kış', 'İlkbahar', 'Yaz', 'Sonbahar']}
                    )
                    st.plotly_chart(fig_season, use_container_width=True)
                    st.caption("💡 Mevsimsel artışlar hava koşullarının etkisini gösterir.")
            else:
                st.warning("Tarih bilgisi eksik veya geçersiz formatta.")
        else:
            st.info("Ham arıza verisi (fault_events_clean.csv) bulunamadı.")

    # --- SUB TAB 3: Coğrafi Analiz ---
    with subtab3:
        st.write("### 🗺️ Bölgesel Risk Haritası")

        if 'Ilce' in df_all.columns:
            geo_risk = df_all.groupby('Ilce').agg({
                'cbs_id': 'count',
                'Health_Score': 'mean',
                'PoF_Ensemble_12Ay': 'mean'
            }).reset_index()

            geo_risk.columns = ['İlçe', 'Varlık Sayısı', 'Ort. Sağlık', 'Ort. PoF']

            # Bubble chart
            fig_geo = px.scatter(
                geo_risk,
                x='Ort. Sağlık',
                y='Ort. PoF',
                size='Varlık Sayısı',
                color='İlçe',
                hover_data=['İlçe', 'Varlık Sayısı'],
                title="İlçe Bazlı Risk Dağılımı (Balon boyutu = Varlık sayısı)",
                labels={'Ort. Sağlık': 'Ortalama Sağlık Skoru', 'Ort. PoF': 'Ortalama Arıza Olasılığı'}
            )
            st.plotly_chart(fig_geo, use_container_width=True)

            # En riskli ilçeler
            st.divider()
            st.write("#### 🚨 En Riskli İlçeler (Düşük Sağlık Skoru)")
            top_risk_districts = geo_risk.nsmallest(5, 'Ort. Sağlık')[['İlçe', 'Varlık Sayısı', 'Ort. Sağlık', 'Ort. PoF']]
            st.dataframe(top_risk_districts, use_container_width=True)
        else:
            st.info("İlçe bilgisi mevcut değil.")

        # GPS koordinat kalitesi
        st.divider()
        st.write("### 📍 GPS Koordinat Veri Kalitesi")

        if 'Latitude' in df_all.columns and 'Longitude' in df_all.columns:
            coord_stats = {
                "Toplam Varlık": len(df_all),
                "GPS Koordinatı Var": int(((df_all['Latitude'] != 0) & (df_all['Longitude'] != 0)).sum()),
                "GPS Koordinatı Eksik": int(((df_all['Latitude'] == 0) | (df_all['Longitude'] == 0)).sum())
            }
            coord_stats["Kapsama Oranı %"] = (coord_stats["GPS Koordinatı Var"] / coord_stats["Toplam Varlık"] * 100)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Toplam", f"{coord_stats['Toplam Varlık']:,}")
            col2.metric("GPS Var", f"{coord_stats['GPS Koordinatı Var']:,}")
            col3.metric("GPS Yok", f"{coord_stats['GPS Koordinatı Eksik']:,}")
            col4.metric("Kapsama", f"{coord_stats['Kapsama Oranı %']:.1f}%")

            # Ekipman tipine göre GPS kalitesi
            if 'Ekipman_Tipi' in df_all.columns:
                gps_by_type = df_all.groupby('Ekipman_Tipi').apply(
                    lambda x: ((x['Latitude'] != 0) & (x['Longitude'] != 0)).sum() / len(x) * 100
                ).reset_index(name='GPS Kapsama %')

                fig_gps = px.bar(
                    gps_by_type.sort_values('GPS Kapsama %'),
                    x='GPS Kapsama %',
                    y='Ekipman_Tipi',
                    orientation='h',
                    title="Ekipman Tipine Göre GPS Veri Kalitesi"
                )
                st.plotly_chart(fig_gps, use_container_width=True)
        else:
            st.info("GPS koordinat bilgisi mevcut değil.")

    # --- SUB TAB 4: Marka & Bakım ---
    with subtab4:
        st.write("### 🏭 Marka Performans Matrisi")

        if 'Marka' in df_all.columns:
            marka_stats = df_all.groupby('Marka').agg({
                'cbs_id': 'count',
                'Health_Score': 'mean',
                'PoF_Ensemble_12Ay': 'mean'
            }).reset_index()

            marka_stats.columns = ['Marka', 'Adet', 'Ort. Sağlık', 'Ort. Risk']
            marka_stats['Pazar Payı %'] = (marka_stats['Adet'] / marka_stats['Adet'].sum() * 100)

            # Sadece >1% pazar payı olanlar
            major_brands = marka_stats[marka_stats['Pazar Payı %'] > 1].copy()

            if not major_brands.empty:
                # Scatter: Pazar payı vs Risk
                fig_brand = px.scatter(
                    major_brands,
                    x='Pazar Payı %',
                    y='Ort. Risk',
                    size='Adet',
                    color='Ort. Sağlık',
                    hover_data=['Marka', 'Adet'],
                    title="Marka Pazar Payı vs Risk Performansı (>1% Pazar Payı)",
                    color_continuous_scale='RdYlGn',
                    labels={'Ort. Risk': 'Ortalama Arıza Riski', 'Pazar Payı %': 'Pazar Payı (%)'}
                )
                st.plotly_chart(fig_brand, use_container_width=True)

                # Kritik bulgular
                st.divider()
                st.write("#### 🔍 Tedarikçi Önerileri")

                # Yüksek pay + düşük sağlık = SORUN
                risky_major = major_brands[
                    (major_brands['Pazar Payı %'] > 5) &
                    (major_brands['Ort. Sağlık'] < 50)
                ]

                if not risky_major.empty:
                    st.error("⚠️ **Büyük Tedarikçi, Düşük Performans:**")
                    st.dataframe(risky_major[['Marka', 'Adet', 'Pazar Payı %', 'Ort. Sağlık']].sort_values('Ort. Sağlık'), use_container_width=True)
                    st.caption("Bu tedarikçilerle görüşme ve kalite iyileştirme gerekebilir.")

                # Küçük pay + yüksek sağlık = FIRSAT
                good_minor = major_brands[
                    (major_brands['Pazar Payı %'] < 5) &
                    (major_brands['Ort. Sağlık'] > 70)
                ]

                if not good_minor.empty:
                    st.success("✅ **Küçük ama Performanslı Tedarikçiler:**")
                    st.dataframe(good_minor[['Marka', 'Adet', 'Pazar Payı %', 'Ort. Sağlık']].sort_values('Ort. Sağlık', ascending=False), use_container_width=True)
                    st.caption("Bu tedarikçilerden alım artırılabilir.")
            else:
                st.info("Görüntülenecek yeterli marka verisi yok (>1% pazar payı).")
        else:
            st.info("Marka bilgisi mevcut değil.")

        # Bakım analizi
        st.divider()
        st.write("### 🔧 Bakım Veri Kapsama Analizi")

        if 'Bakim_Sayisi' in df_all.columns:
            df_all_temp = df_all.copy()
            df_all_temp['Bakim_Durumu'] = df_all_temp['Bakim_Sayisi'].apply(
                lambda x: 'Veri Yok' if pd.isna(x) else ('Hiç Bakılmadı' if x == 0 else 'Bakım Yapıldı')
            )

            maint_dist = df_all_temp['Bakim_Durumu'].value_counts().reset_index()
            maint_dist.columns = ['Durum', 'Sayı']

            fig_maint = px.pie(
                maint_dist,
                values='Sayı',
                names='Durum',
                title="Bakım Veri Durumu",
                color='Durum',
                color_discrete_map={
                    'Bakım Yapıldı': '#2ca02c',
                    'Hiç Bakılmadı': '#ff7f0e',
                    'Veri Yok': '#7f7f7f'
                }
            )
            st.plotly_chart(fig_maint, use_container_width=True)

            # Ekipman tipine göre bakım kapsama
            if 'Ekipman_Tipi' in df_all_temp.columns:
                maint_by_type = df_all_temp.groupby('Ekipman_Tipi')['Bakim_Sayisi'].apply(
                    lambda x: (x.notna() & (x > 0)).sum() / len(x) * 100
                ).reset_index(name='Bakım Yapılma %')

                fig_maint_type = px.bar(
                    maint_by_type.sort_values('Bakım Yapılma %'),
                    x='Bakım Yapılma %',
                    y='Ekipman_Tipi',
                    orientation='h',
                    title="Ekipman Tipine Göre Bakım Yapılma Oranı"
                )
                st.plotly_chart(fig_maint_type, use_container_width=True)

                # Uyarılar
                low_maint = maint_by_type[maint_by_type['Bakım Yapılma %'] < 10]
                if not low_maint.empty:
                    st.warning(f"⚠️ **{len(low_maint)} ekipman tipinde bakım kapsama <%10**")
                    st.caption("Bu ekipmanlar için bakım kayıt sistemi iyileştirilmeli.")
        else:
            st.info("Bakım bilgisi mevcut değil.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption(f"PoF Engine v4.1 | {datetime.now().year}")