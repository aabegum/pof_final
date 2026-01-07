import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Varlık Veri Analizi ve Kalite Paneli",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RENK PALETİ ---
COLORS = {
    'primary': '#0052cc',
    'secondary': '#172B4D',
    'danger': '#FF5630',
    'warning': '#FFAB00',
    'success': '#36B37E',
    'background': '#F4F5F7'
}

# --- DOSYA YOLLARI ---
BASE_DIR = "data"
# pof.py yapısına göre:
INPUT_DIR = os.path.join(BASE_DIR, "girdiler")
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "ara_ciktilar")
OUTPUT_DIR = os.path.join(BASE_DIR, "sonuclar")

# --- YARDIMCI: COLUMN MAPPING ---
# pof.py çıktısındaki İngilizce/Teknik isimleri Dashboard'un beklediği isimlere çevirir
COLUMN_MAP_FAULT = {
    "started at": "Ariza_Baslangic_Zamani",
    "ended at": "Ariza_Bitis_Zamani",
    "Süre_Dakika": "Sure_Dakika"
}

COLUMN_MAP_FEAT = {
    "Tref_Yas_Gun": "ekipman_yasi_gun",
    "Fault_Count": "ariza_sayisi_toplam",
    # duration_days zaten var
    # event zaten var
}

# --- VERİ YÜKLEME FONKSİYONLARI ---
@st.cache_data
def load_data():
    data_dict = {}

    # ---------------------------------------------------------
    # 1. ÖZELLİK SETİ (Feature Matrix) - ozellikler_pof.csv
    # ---------------------------------------------------------
    # pof.py Line 1239'da bu isimle kaydediliyor
    feat_path = os.path.join(INTERMEDIATE_DIR, "ozellikler_pof.csv")

    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path)

        # Sütun İsimlerini Uyumlulaştır
        df_feat = df_feat.rename(columns=COLUMN_MAP_FEAT)

        # Tarih formatlarını düzelt
        date_cols = [col for col in df_feat.columns if 'Tarih' in col or 'Zaman' in col]
        for col in date_cols:
            df_feat[col] = pd.to_datetime(df_feat[col], errors='coerce')

        # ✅ EK VERİ KAYNAKLARI: Koordinat, Müşteri, Bakım verilerini ham dosyalardan ekle
        try:
            raw_dfs = []
            for filename in ['ariza_final.xlsx', 'saglam_final.xlsx']:
                raw_path = os.path.join(INPUT_DIR, filename)
                if os.path.exists(raw_path):
                    df_raw = pd.read_excel(raw_path)
                    # ID sütununu cbs_id olarak normalize et
                    if 'ID' in df_raw.columns:
                        df_raw = df_raw.rename(columns={'ID': 'cbs_id'})

                    # Gerekli sütunları seç
                    needed_cols = ['cbs_id', 'KOORDINAT_X', 'KOORDINAT_Y', 'total customer count',
                                   'Bakım Sayısı', 'Son Bakımdan İtibaren Geçen Gün Sayısı']
                    available_cols = [c for c in needed_cols if c in df_raw.columns]

                    if 'cbs_id' in available_cols:
                        raw_dfs.append(df_raw[available_cols])

            # Tüm ham verileri birleştir
            if raw_dfs:
                df_raw_all = pd.concat(raw_dfs, ignore_index=True).drop_duplicates(subset='cbs_id')

                # cbs_id formatını normalize et
                df_feat['cbs_id'] = df_feat['cbs_id'].astype(str).str.lower().str.strip()
                df_raw_all['cbs_id'] = df_raw_all['cbs_id'].astype(str).str.lower().str.strip()

                # Merge et
                df_feat = df_feat.merge(df_raw_all, on='cbs_id', how='left')
        except Exception as e:
            st.warning(f"⚠️ Ek veri kaynakları yüklenemedi: {e}")

        data_dict['features'] = df_feat
    else:
        st.error(f"⚠️ Özellik dosyası bulunamadı: {feat_path}. Lütfen önce pof.py'yi çalıştırın.")
        st.stop()

    # ---------------------------------------------------------
    # 2. HAM ARIZA VERİSİ - fault_events_clean.csv
    # ---------------------------------------------------------
    fault_path = os.path.join(INTERMEDIATE_DIR, "fault_events_clean.csv")
    if os.path.exists(fault_path):
        df_fault = pd.read_csv(fault_path)

        # İsim eşleştirme
        df_fault = df_fault.rename(columns=COLUMN_MAP_FAULT)

        # ✅ CRITICAL: cbs_id formatını normalize et (df_feat ile uyumlu olması için)
        if 'cbs_id' in df_fault.columns:
            df_fault['cbs_id'] = df_fault['cbs_id'].astype(str).str.lower().str.strip()

        # Kritik: Ariza_Baslangic_Zamani datetime olmalı
        if 'Ariza_Baslangic_Zamani' in df_fault.columns:
            df_fault['Ariza_Baslangic_Zamani'] = pd.to_datetime(df_fault['Ariza_Baslangic_Zamani'], errors='coerce')

        # Süre Saat hesabı (Eğer yoksa dakikadan türet)
        if 'Sure_Dakika' in df_fault.columns and 'Sure_Saat' not in df_fault.columns:
            df_fault['Sure_Saat'] = df_fault['Sure_Dakika'] / 60.0

        data_dict['faults'] = df_fault
    
    # ---------------------------------------------------------
    # 3. KRONİK ANALİZ SONUÇLARI
    # ---------------------------------------------------------
    # pof.py kronik veriyi ayrı bir dosyaya değil, ozellikler_pof.csv içine gömüyor.
    # O yüzden df_feat içinden filtreleyerek oluşturacağız.
    if 'features' in data_dict:
        df = data_dict['features']
        # Kronik bayrağı veya skorları varsa al
        chronic_cols = ['cbs_id', 'Ekipman_Tipi', 'Ilce', 'Chronic_Flag', 'Ariza_Sayisi_90g', 
                       'Chronic_Rate_Yillik', 'MTBF_Bayes_Gun']
        # Sadece var olan kolonları seç
        existing_cols = [c for c in chronic_cols if c in df.columns]
        data_dict['chronic'] = df[existing_cols].copy()
    
    return data_dict

# --- VERİYİ ÇAĞIR ---
try:
    data = load_data()
    df = data['features'] # Ana dataframe (Feature Matrix)
    df_faults = data.get('faults')
    df_chronic = data.get('chronic')
except Exception as e:
    st.error(f"Veri yükleme işlemi sırasında beklenmedik hata: {str(e)}")
    st.stop()

# --- SIDEBAR FİLTRELERİ ---
st.sidebar.title("🛠️ Veri Filtreleri")
st.sidebar.markdown("Analiz kapsamını daraltmak için filtreleri kullanın.")

# Global filtre sıfırlama butonu
if st.sidebar.button("🔄 Tüm Filtreleri Sıfırla", type="secondary"):
    st.session_state.selected_districts = []
    st.session_state.selected_types = []
    st.session_state.selected_brands = []
    st.rerun()

st.sidebar.markdown("---")

# 1. İlçe Filtresi
if 'Ilce' in df.columns:
    districts = sorted(df['Ilce'].dropna().astype(str).unique().tolist())

    # Hızlı seçim butonları
    col_btn_d1, col_btn_d2 = st.sidebar.columns(2)
    with col_btn_d1:
        if st.button("✓ Tümünü Seç", key="btn_select_all_districts"):
            st.session_state.selected_districts = districts
    with col_btn_d2:
        if st.button("✗ Temizle", key="btn_clear_all_districts"):
            st.session_state.selected_districts = []

    # Multiselect
    default_districts = st.session_state.get('selected_districts', [])
    selected_districts = st.sidebar.multiselect(
        "📍 Bölge / İlçe",
        districts,
        default=default_districts
    )

    # Seçimi session_state'e kaydet
    st.session_state.selected_districts = selected_districts
else:
    selected_districts = []

# 2. Ekipman Tipi Filtresi
if 'Ekipman_Tipi' in df.columns:
    types = sorted(df['Ekipman_Tipi'].dropna().astype(str).unique().tolist())

    # Hızlı seçim butonları
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("✓ Tümünü Seç", key="btn_select_all"):
            st.session_state.selected_types = types
    with col_btn2:
        if st.button("✗ Temizle", key="btn_clear_all"):
            st.session_state.selected_types = []

    # Multiselect - session_state'ten default değer al
    default_types = st.session_state.get('selected_types', [])
    selected_types = st.sidebar.multiselect(
        "⚙️ Ekipman Tipi",
        types,
        default=default_types
    )

    # Seçimi session_state'e kaydet
    st.session_state.selected_types = selected_types
else:
    selected_types = []

# 3. Marka Filtresi
if 'Marka' in df.columns:
    brands = sorted(df['Marka'].dropna().astype(str).unique().tolist())

    # Hızlı seçim butonları
    col_btn3, col_btn4 = st.sidebar.columns(2)
    with col_btn3:
        if st.button("✓ Tümünü Seç", key="btn_select_all_brands"):
            st.session_state.selected_brands = brands
    with col_btn4:
        if st.button("✗ Temizle", key="btn_clear_all_brands"):
            st.session_state.selected_brands = []

    # Multiselect - session_state'ten default değer al
    default_brands = st.session_state.get('selected_brands', [])
    selected_brands = st.sidebar.multiselect(
        "🏭 Marka",
        brands,
        default=default_brands
    )

    # Seçimi session_state'e kaydet
    st.session_state.selected_brands = selected_brands
else:
    selected_brands = []

# --- FİLTRELEME MANTIĞI ---
mask = pd.Series([True] * len(df))

if selected_districts:
    mask &= df['Ilce'].isin(selected_districts)
if selected_types:
    mask &= df['Ekipman_Tipi'].isin(selected_types)
if selected_brands:
    mask &= df['Marka'].astype(str).isin(selected_brands)

df_filtered = df[mask].copy()

# Filtre sonrası veri kontrolü
if len(df_filtered) == 0:
    st.warning("Seçilen filtrelere uygun veri bulunamadı. Lütfen filtreleri genişletin.")
    st.stop()

# --- ANA SAYFA ---
st.title("📊 Varlık Envanteri ve Veri Kalitesi Analiz Paneli")
st.markdown(f"""
Bu panel, **{len(df_filtered):,}** adet varlığın mevcut durumunu analiz eder.
**Not:** Burada gösterilen veriler model tahmini değil, `pof.py` tarafından işlenmiş gerçek saha verileridir.
""")
st.markdown("---")

# --- SEKMELER ---
tab_genel, tab_kalite, tab_ariza, tab_survival, tab_chronic = st.tabs([
    "📈 Genel Bakış", 
    "⚠️ Veri Kalitesi Karnesi", 
    "⚡ Arıza Karakteristiği", 
    "⏳ Yaşam Analizi (EDA)",
    "🔄 Tekrarlayan Sorunlar"
])

# =============================================================================
# TAB 1: GENEL BAKIŞ
# =============================================================================
with tab_genel:
    # KPI Kartları
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Toplam Varlık", f"{len(df_filtered):,}")
    
    with c2:
        # Ortalama Yaş (mapping sonrası 'ekipman_yasi_gun' oldu)
        if 'ekipman_yasi_gun' in df_filtered.columns:
            avg_age = (df_filtered['ekipman_yasi_gun'] / 365.25).mean()
            st.metric("Ortalama Yaş (Yıl)", f"{avg_age:.1f}")
        else:
            st.metric("Ortalama Yaş", "Veri Yok")
            
    with c3:
        # ✅ FIX: Arızalı Varlık Sayısı (event=1 olan satır sayısı)
        if 'event' in df_filtered.columns:
            faulty_count = (df_filtered['event'] == 1).sum()
            total_count = len(df_filtered)
            faulty_pct = 100 * faulty_count / total_count if total_count > 0 else 0
            st.metric(
                "Arızalı Varlık Sayısı",
                f"{faulty_count:,}",
                delta=f"{faulty_pct:.1f}% (Toplam içinde)"
            )
        elif 'ariza_sayisi_toplam' in df_filtered.columns:
            total_faults = df_filtered['ariza_sayisi_toplam'].sum()
            st.metric("Toplam Arıza Kaydı", f"{int(total_faults):,}")
    
    with c4:
        # Veri Doluluk Oranı
        completeness = 100 - (df_filtered.isnull().sum().sum() / (df_filtered.shape[0] * df_filtered.shape[1]) * 100)
        st.metric("Veri Doluluk Oranı", f"%{completeness:.1f}")

    st.markdown("---")

    # Grafikler
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Dağılım: Ekipman Tipi")
        fig_type = px.pie(df_filtered, names='Ekipman_Tipi', hole=0.4, 
                          color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_type, use_container_width=True)
        
    with col_right:
        st.subheader("Dağılım: Marka (Top 10)")
        if 'Marka' in df_filtered.columns:
            top_brands = df_filtered['Marka'].value_counts().head(10).reset_index()
            top_brands.columns = ['Marka', 'Adet']
            fig_brand = px.bar(top_brands, x='Adet', y='Marka', orientation='h',
                               text='Adet', color='Adet', color_continuous_scale='Blues')
            fig_brand.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_brand, use_container_width=True)
        else:
            st.info("Marka verisi bulunamadı.")

# =============================================================================
# TAB 2: VERİ KALİTESİ KARNESİ
# =============================================================================
with tab_kalite:
    st.header("🧐 Veri Kalitesi ve Eksiklik Analizi")
    
    # 1. Eksik Veri Heatmap
    missing_data = df_filtered.isnull().sum().reset_index()
    missing_data.columns = ['Kolon', 'Eksik_Sayisi']
    missing_data['Eksik_Orani'] = (missing_data['Eksik_Sayisi'] / len(df_filtered)) * 100
    missing_data = missing_data[missing_data['Eksik_Sayisi'] > 0].sort_values('Eksik_Orani', ascending=False)
    
    col_k1, col_k2 = st.columns([2, 1])
    
    with col_k1:
        if not missing_data.empty:
            fig_missing = px.bar(
                missing_data, 
                x='Eksik_Orani', 
                y='Kolon', 
                orientation='h',
                title="Kolon Bazlı Eksik Veri Oranı (%)",
                color='Eksik_Orani',
                color_continuous_scale='Reds',
                range_x=[0, 100]
            )
            fig_missing.add_vline(x=20, line_dash="dash", line_color="orange", annotation_text="Kritik Eşik %20")
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("🎉 Harika! Seçilen veri setinde hiç eksik veri yok.")
            
    with col_k2:
        st.subheader("Kritik Bulgular")
        if not missing_data.empty:
            high_missing = missing_data[missing_data['Eksik_Orani'] > 50]
            if not high_missing.empty:
                st.error(f"🚨 **{len(high_missing)} Kolonda** %50'den fazla veri eksik.")
                st.dataframe(high_missing[['Kolon', 'Eksik_Orani']].style.format({'Eksik_Orani': '{:.1f}%'}), hide_index=True)
            else:
                st.info("Eksik veriler yönetilebilir seviyede.")
                
    st.divider()
    
    # 2. Mantıksal Tutarsızlıklar
    st.subheader("Mantıksal Veri Kontrolü")
    
    check_cols = st.columns(3)
    
    # Yaş Kontrolü
    with check_cols[0]:
        col_age = 'ekipman_yasi_gun'
        if col_age in df_filtered.columns:
            neg_age = df_filtered[df_filtered[col_age] < 0]
            extreme_age = df_filtered[df_filtered[col_age] > (60 * 365)] # 60 yıl üstü
            
            st.write("**Yaş Verisi:**")
            if len(neg_age) > 0:
                st.warning(f"⚠️ {len(neg_age)} kayıtta negatif yaş tespit edildi.")
            else:
                st.success("✅ Negatif yaş kaydı yok.")
                
            if len(extreme_age) > 0:
                st.info(f"ℹ️ {len(extreme_age)} varlık 60 yaşından büyük.")
        else:
            st.warning("Yaş verisi bulunamadı.")
    
    # Varyans Kontrolü
    with check_cols[2]:
        st.write("**Bilgi İçeriği:**")
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            zero_var_cols = [col for col in numeric_cols if df_filtered[col].std() == 0]
            if zero_var_cols:
                st.warning(f"⚠️ {len(zero_var_cols)} kolonda hiç değişim yok (Sabit değer).")
                with st.expander("Sabit Kolonlar"):
                    st.write(zero_var_cols)
            else:
                st.success("✅ Sayısal kolonlarda varyasyon var.")

    st.divider()

    # 3. Kritik Veri Kontrolleri (v3'ten eklendi)
    st.subheader("Kritik Veri Kontrolleri")

    c1, c2, c3 = st.columns(3)

    # Koordinat Kontrolü
    with c1:
        st.write("**Koordinat Verisi:**")
        # KOORDINAT_X/Y (ham veriden) veya Boylam/Enlem veya X_KOORDINAT/Y_KOORDINAT
        lon_cols = ['KOORDINAT_X', 'X_KOORDINAT', 'Boylam', 'Longitude', 'x_koordinat']
        lat_cols = ['KOORDINAT_Y', 'Y_KOORDINAT', 'Enlem', 'Latitude', 'y_koordinat']

        lon_col = next((c for c in lon_cols if c in df_filtered.columns), None)
        lat_col = next((c for c in lat_cols if c in df_filtered.columns), None)

        if lon_col and lat_col:
            # Koordinat eksiklikleri (null veya 0 olanlar)
            no_coord = (df_filtered[lon_col].isna() | (df_filtered[lon_col] == 0) |
                       df_filtered[lat_col].isna() | (df_filtered[lat_col] == 0)).sum()
            if no_coord > 0:
                pct = 100 * no_coord / len(df_filtered)
                st.warning(f"⚠️ {no_coord} varlıkta koordinat yok ({pct:.1f}%)")
            else:
                st.success("✅ Koordinatlar tam.")
        else:
            st.info("ℹ️ Koordinat sütunları bulunamadı.")

    # Müşteri Verisi Kontrolü
    with c2:
        st.write("**Müşteri Verisi:**")
        cust_cols = ['total customer count', 'Musteri_Sayisi', 'musteri_sayisi']
        cust_col = next((c for c in cust_cols if c in df_filtered.columns), None)

        if cust_col:
            no_cust = (df_filtered[cust_col].fillna(0) == 0).sum()
            if no_cust > 0:
                pct = 100 * no_cust / len(df_filtered)
                st.warning(f"⚠️ {no_cust} varlıkta müşteri verisi 0 ({pct:.1f}%)")
            else:
                st.success("✅ Müşteri verisi tam.")
        else:
            st.info("ℹ️ Müşteri verisi sütunu bulunamadı.")

    # Bakım Verisi Kontrolü
    with c3:
        st.write("**Bakım Verisi:**")
        maint_cols = ['Bakım Sayısı', 'Bakim_Sayisi', 'bakim_sayisi']
        maint_col = next((c for c in maint_cols if c in df_filtered.columns), None)

        if maint_col:
            no_maint = df_filtered[maint_col].isna().sum()
            if no_maint > 0:
                pct = 100 * no_maint / len(df_filtered)
                st.warning(f"⚠️ {no_maint} varlıkta bakım verisi yok ({pct:.1f}%)")
            else:
                st.success("✅ Bakım verisi tam.")
        else:
            st.info("ℹ️ Bakım verisi sütunu bulunamadı.")

# =============================================================================
# TAB 3: ARIZA KARAKTERİSTİĞİ
# =============================================================================
with tab_ariza:
    st.header("⚡ Arıza Karakteristiği ve Trendler")
    
    if df_faults is not None:
        # Sadece filtrelenmiş varlıkların arızalarını al (cbs_id üzerinden join)
        relevant_faults = df_faults[df_faults['cbs_id'].isin(df_filtered['cbs_id'])].copy()
        
        if not relevant_faults.empty:
            # Zaman sütunu: Ariza_Baslangic_Zamani
            time_col = 'Ariza_Baslangic_Zamani'
            
            if time_col in relevant_faults.columns and pd.api.types.is_datetime64_any_dtype(relevant_faults[time_col]):
                # 1. Zaman Serisi
                relevant_faults['YearMonth'] = relevant_faults[time_col].dt.to_period('M').astype(str)
                trend = relevant_faults.groupby('YearMonth').size().reset_index(name='Ariza_Sayisi')
                
                fig_trend = px.line(trend, x='YearMonth', y='Ariza_Sayisi', markers=True,
                                    title="Aylık Arıza Sayısı Trendi",
                                    labels={'YearMonth': 'Ay', 'Ariza_Sayisi': 'Arıza Adedi'})
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # 2. Mevsimsellik ve Süre
                col_a1, col_a2 = st.columns(2)
                
                with col_a1:
                    relevant_faults['Ay'] = relevant_faults[time_col].dt.month_name()
                    seasonality = relevant_faults['Ay'].value_counts().reset_index()
                    seasonality.columns = ['Ay', 'Adet']
                    
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                   'July', 'August', 'September', 'October', 'November', 'December']
                    
                    fig_season = px.bar(seasonality, x='Ay', y='Adet', 
                                        category_orders={'Ay': month_order},
                                        title="Aylara Göre Arıza Dağılımı",
                                        color='Adet', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_season, use_container_width=True)
                    
                with col_a2:
                    if 'Sure_Saat' in relevant_faults.columns:
                        fig_dur = px.histogram(relevant_faults, x='Sure_Saat', nbins=50,
                                               title="Arıza Süresi Dağılımı (Saat)",
                                               log_y=True,
                                               labels={'Sure_Saat': 'Süre (Saat)'})
                        st.plotly_chart(fig_dur, use_container_width=True)
                    else:
                        st.info("Arıza süresi verisi (Sure_Saat) bulunamadı.")
            else:
                st.warning(f"Arıza zaman sütunu ({time_col}) hatalı veya datetime formatında değil.")
        else:
            st.warning("Seçilen filtreler için arıza kaydı bulunamadı.")
    else:
        st.info("Ham arıza verisi (fault_events_clean.csv) yüklenemedi.")

# =============================================================================
# TAB 4: YAŞAM ANALİZİ (EDA)
# =============================================================================
with tab_survival:
    st.header("⏳ Yaşam Analizi (Survival Analysis - EDA)")
    st.markdown("""
    Bu bölüm, **Kaplan-Meier istatistiksel yöntemi** ile varlıkların yaşa bağlı hayatta kalma olasılıklarını gösterir.
    (Veri Kaynağı: `duration_days` ve `event` sütunları)
    """)
    
    if 'duration_days' in df_filtered.columns and 'event' in df_filtered.columns:
        
        try:
            from lifelines import KaplanMeierFitter
            
            kmf = KaplanMeierFitter()
            
            # Veri çok büyükse örneklem al
            if len(df_filtered) > 5000:
                sample_data = df_filtered.sample(5000, random_state=42)
                st.caption(f"ℹ️ Performans için 5.000 kayıtlık rastgele örneklem kullanılıyor.")
            else:
                sample_data = df_filtered
            
            # Global Fit
            kmf.fit(sample_data['duration_days'], event_observed=sample_data['event'], label='Genel Ort.')
            
            # Grafik
            survival_df = kmf.survival_function_.reset_index()
            survival_df.columns = ['Gun', 'Olasilik']
            
            fig_km = go.Figure()
            fig_km.add_trace(go.Scatter(x=survival_df['Gun'], y=survival_df['Olasilik'], 
                                        mode='lines', name='Tüm Seçim',
                                        line=dict(color=COLORS['primary'], width=3)))
            
            # Kırılım (Ekipman Tipi)
            if len(sample_data['Ekipman_Tipi'].unique()) > 1:
                for eq_type in sample_data['Ekipman_Tipi'].unique():
                    subset = sample_data[sample_data['Ekipman_Tipi'] == eq_type]
                    if len(subset) > 50: 
                        kmf_sub = KaplanMeierFitter()
                        kmf_sub.fit(subset['duration_days'], event_observed=subset['event'])
                        sub_df = kmf_sub.survival_function_.reset_index()
                        sub_df.columns = ['Gun', 'Olasilik']
                        fig_km.add_trace(go.Scatter(x=sub_df['Gun'], y=sub_df['Olasilik'],
                                                    mode='lines', name=f"{eq_type}",
                                                    line=dict(dash='dot')))

            fig_km.update_layout(
                title="Kaplan-Meier Yaşam Eğrisi",
                xaxis_title="Geçen Süre (Gün)",
                yaxis_title="Hayatta Kalma Olasılığı P(T > t)",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_km, use_container_width=True)
            
        except ImportError:
            st.error("Lifelines kütüphanesi eksik. Lütfen `pip install lifelines` yapın.")
        except Exception as e:
            st.error(f"Analiz hatası: {str(e)}")
            
    else:
        st.warning("Gerekli kolonlar (`duration_days`, `event`) bulunamadı.")

# =============================================================================
# TAB 5: KRONİK SORUNLAR
# =============================================================================
with tab_chronic:
    st.header("🔄 Tekrarlayan (Kronik) Sorunlu Varlıklar")
    st.markdown("IEEE 1366 standartlarına göre veya yüksek arıza sıklığına sahip varlıklar.")
    
    chronic_assets = pd.DataFrame()
    
    # 1. Pipeline'dan gelen bayrağı kontrol et
    if 'Chronic_Flag' in df_filtered.columns:
        chronic_assets = df_filtered[df_filtered['Chronic_Flag'] == 1].copy()
    # 2. Yoksa manuel hesapla (Son 90 günde veri varsa veya toplam arızadan)
    elif 'Ariza_Sayisi_90g' in df_filtered.columns:
        chronic_assets = df_filtered[df_filtered['Ariza_Sayisi_90g'] >= 3].copy()
        
    if not chronic_assets.empty:
        col_c1, col_c2 = st.columns([1, 3])
        
        with col_c1:
            st.error(f"🚨 **{len(chronic_assets)}** Adet Kronik Varlık")
            if 'MTBF_Bayes_Gun' in chronic_assets.columns:
                avg_mtbf = chronic_assets['MTBF_Bayes_Gun'].mean()
                st.metric("Ortalama MTBF (Gün)", f"{avg_mtbf:.1f}")

        with col_c2:
            st.subheader("Kronik Varlık Listesi")
            
            # Gösterilecek kolonlar (İlçe yoksa koyma)
            cols_to_show = ['cbs_id', 'Ekipman_Tipi', 'Ilce']
            valid_cols_to_show = [c for c in cols_to_show if c in chronic_assets.columns]

            metrics = ['Ariza_Sayisi_90g', 'Chronic_Rate_Yillik', 'ariza_sayisi_toplam']
            
            final_cols = valid_cols_to_show + [c for c in metrics if c in chronic_assets.columns]
            
            st.dataframe(
                chronic_assets[final_cols].sort_values(
                    by=[c for c in metrics if c in chronic_assets.columns], 
                    ascending=False
                ),
                use_container_width=True
            )
    else:
        st.success("✅ Seçilen kriterlerde kronik (tekrarlayan) arızalı varlık tespit edilmedi.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Veri Analiz Modülü v2.0 | {datetime.now().year}")