import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

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

    # Ensemble Onarımı
    # --- EKLE: Sütun Onarma Mantığı ---
    if "PoF_Ensemble_12Ay" not in df.columns:
        # pof.py çıktısındaki pof sütunlarını bul 
        cols = [c for c in df.columns if "12ay" in c.lower() and "pof" in c.lower()]
        df["PoF_Ensemble_12Ay"] = df[cols].mean(axis=1) if cols else 0.0

    # --- YENİ: Ara Dosyaları Yükle ---
    eda_raw_path = os.path.join(INTERMEDIATE_DIR, "fault_events_clean.csv")
    eda_feat_path = os.path.join(INTERMEDIATE_DIR, "model_input_data_full.csv")
    
    df_raw = pd.read_csv(eda_raw_path) if os.path.exists(eda_raw_path) else None
    df_feat = pd.read_csv(eda_feat_path) if os.path.exists(eda_feat_path) else None
    
    # Marka ve Bakım
    marka_path = os.path.join(OUTPUT_DIR, "marka_analysis.csv")
    bakim_path = os.path.join(OUTPUT_DIR, "bakim_analysis.csv")
    df_marka = pd.read_csv(marka_path) if os.path.exists(marka_path) else None
    df_bakim = pd.read_csv(bakim_path) if os.path.exists(bakim_path) else None
    
    return df, df_marka, df_bakim, df_raw, df_feat

df_all, df_marka, df_bakim, df_raw, df_feat = load_pof_data()


# --- SIDEBAR (FİLTRELER) ---
st.sidebar.title("🔍 Şebeke Filtreleri")
districts = df_all['Ilce'].unique().tolist() if 'Ilce' in df_all.columns else ["Tümü"]
selected_district = st.sidebar.multiselect("Bölge / İlçe", districts, default=districts)

eq_types = df_all['Ekipman_Tipi'].unique().tolist()
selected_types = st.sidebar.multiselect("Ekipman Tipi", eq_types, default=eq_types)

risk_classes = ['KRİTİK', 'KRİTİK (KRONİK)', 'YÜKSEK', 'ORTA', 'DÜŞÜK']
selected_risks = st.sidebar.multiselect("Risk Sınıfı", risk_classes, default=risk_classes)

# Filtreleme İşlemi
mask = (df_all['Ekipman_Tipi'].isin(selected_types)) & \
       (df_all['Risk_Sinifi'].isin(selected_risks))
if 'Ilce' in df_all.columns:
    mask &= (df_all['Ilce'].isin(selected_district))

filtered_df = df_all[mask]

# --- ANA PANEL ---
st.title("⚡ PoF3 Varlık Yönetimi Karar Destek Sistemi")
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
    st.subheader("🏭 Marka ve Bakım Performans Karnesi")
    if df_marka is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Marka Bazlı Göreceli Risk (1.0 = Ortalama)**")
            fig_marka = px.bar(df_marka.sort_values('Relative_Risk', ascending=False).head(10), 
                               x='Marka', y='Relative_Risk', color='Relative_Risk',
                               color_continuous_scale='Reds', labels={'Relative_Risk': 'Risk Çarpanı'})
            st.plotly_chart(fig_marka, use_container_width=True)
        with c2:
            st.write("**Bakım Sayısının Arıza Oranına Etkisi**")
            if df_bakim is not None:
                fig_bakim = px.line(df_bakim, x='Bakim_Bin', y='Failure_Rate', markers=True,
                                    title="Bakım Arttıkça Arıza Oranı Değişimi")
                st.plotly_chart(fig_bakim, use_container_width=True)
    else:
        st.info("Marka ve bakım analiz verisi bulunamadı.")

with tab4:
    st.subheader("🧪 Model Doğrulama (Backtesting) ve Teşhis")
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