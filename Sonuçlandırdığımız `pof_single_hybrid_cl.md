Sonuçlandırdığımız `pof_single_hybrid_clean_v2.py` betiğine ve başarılı üretim çalışmasına dayanarak, işte güncellenen **Teknik Dokümantasyonun (v2.0)** Türkçe versiyonu.

Bunu **`PROJE_DOKUMANTASYONU_v2.md`** olarak kaydedebilirsiniz. "Gerçek Arıza" mantığını, Küresel Yedek (Global Fallback) mekanizmasını ve sonuçların nasıl yorumlanacağını doğru bir şekilde yansıtmaktadır.

---

# PoF3 - Varlık Arıza Tahmin Sistemi

## Teknik Dokümantasyon & Metodoloji Rehberi (v2.0)

**Versiyon:** 2.0 (Temiz Üretim / Clean Production)
**Tarih:** Aralık 2025
**Pipeline Betiği:** `pof_single_hybrid_clean_v2.py`

---

## 1. Proje Özeti

**PoF3 (Arıza Olasılığı)** sistemi, elektrik dağıtım şebekesindeki fiziksel varlık arızalarının olasılığını tahmin etmek için tasarlanmış hibrit bir güvenilirlik analiz hattıdır.

### Kritik Stratejik Değişiklik (v2.0)

Önceki versiyonların aksine (kesintileri ve sigorta atıklarını tahmin etmeye çalışan), **v2.0 "Koruma Operasyonları" ile "Fiziksel Arızaları" kesin bir şekilde birbirinden ayırır**.

* **Amaç:** Gerçek varlık yaşlanmasını ve katastrofik arızaları tahmin etmek.
* **Hedef (Target):** `event = 1` sadece fiziksel arızalar için atanır (örn. "Trafo Arızası", "İletken Kopması").
* **Hariç Tutma:** Koruma operasyonları (örn. "Sigorta Atığı") hedef değişkenden çıkarılır ancak **Kronik Stres Özellikleri** (tahminleyici/predictor) olarak modele geri beslenir.

---

## 2. Sistem Mimarisi

Sistem, hem verisi bol varlıkları (Ayırıcılar gibi) hem de verisi az varlıkları (Trafolar gibi) aynı hat üzerinde işleyebilmek için **Tek Geçişli Hibrit Mimari (Single-Pass Hybrid Architecture)** kullanır.

```mermaid
graph TD
    A[Ham Veri: Arızalar & Varlıklar] --> B{Filtre: Gerçek Arıza mı?}
    B -- Evet (Hedef) --> C[Sağkalım Veri Tabanı]
    B -- Hayır (Sigorta Atığı) --> D[Kronik Özellikler]
    C & D --> E[Ana Veri Seti - Master]
    E --> F{Veri Yeterliliği Kontrolü}
    F -- Yeterli Veri (N>100) --> G[Ekipmana Özel Modeller]
    F -- Yetersiz Veri (Trafo/Pano) --> H[Küresel Yedek Model - Global Fallback]
    G & H --> I[Sağkalım Eğrileri (Cox/RSF)]
    I --> J[PoF Hesaplaması (12ay/24ay)]
    J --> K[Final Ensemble & Raporlama]

```

### Temel Bileşenler

1. **Sağkalım Modelleri (Ana Model):** Cox Orantılı Tehlikeler (CoxPH) ve Rastgele Sağkalım Ormanları (RSF), tüm varlık ömrü boyunca riski tahmin eder.
2. **Küresel Yedek Model (Global Fallback):** Nadir arıza yapan ekipmanlar (örn. Trafolar) için *tüm* varlık havuzundan öğrenilen genel bir "yaşlanma eğrisi" oluşturur.
3. **Kronik Skorlama:** Tekrarlayan koruma cihazı operasyonlarına dayalı bir "Stres Skoru" hesaplayan ayrı bir modüldür (IEEE 1366 mantığına benzer).

---

## 3. Metodoloji & Mantık

### 3.1 Arıza Tanımı (Failure Definition)

Pipeline, girdi verilerini katı bir "neden kodu beyaz listesi" (whitelist) kullanarak otomatik olarak filtreler.

| Kategori | Durum | Örnekler | Gerekçe |
| --- | --- | --- | --- |
| **Fiziksel Arıza** | **HEDEF (Target)** | *Trafo Arızası, İletken Kopması, Direk Kırılması* | Varlık tamir veya değişim gerektirir. |
| **Koruma Operasyonu** | **HARİÇ** | *Sigorta Atığı, Termik Açması, TMS Açması* | Varlık şebekeyi korumak için doğru çalışmıştır. |
| **Dışsal/Diğer** | **HARİÇ** | *3. Şahıs Hasarı, Planlı Bakım* | İçsel yaşlanma arızası değildir. |

### 3.2 Ekipman Sınıflandırması (Stratification)

Pipeline, her ekipman tipi için hangi modelleme stratejisinin kullanılacağına otomatik karar verir:

* **Özel Model (Tier 1):** **>100 örnek** ve **>30 arıza** geçmişi olan varlıklar için.
* *Örnek:* Ayırıcı.
* *Sonuç:* Yüksek hassasiyetli, o ekipmana özel tahminler.


* **Küresel Yedek Model (Tier 2):** Yetersiz geçmişe sahip varlıklar için.
* *Örnek:* Trafo, Pano.
* *Sonuç:* Filo ortalamasından türetilen baz risk profili.



### 3.3 "Bebek Ölümleri" Filtresi (ML Skip)

Loglarda `[ML] Skipping 12ay: insufficient positives` göreceksiniz.

* **Sebep:** Makine Öğrenmesi sınıflandırıcıları (XGBoost), belirli bir ufuk içinde (örn. 1 yaşından önce) arıza yapan "Eğitim Örnekleri"ne ihtiyaç duyar.
* **Gerçeklik:** Dağıtım varlıkları nadiren çok genç yaşta arıza yapar. Çoğu arıza 20+ yaşta gerçekleşir.
* **Çözüm:** Pipeline, aşırı uyumlamayı (overfitting) önlemek için bu sınıflandırıcıları bilerek atlar. **PoF (Arıza Olasılığı), yalnızca Sağkalım Modellerinden (Cox/RSF) türetilir.** Bu modeller, "genç yaşta arıza" verisine ihtiyaç duymadan, yaşlı varlıklar için riski matematiksel olarak hesaplayabilir.

---

## 4. Çalıştırma Adımları (Execution Steps)

### Adım 1: Veri Yükleme & Temizleme

* `ariza_final.xlsx` ve `saglam_final.xlsx` dosyalarını yükler.
* **Otomatik Temizleme:** Süreleri dakikaya çevirir, ID'leri normalleştirir, Türkçe tarih formatlarını ayrıştırır (parse).

### Adım 2: Özellik Mühendisliği (Feature Engineering)

* **Yapısal:** Gerilim Seviyesi, Marka, Kurulum Tarihi.
* **Zamansal:** Yaş (`Tref_Yas_Gun`), Mevsimsellik.
* **Kronik:** `Chronic_Index` hesaplar (sigorta atıklarının ağırlıklı frekansı).
* *Not:* Bir Trafo hiç fiziksel arıza yapmasa bile, yüksek Kronik İndeks onun risk skorunu artıracaktır.



### Adım 3: Model Eğitimi (The "Continue" Fix)

* Her ekipman tipi için döngü çalışır.
* `stats['n_events']` kontrol edilir.
* **Veri Yetersizse:** Küresel Model kullanılır → Tahminler üretilir → **`continue`** (özel model eğitimi atlanır).
* **Veri Yeterliyse:** Ekipmana özel Cox/RSF/Weibull modelleri eğitilir.

### Adım 4: Güvenlik Kontrolleri

* **VIF Filtreleme:** Sonsuz varyans enflasyon faktörüne sahip (örn. tek bir gerilim seviyesi varsa) özellikleri otomatik olarak düşürür.
* **Sabit Sütun Düşürme:** Cox regresyonunda "Singular Matrix" hatalarını önler.

---

## 5. Çıktı Dosyaları ve Yorumlama

Tüm çıktılar `data/sonuclar/` klasörüne kaydedilir.

| Dosya Adı | Açıklama | Kullanılacak Ana Sütunlar |
| --- | --- | --- |
| **`pof_predictions_final.csv`** | **ANA RAPOR.** Tüm varlıklar için birleştirilmiş sonuçlar. | `cox_pof_12ay`, `rsf_pof_12ay`, `Health_Score`, `Chronic_Index` |
| `pof_Ayırıcı.csv` | Ayırıcılar için detaylı sonuçlar. | `rsf_pof_12ay` (En yüksek doğruluk) |
| `model_input_data_full.csv` | Hata ayıklama dosyası. Eğitim için kullanılan matris. | Tüm özellikler + `event` + `duration_days` |
| `marka_analysis.csv` | Marka güvenilirliğinin istatistiksel dökümü. | `Failure_Rate`, `Median_Age` |

### Tahminleri Nasıl Okumalısınız?

| Ekipman | Kullanılan Model | Güvenilirlik | Yorumlama Rehberi |
| --- | --- | --- | --- |
| **Ayırıcı** | **Özel (RSF/Cox)** | ⭐⭐⭐⭐⭐ (Yüksek) | 3ay/6ay/12ay sütunlarını güvenle kullanın. Model performansı gayet iyi (Uyum Skoru ~0.65). |
| **Hat** | **Özel (Cox)** | ⭐⭐⭐ (Orta) | Risk ikilidir (0 ya da 1). Yüksek skorlar "anlık riski" gösterir ancak zaman ufku (3 ay vs 6 ay) çok ayırt edici olmayabilir. |
| **Trafo** | **Küresel Yedek** | ⭐⭐ (Baz Seviye) | Skorlar filo genelinde benzer olacaktır. Bakım önceliği için **Kronik İndeks** sütununu kullanın. |
| **Sigorta** | **Özel** | ⭐ (Düşük) | "Arızalar" tamamen rastgeledir. Tahminleyici bakım için kullanmayın, sadece stok planlaması için kullanın. |

---

## 6. Sorun Giderme (Troubleshooting)

### Yaygın Loglar ve Anlamları

**`[ML] Skipping 12ay: insufficient positives (11)`**

* **Durum:** Normal / Beklenen.
* **Anlamı:** İkili sınıflandırıcı eğitmek için yeterli sayıda "genç" arıza yok. Sistem bunun yerine Sağkalım Modellerini kullanıyor.

**`[VIF] Dropping Gerilim_Seviyesi (VIF=inf)`**

* **Durum:** Normal / Sağlıklı.
* **Anlamı:** Sistem gereksiz/tekrarlayan veriyi (örn. sadece tek bir gerilim seviyesi var) tespit etti ve çökmemek için o sütunu sildi.

**`[Trafo] Global Cox failed: ...`**

* **Durum:** Yönetildi (Handled).
* **Anlamı:** Küresel model bile veri şekliyle (shape) zorlandı. Sistem bu varlık için baz bir tahmin (Health=100) atadı ancak varlığı işaretledi.

### Kritik Varlık Tanımı

* **Mevcut Kod Mantığı:** `Health_Score < 40` (PoF > %60).
* **Tavsiye:** Nihai raporda veya gösterge panelinde (dashboard), "Kritik" tanımını kodlanmış 40 eşiği yerine **Risk Skorlarının En Yüksek %5'lik Dilimi** olarak belirleyin. Dağıtım varlıkları pratikte nadiren %60 arıza olasılığına ulaşmadan önce değiştirilirler.

---

**İletişim:** Teknik Analitik Ekibi
**Bakım Sorumlusu:** Begüm Orhan
**Lisans:** Kurumsal İç Kullanım (Internal Enterprise Use)