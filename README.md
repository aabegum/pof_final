PoF3 – Varlık Arıza Riski Analizi

(Probability of Failure – Göreceli Risk Yaklaşımı)

1. Amaç ve Kapsam

Bu çalışma, elektrik dağıtım şebekesindeki varlıkların (Trafo, Ayırıcı, Hat, Sigorta, vb.) gelecekte arıza yaşama risklerini istatistiksel yöntemlerle göreceli olarak sıralamak amacıyla geliştirilmiştir.

Modelin temel hedefi:

“Hangi varlıklar, benzerlerine kıyasla daha yüksek arıza riski taşımaktadır?”

Bu analiz;

bakım önceliklendirme,

saha denetim planlaması,

CAPEX/OPEX karar destek süreçleri

için kullanılmak üzere tasarlanmıştır.

2. Temel Kavramlar (Yanlış Anlaşılmaması İçin)
2.1. Arıza Kaydı ≠ Fiziksel Arıza

EDAŞ sistemlerinde yer alan tüm arıza/kesinti kayıtları fiziksel ekipman arızasını temsil etmez.

Bu nedenle çalışmada:

sigorta atması,

pano kol sigortası,

operasyonel açma-kapamalar,

dış etken kaynaklı kesintiler

modelden hariç tutulmuştur.

📌 Sadece gerçek ekipman arızalarını temsil eden kayıtlar analiz kapsamına alınmıştır.

2.2. Model “Ne Zaman” Değil, “Hangisi” Sorusunu Yanıtlar

Bu model:

“Bu trafo yarın arızalanır mı?” sorusuna cevap vermez.

“Bu trafo, diğer trafolara göre daha mı risklidir?” sorusunu yanıtlar.

Dolayısıyla model çıktıları:

mutlak tarih tahmini değil,

göreceli risk sıralamasıdır.

3. Kullanılan Yöntemler (Özet)
3.1. Sağkalım Analizi (Survival Analysis)

Modelin omurgasını şu yöntemler oluşturur:

Cox Oransal Tehlike Modeli

Weibull Parametrik Model

Random Survival Forest (RSF)

Bu yöntemler sayesinde:

ekipman yaşı,

kronik arıza davranışı,

gözlem süresi farklılıkları

istatistiksel olarak doğru şekilde ele alınmıştır.

3.2. Gecikmeli Giriş (Delayed Entry)

Veri seti 2021 yılından başladığı için, 2021 öncesi kurulmuş ekipmanların geçmişi kısmen bilinmemektedir.

Bu durum, Gecikmeli Giriş (Delayed Entry) yaklaşımı ile çözülmüştür.

Anlamı şudur:

“Bir ekipmanın 2021 öncesinde arızalanıp arızalanmadığı bilinmiyor; ancak 2021’den sonra hayatta kaldığı biliniyor.”

Bu yöntem, eski ekipmanların riskinin yapay olarak düşük görünmesini engeller.

3.3. Kronik Arıza Analizi

Son 90 gün içinde:

sık arızalanan,

tekrar eden problem gösteren

ekipmanlar kronik olarak işaretlenmiştir.

Kronik ekipmanlar:

sağlık skorunda cezalandırılır,

risk sınıfı otomatik olarak yükseltilir.

4. Sağlık Skoru (Health Score) Nasıl Hesaplanır?
4.1. Mutlak Olasılık Neden Kullanılmıyor?

Fiziksel ekipman arızaları nadir olaylardır.
Bu nedenle mutlak arıza olasılıkları genellikle çok düşüktür (%0.1 – %1 gibi).

Bu durum, tüm ekipmanların “çok sağlıklı” görünmesine yol açar.

📌 Bu yüzden mutlak olasılık değil, göreceli risk kullanılmıştır.

4.2. Göreceli Risk (Percentile Yaklaşımı)

Her ekipman, kendi türü içindeki diğer ekipmanlarla karşılaştırılır.

Örnek:

Bir trafo, diğer trafolar arasında %95’lik risk dilimindeyse KRİTİK kabul edilir.

Bu, mutlak arıza olasılığı düşük olsa bile geçerlidir.

5. Risk Sınıfları (EDAŞ Uyumlu)
Risk Sınıfı	Tanım	İstatistiksel Karşılık	Önerilen Aksiyon
KRİTİK	Acil İlgi Gerektirir	En riskli %5	🔴 Derhal saha kontrolü / yenileme planı
YÜKSEK	Yakın Takip	Sonraki %15	🟠 Bakım sıklığı artırılmalı
ORTA	Standart Risk	Sonraki %30	🟡 Rutin bakım
DÜŞÜK	Sağlıklı	En iyi %50	🟢 Müdahale gerekmez

📌 “KRİTİK” etiketi yarın arıza olacak anlamına gelmez.
📌 “KRİTİK”, benzerleri arasında en riskli anlamına gelir.

6. Model Sonuçlarının Doğru Kullanımı
Yapılması Gerekenler ✅

Risk sınıflarını önceliklendirme amacıyla kullanmak

KRİTİK ve YÜKSEK varlıkları saha planına almak

Marka, bakım ve kronik analizlerini destekleyici bilgi olarak görmek

Yapılmaması Gerekenler ❌

“Bu varlık kesin arızalanacak” yorumu yapmak

Tek bir varlık için tarih tahmini istemek

Sağlık skorunu mutlak bir ölçü gibi kullanmak

7. Veri Kısıtları ve Notlar

Analiz dönemi: 2021 – 2025

2021 öncesi arıza geçmişi bilinmemektedir.

Sonuçlar, mevcut veri kalitesi ile sınırlıdır.

Model, zamanla yeni verilerle yeniden eğitilmelidir.

8. Sonuç

Bu çalışma, EDAŞ varlık yönetimi süreçlerinde:

sezgisel kararları sayısallaştıran,

riskleri görünür hale getiren,

bakım ve yatırım kararlarını destekleyen

karar destek sistemi olarak tasarlanmıştır.

Amaç:

“Arızayı kesin tahmin etmek değil, en doğru yere bakmayı sağlamak.”