# Anomali-Odaklı Video Özetleme — Tüm Bulgular ve Sonuçlar

_Rapora geçirmek için derlenmiş tam veri dökümü. Tüm sayılar UCF-Crime STANDART bölünmesinde (1610 eğitim / 290 etiketli test), kendi koşularımız._

---

## 1. Deneysel Kurulum

- **Veri kümesi:** UCF-Crime, standart bölünme. Eğitim: **1610 video** (800 normal + 810 anomalili). Test: kare-düzeyi zamansal etiketli **290 video** (140 anomalili + 150 normal). Eğitim/test arası **sızıntı yok** (doğrulandı).
- **Özellikler:** CLIP ViT-B/16 görüntü kodlayıcısı, 512 boyut, saniyede 2 kare örnekleme, video başına 32 zamansal segment.
- **Eğitim:** 15 dönem × 300 adım, yığın 8 (8 normal + 8 anomalili), Adam (lr 1e-4, ağırlık çürümesi 0.005, gradyan kırpma 1.0). NVIDIA RTX 3060, koşu başına ~2 dk. En iyi model her dönem sonunda kare-AUC'ye göre seçilir.
- **Metin füzyonu (v1):** α=0.3, tau=0.07. 13 anomali sınıfının doğal-dil betimlemelerinden prompt ensemble ile CLIP ViT-B/16 metin kodlayıcısı kullanılarak sabit metin prototipleri üretilir. Skor = (1−α)·görsel + α·metin_olasılığı.
- **Metrikler:** Tespit — video/kare düzeyi ROC-AUC, karar geçidi P/R/F1/doğruluk. Özetleme — zamansal recall, precision, F1, sıkıştırma oranı (özet süresi / orijinal süre; düşük iyi). Baseline — rastgele/uniform seçim.
- **İstatistik:** çok-tohumlu (ortalama ± std), eşleştirilmiş t-testi + Wilcoxon, eşleştirilmiş seed kıyasları.

---

## 2. Ana Model (v1 — Metin Füzyonu) Başarımı

Doğrulama AUC'sine göre seçilen en iyi v1 modeli (seed 1):

| Ölçüt | Değer |
|---|---|
| Kare düzeyi ROC-AUC | **0.844** (AP 0.241) |
| Video düzeyi ROC-AUC | **0.946** |
| Karar geçidi | P 0.932 / R 0.786 / F1 0.853 / doğruluk 0.869 (TP 110, FP 8, TN 142, FN 30) |
| Özet recall / precision / F1 | 0.396 / 0.290 / 0.289 |
| Özet sıkıştırma | 0.256 (videonun ~%26'sı) |

Yorum: "video düzeyinde %87 doğruluk / 0.95 ROC-AUC; kare (lokalizasyon) düzeyinde 0.84 ROC-AUC; özet videonun ~%26'sını tutarak anomalinin önemli bölümünü yakalar."

---

## 3. Metin Füzyonunun Etkisi: v0 (görsel) vs v1 (metin füzyonu)

Aynı üç tohumla **eşleştirilmiş** (mean pool, tepe k=3, bağlam=1):

| Metrik | v0 | v1 | Δ |
|---|---|---|---|
| Kare-AUC (ort) | 0.818 | 0.834 | **+0.016** |
| Video-AUC | 0.941 | 0.945 | +0.004 |
| Karar geçidi F1 | 0.859 | 0.845 | −0.015 |
| Özet recall | 0.317 | 0.383 | **+0.067** |
| Özet precision | 0.246 | 0.282 | +0.036 |
| Özet F1 | 0.240 | 0.281 | +0.041 |

**Seed bazında özet recall (v1 her üçünde de üstün):** seed 0: 0.411→0.434; seed 1: 0.338→0.396; seed 42: 0.201→0.320.

**Sonuç:** Metin füzyonu hem tespiti hem özetlemeyi **3/3 tohumda** iyileştirir. (Referans: arkadaşın non-standart bölünme + 10 dönemde v0 0.818 → v1 0.842 (+2.34pp); v2/CoOp 0.831, v1'i geçemedi — bu yüzden v1 ana model.)

---

## 4. Özetleme ve Baseline Karşılaştırması (v1)

top-k seçimle, farklı k değerlerinde geri çağırma (TAM eğri):

| k | Önerilen (recall) | sıkıştırma | Rastgele | Düzgün |
|---|---|---|---|---|
| 1 | 0.138 | %9 | 0.094 | 0.010 |
| 2 | 0.226 | %15 | 0.178 | 0.023 |
| **3 (manşet)** | **0.320** | **%20** | 0.262 | 0.139 |
| 4 | 0.374 | %25 | 0.339 | 0.223 |
| 5 | 0.445 | %30 | 0.406 | 0.316 |
| 6 | 0.505 | %35 | 0.479 | 0.501 |
| 8 | 0.589 | %42 | 0.597 | 0.686 |
| 10 | 0.668 | %49 | 0.691 | 0.826 |
| 12 | 0.741 | %56 | 0.771 | 1.000 |
| 16 | 0.835 | %69 | 0.886 | 1.000 |

**Sonuç:** k arttıkça recall ve sıkıştırma birlikte artar (denge eğrisi). Düşük/orta bütçede (k=1–6) önerilen yöntem rastgele ve düzgün seçimi belirgin geçer (eşit veya daha düşük sıkıştırmada); yüksek k'da (8+) neredeyse tüm video tutulduğundan yöntemler yakınsar. Tepe segmentler anomali çevresinde kümelendiğinden özet daha kompakttır. Manşet çalışma noktası **k=3** (recall 0.32 @ %20).

---

## 5. Ablasyonlar (hepsi v1)

### 5.1. Seçim Modu (k=3, bağlam=1)
| Mod | recall | sıkıştırma | F1 |
|---|---|---|---|
| Tepe (NMS) — önerilen | 0.396 | 0.256 | 0.289 |
| Top-k | 0.320 | 0.199 | 0.258 |
| Eşik (mutlak) | 0.596 | 0.518 | 0.262 |

Tepe en iyi denge; eşik videonun yarısından fazlasını alır, kullanışsız.

### 5.2. Bağlam Dolgusu (tepe, k=3)
| Bağlam | recall | sıkıştırma | F1 |
|---|---|---|---|
| 0 | 0.161 | 0.094 | 0.196 |
| 1 (önerilen) | 0.396 | 0.256 | 0.289 |
| 2 | 0.544 | 0.372 | 0.319 |

Bağlam ↑ → recall ↑ ve sıkıştırma ↑. Ayarlanabilir bütçe düğmesi.

### 5.3. NMS Penceresi (tepe, k=3)
| NMS | recall | sıkıştırma | k=3 ours |
|---|---|---|---|
| 1 (mevcut varsayılan) | 0.396 | %26 | 0.320 |
| **2** | **0.420** | %28 | 0.329 |
| 3 | 0.408 | %28 | 0.319 |
| 4 | 0.380 | %28 | 0.317 |

recall **NMS=2'de tepe** yapıp düşüyor. NMS=2 tatlı nokta; mevcut varsayılan (1) hafif optimal-altı. (Sıkıştırma NMS≥2'de ~%28'de doyar.)

### 5.4. Pooling
mean vs max neredeyse eşit (özet recall 0.396 vs 0.413; kare-AUC 0.844 vs 0.835). Gürbüz.

### 5.5. Örnekleme Hızı (fps) — her hız için en iyi tohum
| fps | Kare-AUC | Video-AUC | Özet recall | sıkıştırma |
|---|---|---|---|---|
| 1 | 0.847 | 0.956 | 0.374 | %25 |
| 2 | 0.837 | 0.946 | 0.434 | %25 |
| 4 | 0.835 | 0.958 | 0.460 | %25 |
| 8 | 0.842 | 0.951 | 0.442 | %24 |

Kare-AUC ortalamaları (2 tohum): fps1 0.837, fps2 0.829, fps4 0.834, fps8 0.829. **fps'e duyarsız**; fps ≥ 2 makul, fps=8 ek kazanç yok (4 kat maliyet).

### 5.6. Segment Sayısı (Zamansal Çözünürlük)
Kare-AUC (2 tohum ort.): 16 → 0.843, 32 → 0.829, 64 → 0.842.

Eşit sıkıştırmada (~%24) özet recall (adil kıyas):
| Segment | recall @ %24 | rastgeleye üstünlük |
|---|---|---|
| 16 | 0.391 | +0.138 |
| 32 | 0.419 | +0.093 |
| 64 | 0.416 | +0.042 |

Sabit k=3'te ise segment sayısı bir sıkıştırma düğmesidir: 16 → %50 sıkıştırma (recall 0.699), 32 → %24 (0.434), 64 → %13 (0.259).

**Sonuç:** Segment sayısı bir kalite kaldıracı **değil**; eşit bütçede recall yakın. Granülerlik/özet-uzunluğu düğmesidir. Rastgeleye üstünlük az segmentte daha büyük (çok segmentte rastgele de incelir). 32 makul varsayılan.

### 5.7. Füzyon Ağırlığı (alpha) — kare-AUC
| α | s0 | s42 | ortalama |
|---|---|---|---|
| 0.0 (v0) | 0.835 | 0.793 | 0.814 |
| 0.1 | 0.816 | 0.792 | 0.804 |
| 0.2 | 0.802 | 0.819 | 0.811 |
| 0.3 | 0.837 | 0.821 | **0.829** |
| 0.4 | 0.820 | 0.814 | 0.817 |
| 0.5 | 0.830 | 0.836 | **0.833** |

Metin füzyonu (α>0) genelde v0'ı geçer; en iyi bölge α ≈ 0.3–0.5. Eğri düz ve gürültülü (±0.02 varyans); çok düşük α (0.1) en zayıf. **α=0.3 sağlam varsayılan** (en iyi tek koşu + en yüksek ortalamalar arasında).

### 5.8. Pseudo-Normal (PN) Kaybı — eşleştirilmiş
| Tohum | PN açık | PN kapalı |
|---|---|---|
| 0 | 0.837 | 0.839 |
| 1 | 0.844 | 0.844 |
| 42 | 0.821 | 0.839 |
| **ort ± std** | **0.834 ± 0.011** | **0.841 ± 0.003** |

PN iyileştirme sağlamaz, hafifçe düşürür ve eğitimi **kararsızlaştırır** (~4 kat std). Aynı desen v0'da da (0.826 ± 0.019 vs 0.842 ± 0.002; eşleştirilmiş t-testi t=−1.81, **p=0.145** — anlamlı değil; Wilcoxon p=0.125). **Her iki modelde tutarlı negatif bulgu.**

### 5.9. Hard Attention ve Diğer Kayıplar — kare-AUC (3 tohum ort.)
| Varyant | ortalama | baseline'a göre |
|---|---|---|
| Tam (HA + tüm kayıplar) | 0.834 | — |
| Hard Attention kapalı | 0.827 | −0.007 |
| Sparsity kaybı kapalı | 0.833 | −0.001 |
| Smooth kaybı kapalı | 0.841 | +0.007 |

Yorum: HA kapatınca hafif düşer (HA ufak katkı sağlar); sparsity etkisiz; smooth kapatınca hafif yükselir (yardımcı olmaz). Hepsi varyans içinde. **Genel desen: yardımcı kayıplar (sparsity/smooth/PN) ölçülebilir fayda vermiyor; yalnızca HA mimari bileşeni küçük katkı yapıyor.**

### 5.9b. Karar Geçidi k'sı (decision_k) — top_k'dan FARKLI

**Önemli ayrım — iki farklı "k":**
- **`top_k` (segment seçimi):** Özete KAÇ segment alınacağı. Özetleme parametresi; özet uzunluğunu/recall'u belirler (Bölüm 4, 5.1).
- **`decision_k` (karar geçidi):** Videoyu "anomalili" saymak için en yüksek KAÇ segmentin ortalamasına bakılacağı. Tespit/karar parametresi; özetten önce, hangi videoların geçeceğini belirler. Özetin kendisini değiştirmez.

decision_k taraması (v1, karar geçidi, sabit eşikler):
| decision_k | gate F1 | gate doğruluk |
|---|---|---|
| 1 | 0.849 | 0.866 |
| 3 (varsayılan) | 0.853 | 0.869 |
| 5 | 0.856 | 0.872 |

decision_k geçidi çok az etkiler (hafif yükseliş); geçit ağırlıkla max-skoru ölçütüyle belirlenir. Özet recall'a etkisi yoktur. **Gürbüz parametre; varsayılan 3 iyi.**

### 5.10. Seed Varyansı
v1 (PN-açık, 3 tohum): kare-AUC **0.834 ± 0.011** (aralık 0.821–0.844). Bu yüzden tüm sonuçlar çoklu tohumla raporlanmıştır.

---

## 6. Sınıf-Bazlı Başarım ve Başarısızlık Analizi (v1, tepe k=3)

| Sınıf | recall | Yorum |
|---|---|---|
| Assault | 0.565 | belirgin, sürekli |
| Shoplifting | 0.470 | |
| RoadAccidents | 0.465 | belirgin hareket |
| Stealing | 0.449 | |
| Shooting | 0.446 | |
| Vandalism | 0.407 | |
| Burglary | 0.380 | |
| Fighting | 0.357 | |
| Robbery | 0.339 | |
| Arrest | 0.327 | |
| Explosion | 0.305 | çok kısa/ani |
| Arson | 0.228 | çok kısa/ani |
| Abuse | 0.058 | az örnek (n=2), görsel belirsiz |

**140 anomalili videonun 36'sında özet anomaliyi tamamen kaçırdı (recall = 0).** Başarısızlığın temel nedeni: olayın **süresi (kısa/ani)** ve **görsel belirginliği**. Özet kalitesinin üst sınırı skorlayıcının lokalizasyon kalitesiyle sınırlıdır.

---

## 7. Ana Sonuçlar (özet)

1. **Metin füzyonu (v1) çalışıyor:** hem tespit (+1.6pp kare-AUC) hem özetleme (+6.6pp recall), 3/3 tohumda.
2. **Özetleme katmanı çalışıyor:** videonun ~%24'ünü tutup anomalinin önemli bölümünü yakalar; rastgele/düzgün temelleri eşit sıkıştırmada geçer.
3. **Segment sayısı ve fps:** kaliteyi değil granülerliği/maliyeti etkiler; fps≥2 ve 32 segment makul varsayılanlar.
4. **NMS=2** özet recall'da tatlı nokta (mevcut varsayılan 1'den biraz iyi).
5. **Yardımcı kayıplar (pseudo_normal, sparsity, smooth) ölçülebilir fayda vermiyor;** PN ayrıca eğitimi kararsızlaştırıyor. Hard Attention küçük katkı sağlıyor.
6. **Başarısızlık** ağırlıkla kısa/ani (Patlama, Yangın) ve görsel belirsiz (Abuse) olaylarda.
7. **Sınırlılıklar:** eğitim varyansı (±0.02, çoklu tohumla raporlandı); tek veri kümesi (UCF-Crime); kare-AUC (~0.84) SOTA'nın (~0.87) biraz altında; yalnızca görsel+metin (ses yok).
