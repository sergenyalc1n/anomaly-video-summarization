# Anomali-Odaklı Video Özetleme (CLIP-TSA + Metin Füzyonu)

Ham bir güvenlik kamerası videosundaki **anomali anlarını otomatik tespit edip kesen** ve bunları tek bir kısa **özet videoda** birleştiren, zayıf-denetimli uçtan uca bir sistem. Skorlayıcı olarak [CLIP-TSA](https://github.com/joos2010kj/CLIP-TSA) kullanılır; üzerine, anomali sınıflarının metinsel betimlemelerinden gelen bilgiyi katan bir **metin-görüntü füzyonu (v1)** eklenmiştir.

## Yöntem (özet)

1. **Özellik çıkarımı** — her kareden CLIP ViT-B/16 gömme vektörü, 32 zamansal segmente havuzlanır.
2. **Anomali skorlama** — CLIP-TSA her segmente [0,1] anomali skoru verir. **v1**'de skor, sabit metin prototipleriyle harmanlanır: `skor = (1−α)·görsel + α·metin` (α=0.3).
3. **Karar geçidi** — video, max skoru ve top-k ortalaması iki eşiği birden geçerse "anomalili" sayılır.
4. **Özetleme** — anomalili segmentler tepe seçimi (1B NMS) ile seçilir, bağlam dolgusu eklenir, birleştirilir ve ffmpeg ile özet `.mp4` üretilir.

## Sonuçlar (UCF-Crime standart bölünmesi, 290 etiketli test)

| | v0 (görsel) | **v1 (metin füzyonu)** |
|---|---|---|
| Kare düzeyi ROC-AUC | 0.818 | **0.834** (en iyi 0.844) |
| Video düzeyi ROC-AUC | 0.941 | **0.946** |
| Özet recall (@ ~%24) | 0.317 | **0.383** |

Metin füzyonu eşleştirilmiş üç tohumda **hem tespiti (+1.6pp) hem özetlemeyi (+6.6pp)** iyileştirir. Özet, videonun ~%24'ünü tutarak anomalinin önemli bölümünü yakalar ve eşit sıkıştırmada rastgele/düzgün seçim temellerini geçer.

Tüm ablation'lar (fps, segment sayısı, alpha, NMS, pseudo-normal, Hard Attention, sparsity/smooth, decision_k, seçim modu, bağlam, pooling, sınıf-bazlı) ve sayısal detaylar için **[BULGULAR.md](BULGULAR.md)** dosyasına bakın.

## Kurulum

```bash
pip install -r requirements.txt          # torch (CUDA), open-clip, opencv, scikit-learn, scipy, tqdm
# ffmpeg sistemde kurulu olmalı (video kesme/birleştirme için)

# v1 metin prototiplerini bir kez üret (CLIP text encoder ile)
python tools/build_text_prototypes_v1.py    # -> data/ucf/text_embeds_v1.npz
```

> Not: feature `.npy` dosyaları ve eğitim checkpoint'leri repoya dahil değildir (büyük). Özellikler `data/ucf/features/...` altında beklenir; standart bölünmeye hazırlamak için `build_fps_sets.py` kullanılabilir.

## Kullanım

### Eğitim (v1 — metin füzyonu)

```bash
python train_cliptsa_ucf_v1.py \
  --feature_root data/ucf/features \
  --output_dir checkpoints/v1 \
  --epochs 15 --alpha 0.3 \
  --text_embeds data/ucf/text_embeds_v1.npz \
  --select_by auc --seed 1
```

v0 (görsel-yalnız) için `train_cliptsa_ucf.py` (aynı argümanlar, metin füzyonu yok).

### Değerlendirme (özetleme + tespit metrikleri)

```bash
python evaluate_summary.py --variant v1 \
  --checkpoint checkpoints/v1/cliptsa_ucf_v1_best.pkl \
  --feature_root data/ucf/features/test \
  --alpha 0.3 --text_embeds data/ucf/text_embeds_v1.npz \
  --output outputs/summary_v1.json
```

`--variant v0` ile görsel-yalnız değerlendirilir. Çıktı: frame/video ROC-AUC, karar geçidi, özet recall/precision/F1/sıkıştırma, rastgele/uniform baseline ve ablation CSV'leri.

### Tek video üzerinde özet üretme

```bash
python infer_summary_v1.py \
  --video videos/Shooting002_x264.mp4 \
  --feature_file data/ucf/features/test/Shooting/Shooting002_x264.npy \
  --checkpoint checkpoints/v1/cliptsa_ucf_v1_best.pkl \
  --text_embeds data/ucf/text_embeds_v1.npz \
  --alpha 0.3 \
  --output_dir outputs/
```

## Önemli argümanlar

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--variant` | v0 | v0=görsel, v1=metin füzyonu (evaluate_summary) |
| `--alpha` | 0.3 | metin füzyon ağırlığı |
| `--num_segments` | 32 | zamansal segment sayısı (16/32/64) |
| `--main_top_k` | 3 | özete alınacak segment sayısı |
| `--nms_window` | 1 | tepe bastırma penceresi (2 önerilir) |
| `--main_context` | 1 | bağlam dolgusu (segment) |

## Depo yapısı

```
src/                       # çekirdek kütüphane (skorlama, özetleme, v1 füzyon)
train_cliptsa_ucf.py       # v0 eğitim    train_cliptsa_ucf_v1.py  # v1 eğitim
evaluate_summary.py        # özetleme + tespit değerlendirmesi (v0/v1)
build_fps_sets.py          # feature setlerini standart bölünmeye hazırlama
analyze_pn_ablation.py     # pseudo-normal ablation istatistiği
tools/build_text_prototypes_v1.py   # v1 metin prototipleri
BULGULAR.md                # tüm bulgular ve tablolar
```

## Lisans

Bu depodaki özgün kod **MIT Lisansı** ile sunulmuştur (bkz. [LICENSE](LICENSE)). `third_party/CLIP-TSA/` dizini [CLIP-TSA](https://github.com/joos2010kj/CLIP-TSA) projesine aittir ve kendi lisans koşullarına tabidir.

## Atıf

```bibtex
@inproceedings{joo2023cliptsa,
  title = {CLIP-TSA: CLIP-Assisted Temporal Self-Attention for Weakly-Supervised Video Anomaly Detection},
  author = {Joo, Hyekang Kevin and Vo, Khoa and Yamazaki, Kashu and Le, Ngan},
  booktitle = {IEEE ICIP}, pages = {3230--3234}, year = {2023}
}
```
