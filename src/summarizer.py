import numpy as np


def select_segments(
    scores,
    mode="peak",
    top_k=3,
    nms_window=1,
    segment_threshold=0.50,
    budget_ratio=None,
    min_segments=1,
):
    """Anomali videosunda HANGI segmentlerin ozete girecegini secer.

    ONEMLI: Burada "anomali var mi" karari VERILMEZ. O karar video
    seviyesinde has_anomaly() ile (mutlak esikle) verilir. Bu fonksiyon
    yalnizca "anomali var" denen videoda segment secer ve bunu GORECELI
    (skor siralamasina dayali) yapar -- cunku CLIP-TSA/MIL skorlari
    segmentler arasinda birbirine yakin gelebiliyor, mutlak esik bu yuzden
    burada saglam degil.

    mode="peak"      : skor egrisindeki yerel tepeleri NMS ile sec
                       (varsayilan). Ayni olaydan gelen bitisik segmentleri
                       bastirir, en yuksek top_k tepeyi birakir.
    mode="topk"      : sadece en yuksek skorlu top_k segmenti al (bastirma yok).
    mode="threshold" : ESKI davranis -> mutlak segment_threshold ustu segmentler
                       (geriye donuk uyumluluk icin).
    """
    scores = np.asarray(scores, dtype=np.float32)
    num_segments = len(scores)

    if num_segments == 0:
        return []

    mode = (mode or "peak").lower()

    # --- Eski mutlak esik modu (geriye donuk uyumluluk) ---
    if mode == "threshold":
        candidate_indices = np.where(scores >= segment_threshold)[0].tolist()
        if len(candidate_indices) == 0:
            return []

        # Budget mantigi devre disi: esigi gecen TUM segmentler ozete dahil edilir.
        # Geri acmak istersen budget_ratio'ya bir oran ver ( or. 0.20).
        if budget_ratio is None or budget_ratio <= 0:
            return sorted(candidate_indices)

        budget = max(min_segments, int(round(num_segments * budget_ratio)))
        budget = min(budget, len(candidate_indices))
        candidate_indices = sorted(
            candidate_indices,
            key=lambda i: float(scores[i]),
            reverse=True,
        )
        return sorted(candidate_indices[:budget])

    # --- Goreceli secim: top-k / peak (1D NMS) ---
    top_k = max(min_segments, min(int(top_k), num_segments))
    # topk modunda bastirma yok; peak modunda komsulari bastir.
    window = 0 if mode == "topk" else max(0, int(nms_window))

    # Skoru buyukten kucuge dolas, greedy NMS: secilen bir tepenin
    # +-window komsulugundaki segmentleri atla.
    order = np.argsort(scores)[::-1]
    selected = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - s) > window for s in selected):
            selected.append(idx)
            if len(selected) >= top_k:
                break

    return sorted(selected)


def segments_to_time_ranges(
    selected_segments,
    num_segments,
    total_duration,
    context_sec=2.0,
):
    if not selected_segments:
        return []

    seg_dur = total_duration / float(num_segments)
    ranges = []

    for idx in sorted(selected_segments):
        start = max(0.0, idx * seg_dur - context_sec)
        end = min(total_duration, (idx + 1) * seg_dur + context_sec)
        ranges.append((start, end))

    return ranges
