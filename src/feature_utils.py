import numpy as np


def process_feat(feat, length=32, pool="mean"):
    """
    Değişken uzunluklu (T, F) feature dizisini sabit length adet
    zamansal segmente indirir.

    pool:
      "mean" -> her segmentteki kareleri ortalar (varsayilan, eski davranis).
      "max"  -> her segmentte eleman-bazli max alir (pik korunur), sonra
                L2 yeniden normalize eder (CLIP feature'lari birim kurede tutmak icin).
    """
    feat = np.asarray(feat, dtype=np.float32)

    if feat.ndim != 2:
        raise ValueError(f"Beklenen feature shape (T, F), gelen: {feat.shape}")

    n, f = feat.shape
    if n == 0:
        raise ValueError("Boş feature dizisi.")

    edges = np.linspace(0, n, length + 1, dtype=int)
    out = np.zeros((length, f), dtype=np.float32)

    for i in range(length):
        start, end = edges[i], edges[i + 1]

        if end <= start:
            out[i] = feat[min(start, n - 1)]
        elif pool == "max":
            seg = feat[start:end].max(axis=0)
            norm = np.linalg.norm(seg)
            out[i] = seg / norm if norm > 0 else seg
        else:  # mean
            out[i] = feat[start:end].mean(axis=0)

    return out
