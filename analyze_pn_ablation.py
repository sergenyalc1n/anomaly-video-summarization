"""
pseudo_normal (PN) ablasyonunu BILIMSEL olarak degerlendirir.

checkpoints/fps2_s{seed}        -> PN ACIK  (varsayilan lambda=1.0)
checkpoints/fps2_nopn_s{seed}   -> PN KAPALI (lambda=0)

Her checkpoint'in icindeki 'frame_auc' (egitimde secilen en iyi epoch'un
frame-level AUC'si) okunur. Ayni seed'ler eslestirilir; betimsel istatistik
(ortalama, std, min, max), eslestirilmis fark, eslestirilmis t-testi ve
Wilcoxon isaretli sira testi hesaplanir. Kucuk n uyarisi basilir.

Kullanim:
    python analyze_pn_ablation.py
"""
import glob
import os
import re

import numpy as np
import torch

try:
    from scipy import stats as st
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def read_frame_auc(ckpt_path):
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    fa = ck.get("frame_auc", None)
    seed = None
    args = ck.get("args", {})
    if isinstance(args, dict):
        seed = args.get("seed", None)
        pnl = args.get("pseudo_normal_lambda", None)
    else:
        pnl = None
    return fa, seed, pnl


def collect(prefix):
    """prefix: 'fps2_s' (PN on) veya 'fps2_nopn_s' (PN off). seed->frame_auc"""
    out = {}
    for d in glob.glob(f"checkpoints/{prefix}*"):
        m = re.search(rf"{re.escape(prefix)}(\d+)$", d.replace("\\", "/"))
        if not m:
            continue
        seed = int(m.group(1))
        best = os.path.join(d, "cliptsa_ucf_best.pkl")
        if not os.path.exists(best):
            continue
        fa, _, _ = read_frame_auc(best)
        if fa is not None:
            out[seed] = float(fa)
    return out


def desc(name, vals):
    a = np.array(vals, dtype=float)
    print(f"  {name:10s} n={len(a)}  ort={a.mean():.4f}  std={a.std(ddof=1):.4f}  "
          f"min={a.min():.4f}  max={a.max():.4f}")
    return a


def main():
    # PN-kapali klasorler 'fps2_nopn_s' ile baslar; PN-acik 'fps2_s' AMA
    # 'fps2_nopn_s' de 'fps2_s' ile eslesmesin diye ayri topluyoruz.
    pn_off = collect("fps2_nopn_s")
    pn_on_all = collect("fps2_s")
    pn_on = {s: v for s, v in pn_on_all.items() if s not in pn_off or True}
    # 'fps2_s{seed}' globu 'fps2_nopn_s{seed}'yi yakalamaz cunku prefix 'fps2_s'
    # ile 'fps2_nopn_s' farkli; yine de guvenlik icin nopn dizinlerini disla:
    pn_on = {}
    for d in glob.glob("checkpoints/fps2_s*"):
        dd = d.replace("\\", "/")
        if "nopn" in dd:
            continue
        m = re.search(r"fps2_s(\d+)$", dd)
        if not m:
            continue
        best = os.path.join(d, "cliptsa_ucf_best.pkl")
        if os.path.exists(best):
            fa, _, _ = read_frame_auc(best)
            if fa is not None:
                pn_on[int(m.group(1))] = float(fa)

    print("=== PN ablasyonu: frame-level AUC (checkpoint'lerden) ===\n")
    print("Seed bazinda:")
    seeds = sorted(set(pn_on) | set(pn_off))
    print(f"  {'seed':>5} | {'PN-acik':>9} | {'PN-kapali':>10} | {'fark(acik-kapali)':>18}")
    diffs = []
    for s in seeds:
        a = pn_on.get(s); b = pn_off.get(s)
        a_s = f"{a:.4f}" if a is not None else "   -   "
        b_s = f"{b:.4f}" if b is not None else "   -   "
        if a is not None and b is not None:
            d = a - b
            diffs.append(d)
            d_s = f"{d:+.4f}"
        else:
            d_s = "   -   "
        print(f"  {s:>5} | {a_s:>9} | {b_s:>10} | {d_s:>18}")

    print("\nBetimsel:")
    if pn_on:
        on = desc("PN-acik", list(pn_on.values()))
    if pn_off:
        off = desc("PN-kapali", list(pn_off.values()))

    if len(diffs) >= 2:
        diffs = np.array(diffs)
        print(f"\nEslestirilmis fark (PN-acik - PN-kapali): "
              f"ort={diffs.mean():+.4f}  std={diffs.std(ddof=1):.4f}  n={len(diffs)}")
        if HAVE_SCIPY:
            t, p_t = st.ttest_rel(
                [pn_on[s] for s in seeds if s in pn_on and s in pn_off],
                [pn_off[s] for s in seeds if s in pn_on and s in pn_off],
            )
            print(f"  Eslestirilmis t-testi: t={t:.3f}  p={p_t:.4f}")
            try:
                w, p_w = st.wilcoxon(
                    [pn_on[s] for s in seeds if s in pn_on and s in pn_off],
                    [pn_off[s] for s in seeds if s in pn_on and s in pn_off],
                )
                print(f"  Wilcoxon: W={w:.3f}  p={p_w:.4f}")
            except Exception as e:
                print(f"  Wilcoxon hesaplanamadi (kucuk n): {e}")
            # yorum
            print("\n=== YORUM ===")
            if p_t < 0.05:
                yon = "PN-acik daha iyi" if diffs.mean() > 0 else "PN-kapali daha iyi"
                print(f"  Fark istatistiksel anlamli (p<0.05): {yon}.")
            else:
                print("  Fark istatistiksel ANLAMLI DEGIL (p>=0.05): mevcut kanitla "
                      "PN'in frame-AUC'yi degistirdigi soylenemez.")
        # varyans karsilastirmasi
        if pn_on and pn_off:
            print(f"\n  Kararlilik (std): PN-acik={np.std(list(pn_on.values()),ddof=1):.4f}  "
                  f"PN-kapali={np.std(list(pn_off.values()),ddof=1):.4f}  "
                  f"(dusuk std = daha kararli egitim)")
        print("\n  NOT: n kucuk; bu testler dusuk gucludur, sonucu temkinli yorumla.")
    else:
        print("\nYeterli eslestirilmis seed yok; once egitim koslarini tamamla.")


if __name__ == "__main__":
    main()
