"""
Anomali-odakli video OZETLEME degerlendirmesi.

CLIP-TSA bir anomali SKORLAYICI; bu projenin asil katkisi skorlardan
ozet video ureten katman (video-seviyesi karar gecidi + segment secimi +
zamansal birlestirme). Bu script o katmani SAYISAL olarak olcer.

Uretilen metrikler:
  (A) Ozetleme kalitesi: temporal recall / precision / F1 ve sikistirma orani
      (ozet suresi / orijinal sure). Tum hesap FRAME uzayinda yapilir, yani
      saniye/fps bilgisi gerekmez -> ground-truth frame araliklariyla dogrudan
      kiyaslanir.
  (B) Ablation'lar: secim modu (peak/topk/threshold), top_k, pooling (mean/max),
      context (segment cinsinden) -> her birinin recall/sikistirma etkisi.
  (C) Baseline: ayni butcede RASTGELE ve UNIFORM segment secimi. Recall-sikistirma
      denge egrisi (ours vs random vs uniform) -> yontemin gercekten ise
      yaradigini gosterir.
  (D) Model: frame-level ROC-AUC/AP (pooling basina) + video-seviyesi karar
      gecidi ROC-AUC ve secilen esikte precision/recall/F1/accuracy.

Cikti:
  --output (JSON)        : tum toplu + per-video metrikler
  --curve_csv (CSV)      : recall-sikistirma egrisi (ours/random/uniform)
  --ablation_csv (CSV)   : ablation tablosu

Kullanim (ornek):
  python evaluate_summary.py \
    --annotation UCF_Crime/meta/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --feature_root data/ucf/features/test \
    --frame_counts data/ucf/features/frame_counts.json \
    --checkpoint checkpoints/ucf_loc/cliptsa_ucf_best.pkl \
    --output outputs/summary_eval.json

Iki checkpoint'i (or. pseudo_normal acik vs kapali) karsilastirmak icin
scripti iki kez farkli --checkpoint ve --output ile calistir.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.device import get_device
from src.cliptsa_adapter import CLIPTSAScorer
from src.feature_utils import process_feat
from src.summarizer import select_segments
from src.anomaly_decision import has_anomaly

# Kanitlanmis yardimcilar mevcut eval scriptinden yeniden kullaniliyor
from evaluate_frame_level_auc import (
    parse_temporal_annotations,
    load_frame_counts,
    find_feature,
    expand_to_frames,
)


# --------------------------------------------------------------------------
# Frame-uzayi yardimcilari
# --------------------------------------------------------------------------

def segment_of_frame(n_frames, num_segments):
    """Her frame'in ait oldugu segment id'si (expand_to_frames ile tutarli)."""
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    return (np.arange(n_frames) * num_segments // n_frames).clip(0, num_segments - 1)


def gt_frame_mask(spans, n_frames):
    gt = np.zeros(n_frames, dtype=np.int32)
    for s, e in spans:
        if s >= 0 and e >= 0:
            gt[s:min(e + 1, n_frames)] = 1
    return gt


def selected_frame_mask(selected_segments, n_frames, num_segments, context=0):
    """Secilen segmentleri (+context komsulari) frame maskesine cevirir."""
    if len(selected_segments) == 0:
        return np.zeros(n_frames, dtype=np.int32)
    expanded = set()
    for i in selected_segments:
        for j in range(i - context, i + context + 1):
            if 0 <= j < num_segments:
                expanded.add(j)
    seg_id = segment_of_frame(n_frames, num_segments)
    return np.isin(seg_id, list(expanded)).astype(np.int32)


def summarization_scores(sel_mask, gt_mask):
    """recall/precision/f1/compression don."""
    n = len(gt_mask)
    sel = float(sel_mask.sum())
    gt = float(gt_mask.sum())
    inter = float((sel_mask & gt_mask).sum())
    recall = inter / gt if gt > 0 else float("nan")
    precision = inter / sel if sel > 0 else 0.0
    if recall != recall or (recall + precision) == 0:  # nan veya 0
        f1 = float("nan") if recall != recall else 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    compression = sel / n if n > 0 else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "compression": compression,
    }


def nanmean(vals):
    arr = np.array([v for v in vals if v == v], dtype=np.float64)  # nan filtre
    return float(arr.mean()) if len(arr) else float("nan")


def topk_segments(scores, k):
    k = max(1, min(int(k), len(scores)))
    return sorted(np.argsort(scores)[::-1][:k].tolist())


def random_segments(num_segments, k, rng):
    k = max(1, min(int(k), num_segments))
    return sorted(rng.choice(num_segments, size=k, replace=False).tolist())


def uniform_segments(num_segments, k):
    k = max(1, min(int(k), num_segments))
    idx = np.linspace(0, num_segments - 1, k).round().astype(int)
    return sorted(np.unique(idx).tolist())


# --------------------------------------------------------------------------
# Skorlama: tum etiketli test videolarini bir pooling icin skorla ve cache'le
# --------------------------------------------------------------------------

def score_all_videos(ann, counts, feature_root, scorer, num_segments, pool):
    """
    Her video icin (stem -> dict) cache:
      scores (num_segments,), n_frames, gt_mask, label, class
    """
    cache = {}
    missing_feat, missing_count = [], []
    for stem, info in ann.items():
        feat_path = find_feature(feature_root, stem)
        if feat_path is None:
            missing_feat.append(stem)
            continue
        if stem not in counts:
            missing_count.append(stem)
            continue
        n_frames = int(counts[stem])
        raw = np.load(feat_path, allow_pickle=True).astype(np.float32)
        feats = process_feat(raw, length=num_segments, pool=pool)
        seg_scores = np.asarray(scorer.score_features(feats), dtype=np.float32)[:num_segments]
        gt = gt_frame_mask(info["spans"], n_frames)
        label = 0 if str(info["class"]).lower().startswith("normal") else 1
        cache[stem] = {
            "scores": seg_scores,
            "n_frames": n_frames,
            "gt_mask": gt,
            "label": label,
            "class": info["class"],
        }
    return cache, missing_feat, missing_count


# --------------------------------------------------------------------------
# (D) Model seviyesi metrikler
# --------------------------------------------------------------------------

def model_level_metrics(cache, num_segments, max_threshold, topk_mean_threshold, decision_k):
    all_scores, all_labels = [], []   # frame-level AUC icin
    vid_max, vid_topk, vid_label = [], [], []
    gate_tp = gate_fp = gate_tn = gate_fn = 0

    for stem, d in cache.items():
        seg = d["scores"]
        n_frames = d["n_frames"]
        frame_scores = expand_to_frames(seg, n_frames)
        all_scores.append(frame_scores)
        all_labels.append(d["gt_mask"])

        k = max(1, min(decision_k, len(seg)))
        mx = float(seg.max())
        tk = float(np.mean(np.sort(seg)[-k:]))
        vid_max.append(mx)
        vid_topk.append(tk)
        vid_label.append(d["label"])

        decided = (mx >= max_threshold) and (tk >= topk_mean_threshold)
        if d["label"] == 1 and decided:
            gate_tp += 1
        elif d["label"] == 0 and decided:
            gate_fp += 1
        elif d["label"] == 0 and not decided:
            gate_tn += 1
        else:
            gate_fn += 1

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    frame_auc = float(roc_auc_score(y_true, y_score)) if y_true.sum() > 0 else float("nan")
    frame_ap = float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else float("nan")

    vid_label = np.array(vid_label)
    video_auc_max = float(roc_auc_score(vid_label, vid_max)) if len(set(vid_label.tolist())) > 1 else float("nan")
    video_auc_topk = float(roc_auc_score(vid_label, vid_topk)) if len(set(vid_label.tolist())) > 1 else float("nan")

    prec = gate_tp / (gate_tp + gate_fp) if (gate_tp + gate_fp) > 0 else 0.0
    rec = gate_tp / (gate_tp + gate_fn) if (gate_tp + gate_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (gate_tp + gate_tn) / max(1, (gate_tp + gate_fp + gate_tn + gate_fn))

    return {
        "frame_level_roc_auc": frame_auc,
        "frame_level_average_precision": frame_ap,
        "video_level_roc_auc_max": video_auc_max,
        "video_level_roc_auc_topk_mean": video_auc_topk,
        "decision_gate": {
            "threshold_max": max_threshold,
            "threshold_topk_mean": topk_mean_threshold,
            "tp": gate_tp, "fp": gate_fp, "tn": gate_tn, "fn": gate_fn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        },
    }


# --------------------------------------------------------------------------
# (A) + (B) Ozetleme metrikleri ve ablation'lar (anormal videolarda)
# --------------------------------------------------------------------------

def summarize_config(cache, num_segments, mode, top_k, context,
                     nms_window=1, segment_threshold=0.6, only_abnormal=True):
    """Tek bir konfigurasyon icin anormal videolar uzerinde ortalama metrikler."""
    recs, precs, f1s, comps = [], [], [], []
    per_video = []
    for stem, d in cache.items():
        if only_abnormal and d["label"] != 1:
            continue
        if d["gt_mask"].sum() == 0:
            continue
        sel = select_segments(
            d["scores"], mode=mode, top_k=top_k,
            nms_window=nms_window, segment_threshold=segment_threshold,
        )
        mask = selected_frame_mask(sel, d["n_frames"], num_segments, context=context)
        m = summarization_scores(mask, d["gt_mask"])
        recs.append(m["recall"]); precs.append(m["precision"])
        f1s.append(m["f1"]); comps.append(m["compression"])
        per_video.append({"stem": stem, "class": d["class"], "n_selected_seg": len(sel), **m})
    return {
        "mode": mode, "top_k": top_k, "context": context,
        "n_videos": len(recs),
        "mean_recall": nanmean(recs),
        "mean_precision": nanmean(precs),
        "mean_f1": nanmean(f1s),
        "mean_compression": nanmean(comps),
    }, per_video


# --------------------------------------------------------------------------
# (C) Baseline + recall-sikistirma egrisi
# --------------------------------------------------------------------------

def tradeoff_curve(cache, num_segments, k_values, context, seed=0, n_random=20):
    """Her k icin ours(topk) / random / uniform ortalama (compression, recall)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in k_values:
        ours_r, ours_c = [], []
        rand_r, rand_c = [], []
        unif_r, unif_c = [], []
        for stem, d in cache.items():
            if d["label"] != 1 or d["gt_mask"].sum() == 0:
                continue
            ns, gt = d["n_frames"], d["gt_mask"]
            # ours
            sel = topk_segments(d["scores"], k)
            m = summarization_scores(selected_frame_mask(sel, ns, num_segments, context), gt)
            ours_r.append(m["recall"]); ours_c.append(m["compression"])
            # uniform
            su = uniform_segments(num_segments, k)
            mu = summarization_scores(selected_frame_mask(su, ns, num_segments, context), gt)
            unif_r.append(mu["recall"]); unif_c.append(mu["compression"])
            # random (n_random ortalamasi)
            rr, rc = [], []
            for _ in range(n_random):
                sr = random_segments(num_segments, k, rng)
                mr = summarization_scores(selected_frame_mask(sr, ns, num_segments, context), gt)
                rr.append(mr["recall"]); rc.append(mr["compression"])
            rand_r.append(nanmean(rr)); rand_c.append(nanmean(rc))
        rows.append({
            "k": k,
            "ours_recall": nanmean(ours_r), "ours_compression": nanmean(ours_c),
            "random_recall": nanmean(rand_r), "random_compression": nanmean(rand_c),
            "uniform_recall": nanmean(unif_r), "uniform_compression": nanmean(unif_c),
        })
    return rows


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Anomali-odakli ozetleme degerlendirmesi")
    p.add_argument("--annotation", default="UCF_Crime/meta/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    p.add_argument("--feature_root", default="data/ucf/features/test")
    p.add_argument("--frame_counts", default="data/ucf/features/frame_counts.json")
    p.add_argument("--videos_dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cliptsa_repo", default="third_party/CLIP-TSA")
    p.add_argument("--feature_size", type=int, default=512)
    p.add_argument("--num_segments", type=int, default=32)
    p.add_argument("--k", type=float, default=0.95)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--disable_HA", action="store_true")
    # --- v0 (gorsel) / v1 (text fuzyonu) secimi ---
    p.add_argument("--variant", choices=["v0", "v1"], default="v0",
                   help="v0=duz gorsel scorer; v1=CLIP-TSA + text-prototip fuzyonu")
    p.add_argument("--text_embeds", default="data/ucf/text_embeds_v1.npz",
                   help="v1 icin onceden hesaplanmis text prototipleri (.npz)")
    p.add_argument("--alpha", type=float, default=0.3, help="v1 fuzyon agirligi")
    p.add_argument("--tau", type=float, default=0.07, help="v1 text benzerligi sicakligi")
    # karar geciti esikleri (ucf_loc Youden varsayilanlari)
    p.add_argument("--max_threshold", type=float, default=0.5472)
    p.add_argument("--topk_mean_threshold", type=float, default=0.4576)
    p.add_argument("--decision_k", type=int, default=3)
    # ana ozetleme konfigurasyonu
    p.add_argument("--main_mode", default="peak", choices=["peak", "topk", "threshold"])
    p.add_argument("--main_top_k", type=int, default=3)
    p.add_argument("--main_context", type=int, default=1, help="context (segment cinsinden)")
    p.add_argument("--nms_window", type=int, default=1, help="peak modunda komsu bastirma penceresi")
    # cikti
    p.add_argument("--output", default="outputs/summary_eval.json")
    p.add_argument("--curve_csv", default="outputs/summary_tradeoff_curve.csv")
    p.add_argument("--ablation_csv", default="outputs/summary_ablation.csv")
    p.add_argument("--pools", default="mean,max", help="virgulle: degerlendirilecek pooling'ler")
    args = p.parse_args()

    ann = parse_temporal_annotations(args.annotation)
    counts = load_frame_counts(args.frame_counts, args.videos_dir)
    device = get_device()
    print(f"Cihaz: {device}")
    print(f"Anotasyon: {len(ann)} video | Frame sayisi bilinen: {len(counts)}")

    if args.variant == "v1":
        from src.cliptsa_adapter_v1 import CLIPTSAScorerV1
        scorer = CLIPTSAScorerV1(
            repo_dir=args.cliptsa_repo,
            checkpoint=args.checkpoint,
            text_embeds=args.text_embeds,
            feature_size=args.feature_size,
            k=args.k,
            num_samples=args.num_samples,
            enable_HA=not args.disable_HA,
            alpha=args.alpha,
            tau=args.tau,
            device=device,
        )
    else:
        scorer = CLIPTSAScorer(
            repo_dir=args.cliptsa_repo,
            checkpoint=args.checkpoint,
            feature_size=args.feature_size,
            k=args.k,
            num_samples=args.num_samples,
            enable_HA=not args.disable_HA,
            device=device,
        )

    pools = [s.strip() for s in args.pools.split(",") if s.strip()]
    results = {"checkpoint": args.checkpoint, "num_segments": args.num_segments, "pools": {}}

    ablation_rows = []
    curve_rows_main = None

    for pool in pools:
        print(f"\n=== Pooling: {pool} ===")
        cache, mfeat, mcount = score_all_videos(
            ann, counts, args.feature_root, scorer, args.num_segments, pool
        )
        n_abn = sum(1 for d in cache.values() if d["label"] == 1 and d["gt_mask"].sum() > 0)
        n_nor = sum(1 for d in cache.values() if d["label"] == 0)
        print(f"  Skorlanan video: {len(cache)} (anormal+gt: {n_abn}, normal: {n_nor})")
        if mfeat:
            print(f"  Feature bulunamayan: {len(mfeat)}")
        if mcount:
            print(f"  Frame sayisi yok: {len(mcount)}")

        # (D) model seviyesi
        model_m = model_level_metrics(
            cache, args.num_segments,
            args.max_threshold, args.topk_mean_threshold, args.decision_k,
        )

        # (A) ana konfigurasyon
        main_cfg, main_per_video = summarize_config(
            cache, args.num_segments,
            mode=args.main_mode, top_k=args.main_top_k, context=args.main_context,
            nms_window=args.nms_window,
        )

        # (B) ablation'lar
        abl = []
        # secim modu
        for mode in ["peak", "topk", "threshold"]:
            cfg, _ = summarize_config(cache, args.num_segments, mode=mode,
                                      top_k=args.main_top_k, context=args.main_context)
            cfg["ablation"] = "select_mode"
            abl.append(cfg)
        # top_k
        for tk in [1, 2, 3, 5]:
            cfg, _ = summarize_config(cache, args.num_segments, mode="topk",
                                      top_k=tk, context=args.main_context)
            cfg["ablation"] = "top_k"
            abl.append(cfg)
        # context
        for ctx in [0, 1, 2]:
            cfg, _ = summarize_config(cache, args.num_segments, mode=args.main_mode,
                                      top_k=args.main_top_k, context=ctx)
            cfg["ablation"] = "context"
            abl.append(cfg)
        for cfg in abl:
            ablation_rows.append({"pool": pool, **cfg})

        # (C) recall-sikistirma egrisi
        k_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
        curve = tradeoff_curve(cache, args.num_segments, k_values, context=args.main_context)
        if pool == "mean":
            curve_rows_main = curve

        results["pools"][pool] = {
            "model_level": model_m,
            "summary_main": main_cfg,
            "ablations": abl,
            "tradeoff_curve": curve,
            "per_video_main": main_per_video,
            "missing_feature": mfeat,
            "missing_frame_count": mcount,
        }

        # konsol ozeti
        print(f"  frame-AUC={model_m['frame_level_roc_auc']:.4f} | "
              f"video-AUC(max)={model_m['video_level_roc_auc_max']:.4f} | "
              f"gate F1={model_m['decision_gate']['f1']:.3f}")
        print(f"  ANA ozet ({args.main_mode}, k={args.main_top_k}, ctx={args.main_context}): "
              f"recall={main_cfg['mean_recall']:.3f} "
              f"precision={main_cfg['mean_precision']:.3f} "
              f"F1={main_cfg['mean_f1']:.3f} "
              f"sikistirma={main_cfg['mean_compression']:.3f}")

    # ----- dosyalar -----
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON kaydedildi: {out}")

    # ablation CSV
    if ablation_rows:
        ac = Path(args.ablation_csv)
        with open(ac, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pool", "ablation", "mode", "top_k", "context",
                                              "n_videos", "mean_recall", "mean_precision",
                                              "mean_f1", "mean_compression"])
            w.writeheader()
            for r in ablation_rows:
                w.writerow({kk: r.get(kk) for kk in w.fieldnames})
        print(f"Ablation CSV: {ac}")

    # egri CSV (mean pooling)
    if curve_rows_main:
        cc = Path(args.curve_csv)
        with open(cc, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve_rows_main[0].keys()))
            w.writeheader()
            for r in curve_rows_main:
                w.writerow(r)
        print(f"Egri CSV: {cc}")

    # ----- kisa rapor -----
    print("\n" + "=" * 64)
    print("OZET (mean pooling)")
    mp = results["pools"].get("mean") or next(iter(results["pools"].values()))
    ml = mp["model_level"]
    sm = mp["summary_main"]
    print(f"  Frame-level ROC-AUC      : {ml['frame_level_roc_auc']:.4f}")
    print(f"  Video-level ROC-AUC (max): {ml['video_level_roc_auc_max']:.4f}")
    print(f"  Karar geciti F1/Acc      : {ml['decision_gate']['f1']:.3f} / {ml['decision_gate']['accuracy']:.3f}")
    print(f"  Ozet recall              : {sm['mean_recall']:.3f}")
    print(f"  Ozet sikistirma          : {sm['mean_compression']:.3f}  "
          f"(ortalama videonun ~%{100*sm['mean_compression']:.0f}'i)")
    if curve_rows_main:
        # k=3 satirinda ours vs random/uniform
        row3 = next((r for r in curve_rows_main if r["k"] == 3), curve_rows_main[0])
        print(f"  k=3'te recall: OURS={row3['ours_recall']:.3f}  "
              f"random={row3['random_recall']:.3f}  uniform={row3['uniform_recall']:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
