"""v1 (text-fuzyon) ile etiketsiz anomali odakli video ozetleme.

infer_summary.py ile AYNI akis; tek fark skorlayici: duz CLIPTSAScorer yerine
CLIPTSAScorerV1 (CLIP-TSA + DONUK text-prototip fuzyonu, sabit alpha=0.3).
Boylece inference, ablation'da en iyi cikan v1 modelini birebir kullanir.

Kullanim (onerilen v1 checkpoint'i: ablation seed 0 = 0.8450):
  python infer_summary_v1.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/ablation/alpha0.3_seed0/cliptsa_ucf_v1_best.pkl \
    --output_dir outputs

  # Hazir .npy feature varsa:
  python infer_summary_v1.py --video v.mp4 \
    --checkpoint checkpoints/ablation/alpha0.3_seed0/cliptsa_ucf_v1_best.pkl \
    --feature_file path/to/feats.npy

Notlar:
  - alpha/tau, modeli EGITTIGIN degerlerle ayni olmali (varsayilan 0.3 / 0.07).
  - text_embeds, gorsel feature'lari cikardigin CLIP backbone'uyla ayni olmali
    (data/ucf/text_embeds_v1.npz; gerekiyorsa tools/build_text_prototypes_v1.py).
"""

import argparse
import json
from pathlib import Path

from src.device import get_device
from src.clip_features import extract_clip_features, temporal_segment_features
from src.cliptsa_adapter_v1 import CLIPTSAScorerV1
from src.anomaly_decision import has_anomaly
from src.summarizer import select_segments, segments_to_time_ranges
from src.video_io import get_video_duration, merge_time_ranges, write_summary_video

# feature okuma yardimcisini temel scriptten aynen kullan (kod tekrari yok).
from infer_summary import load_feature_file


def main():
    parser = argparse.ArgumentParser(
        description="v1 (text-fuzyon) CLIP-TSA tabanli anomali odakli video ozetleme"
    )

    parser.add_argument("--video", required=True, help="Ozet cikarilacak ham video yolu")
    parser.add_argument("--output_dir", default="outputs")

    parser.add_argument("--cliptsa_repo", default="third_party/CLIP-TSA")
    parser.add_argument("--checkpoint", required=True,
                        help="Egitilmis v1 checkpoint yolu (cliptsa_ucf_v1_best.pkl)")

    parser.add_argument("--feature_file", default=None,
                        help="Varsa CLIP-TSA uyumlu .npy feature. Yoksa videodan cikarilir.")

    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--sample_fps", type=int, default=2)
    parser.add_argument("--pool", choices=["mean", "max"], default="mean",
                        help="segment havuzlama; egittigin modelle AYNI olmali")

    parser.add_argument("--feature_size", type=int, default=512)
    parser.add_argument("--k", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--disable_HA", action="store_true")

    # --- v1 text fuzyon parametreleri (egitimle AYNI olmali) ---
    parser.add_argument("--text_embeds", default="data/ucf/text_embeds_v1.npz")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.07)

    # --- video-level karar esikleri ---
    parser.add_argument("--max_threshold", type=float, default=0.5472)
    parser.add_argument("--topk_mean_threshold", type=float, default=0.4576)
    parser.add_argument("--decision_k", type=int, default=3)

    # --- segment secimi (goreceli) ---
    parser.add_argument("--select_mode", choices=["peak", "topk", "threshold"], default="peak")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--nms_window", type=int, default=1)
    parser.add_argument("--segment_threshold", type=float, default=0.60)
    parser.add_argument("--budget_ratio", type=float, default=None)
    parser.add_argument("--context_sec", type=float, default=2.0)

    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    device = get_device()
    print(f"Cihaz: {device}")

    total_duration = get_video_duration(video_path)
    print(f"Video suresi: {total_duration:.2f} saniye")

    if args.feature_file:
        print("Feature dosyadan okunuyor.")
        segment_features = load_feature_file(args.feature_file,
                                             num_segments=args.num_segments, pool=args.pool)
        if segment_features.ndim >= 2:
            args.num_segments = segment_features.shape[-2]
            args.feature_size = segment_features.shape[-1]
    else:
        print("Feature dosyasi verilmedi; ham videodan CLIP feature cikariliyor.")
        frame_features = extract_clip_features(video_path, device=device, sample_fps=args.sample_fps)
        segment_features = temporal_segment_features(frame_features, num_segments=args.num_segments)
        args.feature_size = segment_features.shape[-1]

    print(f"Feature shape: {segment_features.shape} | size: {args.feature_size} | "
          f"segment: {args.num_segments}")

    # >>> TEK FARK: v1 (text-fuzyon) skorlayici <<<
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

    print("v1 anomaly score (fused) uretiyor.")
    scores = scorer.score_features(segment_features)

    if len(scores) != args.num_segments:
        args.num_segments = len(scores)

    print("\nSegment skorlari:")
    seg_dur = total_duration / args.num_segments
    for i, s in enumerate(scores):
        print(f"[{i:02d}] {i * seg_dur:7.2f}s - {(i + 1) * seg_dur:7.2f}s | score={s:.4f}")

    anomaly_exists, decision_info = has_anomaly(
        scores, max_threshold=args.max_threshold,
        topk_mean_threshold=args.topk_mean_threshold, k=args.decision_k,
    )

    report = {
        "video": str(video_path),
        "variant": "v1",
        "alpha": args.alpha,
        "tau": args.tau,
        "duration": total_duration,
        "feature_shape": list(segment_features.shape),
        "num_segments": args.num_segments,
        "scores": scores.tolist(),
        "decision": decision_info,
        "selected_segments": [],
        "time_ranges": [],
        "summary_created": False,
    }
    report_path = output_dir / f"{video_path.stem}_v1_report.json"

    if not anomaly_exists:
        print("\nSONUC: Anomali bulunamadi. Ozet video olusturulmadi.")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Rapor: {report_path}")
        return

    selected = select_segments(
        scores, mode=args.select_mode, top_k=args.top_k, nms_window=args.nms_window,
        segment_threshold=args.segment_threshold, budget_ratio=args.budget_ratio,
    )

    if not selected:
        print("\nSONUC: Video-level anomali var gibi ama esik ustu segment yok.")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Rapor: {report_path}")
        return

    ranges = segments_to_time_ranges(
        selected, num_segments=args.num_segments,
        total_duration=total_duration, context_sec=args.context_sec,
    )
    ranges = merge_time_ranges(ranges)

    output_video = output_dir / f"{video_path.stem}_v1_anomaly_summary.mp4"
    created = write_summary_video(video_path, ranges, output_video)

    report["select_mode"] = args.select_mode
    report["top_k"] = args.top_k
    report["nms_window"] = args.nms_window
    report["selected_segments"] = selected
    report["time_ranges"] = ranges
    report["summary_created"] = bool(created)
    report["summary_video"] = str(output_video)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSONUC: Anomali bulundu.")
    print("Secilen segmentler:", selected)
    print("Zaman araliklari:")
    for start, end in ranges:
        print(f"  {start:.2f}s - {end:.2f}s")
    print(f"Ozet video: {output_video}")
    print(f"Rapor: {report_path}")


if __name__ == "__main__":
    main()
