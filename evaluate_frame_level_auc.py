"""
Mantik:
  1. Her test videosu icin model 32 segment skoru uretir.
  2. Bu 32 skor, videonun orijinal frame sayisina (N) yayilir -> frame-level skor.
  3. Temporal anotasyondan frame-level 0/1 ground-truth kurulur.
  4. Tum test videolarinin frame'leri birlestirilip tek bir ROC-AUC hesaplanir.

Gerekenler:
  --annotation    : Temporal_Anomaly_Annotation_for_Testing_Videos.txt
                    (satir: <Name>.mp4  <Class>  s1 e1 s2 e2 ; kullanilmayan = -1)
  --feature_root  : test feature klasoru (features/test). .npy'ler stem ile bulunur.
  --frame_counts  : Colab extractor'in urettigi frame_counts.json
                    (stem -> {"frames": N, "fps": F}  veya  stem -> N)
                    Alternatif: --videos_dir vererek frame sayisi videolardan okunur.
  --checkpoint    : egitilmis CLIP-TSA checkpoint
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.device import get_device
from src.cliptsa_adapter import CLIPTSAScorer
from src.feature_utils import process_feat


def parse_temporal_annotations(path):
    ann = {}
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            name, cls = parts[0], parts[1]
            nums = []
            for tok in parts[2:6]:
                try:
                    nums.append(int(tok))
                except ValueError:
                    nums.append(-1)
            while len(nums) < 4:
                nums.append(-1)
            stem = Path(name).stem
            ann[stem] = {
                "class": cls,
                "spans": [(nums[0], nums[1]), (nums[2], nums[3])],
            }
    return ann


def load_frame_counts(frame_counts_path, videos_dir):
    counts = {}
    if frame_counts_path and Path(frame_counts_path).exists():
        with open(frame_counts_path) as f:
            raw = json.load(f)
        for stem, v in raw.items():
            counts[Path(stem).stem] = int(v["frames"]) if isinstance(v, dict) else int(v)
        return counts

    if videos_dir:
        import cv2
        for vp in Path(videos_dir).rglob("*"):
            if vp.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"):
                cap = cv2.VideoCapture(str(vp))
                counts[vp.stem] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
        return counts

    raise SystemExit("Frame sayisi yok: --frame_counts veya --videos_dir ver.")


def find_feature(feature_root, stem):
    hits = list(Path(feature_root).rglob(stem + ".npy"))
    return hits[0] if hits else None


def expand_to_frames(seg_scores, n_frames):
    seg = np.asarray(seg_scores, dtype=np.float32)
    s = len(seg)
    if n_frames <= 0 or s == 0:
        return np.zeros(max(n_frames, 0), dtype=np.float32)
    idx = (np.arange(n_frames) * s // n_frames).clip(0, s - 1)
    return seg[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--feature_root", default="data/ucf/features/test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame_counts", default=None)
    parser.add_argument("--videos_dir", default=None)
    parser.add_argument("--output", default="outputs/ucf_frame_level_eval.json")
    parser.add_argument("--cliptsa_repo", default="third_party/CLIP-TSA")
    parser.add_argument("--feature_size", type=int, default=512)
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--k", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--disable_HA", action="store_true")
    args = parser.parse_args()

    ann = parse_temporal_annotations(args.annotation)
    print(f"Anotasyondaki test videosu: {len(ann)}")

    counts = load_frame_counts(args.frame_counts, args.videos_dir)
    print(f"Frame sayisi bilinen video: {len(counts)}")

    device = get_device()
    print(f"Cihaz: {device}")

    scorer = CLIPTSAScorer(
        repo_dir=args.cliptsa_repo,
        checkpoint=args.checkpoint,
        feature_size=args.feature_size,
        k=args.k,
        num_samples=args.num_samples,
        enable_HA=not args.disable_HA,
        device=device,
    )

    all_scores, all_labels = [], []
    used, missing_feat, missing_count = 0, [], []
    per_video = []

    for stem, info in ann.items():
        feat_path = find_feature(args.feature_root, stem)
        if feat_path is None:
            missing_feat.append(stem)
            continue
        if stem not in counts:
            missing_count.append(stem)
            continue

        n_frames = counts[stem]
        raw = np.load(feat_path, allow_pickle=True).astype(np.float32)
        feats = process_feat(raw, length=args.num_segments)
        seg_scores = scorer.score_features(feats)

        frame_scores = expand_to_frames(seg_scores, n_frames)
        gt = np.zeros(n_frames, dtype=np.int32)
        for s, e in info["spans"]:
            if s >= 0 and e >= 0:
                gt[s:min(e + 1, n_frames)] = 1

        all_scores.append(frame_scores)
        all_labels.append(gt)
        used += 1
        per_video.append({
            "stem": stem,
            "class": info["class"],
            "n_frames": int(n_frames),
            "anomalous_frames": int(gt.sum()),
            "max_score": float(seg_scores.max()),
        })

    if used == 0:
        raise SystemExit("Hicbir video eslesmedi. feature_root / frame_counts / anotasyonu kontrol et.")

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    frame_auc = float(roc_auc_score(y_true, y_score))
    frame_ap = float(average_precision_score(y_true, y_score))

    results = {
        "frame_level_roc_auc": frame_auc,
        "frame_level_average_precision": frame_ap,
        "videos_used": used,
        "total_frames": int(len(y_true)),
        "anomalous_frames": int(y_true.sum()),
        "missing_feature": missing_feat,
        "missing_frame_count": missing_count,
        "per_video": per_video,
    }

    print("\n" + "=" * 60)
    print(f"FRAME-LEVEL ROC-AUC : {frame_auc:.4f}   <-- SOTA (~0.87) ile kiyasla")
    print(f"Average Precision   : {frame_ap:.4f}")
    print(f"Kullanilan video    : {used}")
    if missing_feat:
        print(f"Feature bulunamayan : {len(missing_feat)} (ilk 5: {missing_feat[:5]})")
    if missing_count:
        print(f"Frame sayisi yok    : {len(missing_count)} (ilk 5: {missing_count[:5]})")
    print("=" * 60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
