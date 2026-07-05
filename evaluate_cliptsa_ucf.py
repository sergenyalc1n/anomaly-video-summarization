import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm

from src.device import get_device
from src.cliptsa_adapter import CLIPTSAScorer
from src.feature_utils import process_feat


def is_normal_file(path):
    p = Path(path)
    parent = p.parent.name.lower()
    name = p.name.lower()
    return parent == "normal" or name.startswith("normal")


def aggregate_scores(scores, topk=3):
    scores = np.asarray(scores, dtype=np.float32)
    k = max(1, min(topk, len(scores)))

    return {
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "topk_mean": float(np.mean(np.sort(scores)[-k:])),
    }


def best_threshold_youden(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))

    return {
        "threshold": float(thresholds[idx]),
        "tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "youden_j": float(j[idx]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", default="data/ucf/features/test")
    parser.add_argument("--cliptsa_repo", default="third_party/CLIP-TSA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/ucf_video_level_eval.json")

    parser.add_argument("--feature_size", type=int, default=512)
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--k", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--disable_HA", action="store_true")
    parser.add_argument("--topk", type=int, default=3)

    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    if not feature_root.exists():
        raise FileNotFoundError(feature_root)

    files = sorted(feature_root.rglob("*.npy"))
    if not files:
        raise RuntimeError(f"Test feature bulunamadı: {feature_root}")

    device = get_device()
    print(f"Cihaz: {device}")
    print(f"Test video feature sayısı: {len(files)}")

    scorer = CLIPTSAScorer(
        repo_dir=args.cliptsa_repo,
        checkpoint=args.checkpoint,
        feature_size=args.feature_size,
        k=args.k,
        num_samples=args.num_samples,
        enable_HA=not args.disable_HA,
        device=device,
    )

    y_true = []
    max_scores = []
    mean_scores = []
    topk_mean_scores = []
    per_video = []

    for file in tqdm(files, desc="Test videoları skorlanıyor"):
        raw_features = np.load(file, allow_pickle=True).astype(np.float32)

        # Kritik düzeltme:
        # Eğitimde olduğu gibi her video sabit 32 segmente indiriliyor.
        features = process_feat(raw_features, length=args.num_segments)

        scores = scorer.score_features(features)

        label = 0 if is_normal_file(file) else 1
        aggr = aggregate_scores(scores, topk=args.topk)

        y_true.append(label)
        max_scores.append(aggr["max"])
        mean_scores.append(aggr["mean"])
        topk_mean_scores.append(aggr["topk_mean"])

        per_video.append({
            "file": str(file),
            "label": int(label),
            "label_name": "normal" if label == 0 else "abnormal",
            "raw_shape": list(raw_features.shape),
            "processed_shape": list(features.shape),
            "num_scores": int(len(scores)),
            "max_score": aggr["max"],
            "mean_score": aggr["mean"],
            "topk_mean_score": aggr["topk_mean"],
        })

    y_true = np.asarray(y_true, dtype=np.int32)

    results = {
        "checkpoint": args.checkpoint,
        "feature_root": str(feature_root),
        "num_videos": int(len(files)),
        "num_normal": int((y_true == 0).sum()),
        "num_abnormal": int((y_true == 1).sum()),
        "metrics": {},
        "per_video": per_video,
    }

    print("\nSınıf dağılımı:")
    print(f"Normal:   {results['num_normal']}")
    print(f"Anormal:  {results['num_abnormal']}")

    if results["num_normal"] == 0 or results["num_abnormal"] == 0:
        raise RuntimeError("AUC için hem normal hem anormal test videosu gerekir.")

    score_sets = {
        "max": np.asarray(max_scores),
        "mean": np.asarray(mean_scores),
        "topk_mean": np.asarray(topk_mean_scores),
    }

    print("\nVideo-level değerlendirme:")
    print("-" * 78)

    for name, values in score_sets.items():
        roc_auc = roc_auc_score(y_true, values)
        ap = average_precision_score(y_true, values)
        threshold_info = best_threshold_youden(y_true, values)

        results["metrics"][name] = {
            "roc_auc": float(roc_auc),
            "average_precision": float(ap),
            "best_threshold_youden": threshold_info,
        }

        print(
            f"{name:<12} "
            f"ROC-AUC={roc_auc:.4f} | "
            f"AP={ap:.4f} | "
            f"best_thr={threshold_info['threshold']:.4f} | "
            f"TPR={threshold_info['tpr']:.4f} | "
            f"FPR={threshold_info['fpr']:.4f}"
        )

    print("-" * 78)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nKaydedildi: {output}")


if __name__ == "__main__":
    main()
