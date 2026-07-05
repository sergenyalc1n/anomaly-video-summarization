import numpy as np


def has_anomaly(
    scores,
    max_threshold=0.5472,
    topk_mean_threshold=0.4576,
    k=3,
):
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:
        return False, {
            "max_score": 0.0,
            "topk_mean": 0.0,
            "decision": False,
        }

    k = max(1, min(k, len(scores)))
    max_score = float(scores.max())
    topk_mean = float(np.mean(np.sort(scores)[-k:]))

    decision = max_score >= max_threshold and topk_mean >= topk_mean_threshold

    return decision, {
        "max_score": max_score,
        "topk_mean": topk_mean,
        "decision": bool(decision),
        "max_threshold": max_threshold,
        "topk_mean_threshold": topk_mean_threshold,
        "k": k,
    }
