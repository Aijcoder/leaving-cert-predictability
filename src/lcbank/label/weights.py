"""Topic weights from scoring steps (the build specification Appendix B.1, binding).

The model outputs steps; this script computes percentages.
"""
MAX_TOPICS, MIN_WEIGHT, STEP = 3, 20, 10


def compute_weights(steps):
    per_topic = {}
    for s in steps:
        per_topic[s["topic_id"]] = per_topic.get(s["topic_id"], 0.0) + float(s["marks"])
    total = sum(per_topic.values())
    if total == 0:
        return []
    shares = sorted(((t, 100.0 * m / total) for t, m in per_topic.items()), key=lambda kv: -kv[1])
    kept = [(t, sh) for i, (t, sh) in enumerate(shares) if i < MAX_TOPICS and (i == 0 or sh >= MIN_WEIGHT)]
    kept_total = sum(sh for _, sh in kept)
    scaled = {t: sh * 100.0 / kept_total for t, sh in kept}
    weights = {t: int(v // STEP) * STEP for t, v in scaled.items()}
    leftover = (100 - sum(weights.values())) // STEP
    for t in sorted(scaled, key=lambda t: -(scaled[t] - weights[t]))[:leftover]:
        weights[t] += STEP
    return [{"topic_id": t, "weight": w} for t, w in sorted(weights.items(), key=lambda kv: -kv[1])]
