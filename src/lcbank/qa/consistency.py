"""Label agreement between two labelling runs (§13 A2/A3): mean weight overlap and main-topic agreement."""
import json
from collections import defaultdict

from lcbank.common.paths import DATA_DERIVED


def overlap(w1, w2):
    a = {w["topic_id"]: w["weight"] for w in w1}
    b = {w["topic_id"]: w["weight"] for w in w2}
    return sum(min(a[k], b.get(k, 0)) for k in a) / 100.0


def main_topic(w):
    return max(w, key=lambda x: x["weight"])["topic_id"] if w else None


def compare(datasets, run_a, run_b):
    out = {}
    for ds in datasets:
        runs = defaultdict(dict)
        for line in open(DATA_DERIVED / "labels" / f"labels_raw_{ds}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["
                runs[r["run"]][r["item_id"]] = r["weights"]
        common = sorted(set(runs[run_a]) & set(runs[run_b]))
        if not common:
            continue
        ov = [overlap(runs[run_a][i], runs[run_b][i]) for i in common]
        agree = [main_topic(runs[run_a][i]) == main_topic(runs[run_b][i]) for i in common]
        out[ds] = {"n": len(common), "mean_overlap": sum(ov) / len(ov), "main_topic_agreement": sum(agree) / len(agree),
                   "disagreements": [(i, main_topic(runs[run_a][i]), main_topic(runs[run_b][i]))
                                     for i, g in zip(common, agree) if not g]}
    return out


if __name__ == "__main__":
    import sys
    a, b = sys.argv[1], sys.argv[2]
    res = compare(["PHY", "PM", "PMT", "AM2"], a, b)
    for ds, r in res.items():
        print(f"{ds}: n={r['n']} overlap={r['mean_overlap']:.3f} main_agree={r['main_topic_agreement']:.3f} "
              f"disagree={r['disagreements'][:4]}")
    n = sum(r["n"] for r in res.values())
    print(f"ALL: n={n} overlap={sum(r['mean_overlap'] * r['n'] for r in res.values()) / n:.3f} "
          f"main_agree={sum(r['main_topic_agreement'] * r['n'] for r in res.values()) / n:.3f}")
