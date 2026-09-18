"""Phase 8.7: simulator and controls (§15.7).

Synthetic sittings reuse each dataset's real rules trees, leaf marks, leaf-topic structure, topic count and number of
sittings; only which topics appear is simulated.
  G0: per-topic appearance rate ~ Beta(2, 3), no temporal effect (true timing gain = 0).
  G1: log-odds + delta * 1[d >= 3], delta in {0.5, 1.0, 1.5}.
Controls: false-positive rate of H2/H3 under G0 at alpha = 0.05 must be 2-8%; power at delta = 1.5 >= 80%;
bootstrap 95% CI coverage of the true H2 value (0 under G0) must be 90-98%.
Run: python -m lcbank.analysis.sim --runs 500 --perm 199 --workers 8
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from lcbank.analysis.data import APPEARS_THRESHOLD, DatasetData, Sitting, load_dataset
from lcbank.analysis.evaluate import SEEDS
from lcbank.analysis.inference import (h2_statistic, h3_beta, moving_block_bootstrap, permutation_p,
                                       permute_sitting_order)
from lcbank.common.paths import ANALYSIS

ALPHA = 0.05
K_DEFAULT = 100
MIN_TRAIN = {"PHY": 8, "PM": 5}


def draw_appears(T, n_sittings, rng, delta=0.0):
    """Appearance matrix under G0 (delta = 0) or G1 (delta > 0)."""
    p = np.clip(rng.beta(2, 3, size=T), 1e-3, 1 - 1e-3)
    base_logit = np.log(p / (1 - p))
    appears = np.zeros((n_sittings, T), dtype=bool)
    for i in range(n_sittings):
        logit = base_logit.copy()
        if delta and i > 0:
            gaps = np.zeros(T, dtype=int)
            for k in range(T):
                idx = np.flatnonzero(appears[:i, k])
                gaps[k] = (i - idx[-1]) if len(idx) else 0
            logit += delta * (gaps >= 3)   # §15.7 G1: +delta on the log-odds when d >= 3 (never-appeared: no boost)
        appears[i] = rng.random(T) < 1 / (1 + np.exp(-logit))
        if not appears[i].any():
            appears[i, rng.integers(T)] = True
    return appears


def assign_topics(template, appears, rng):
    """Give each sitting's leaves topics drawn from that sitting's appearing set, keeping the template's
    weight shapes (e.g. 60/40) and marks. Round-robin keeps every appearing topic above the 1.5% threshold."""
    sittings = []
    for i, tmpl in enumerate(template.sittings):
        present = np.flatnonzero(appears[i])
        order = rng.permutation(len(tmpl.items))
        items, pos = [], 0
        for r in order:
            it = tmpl.items[r]
            shape = [w["weight"] for w in (it["topics"] or [{"weight": 100}])]
            picks = [present[(pos + j) % len(present)] for j in range(len(shape))]
            pos += 1
            uniq = {}
            for t, w in zip(picks, shape):
                uniq[t] = uniq.get(t, 0) + w
            items.append({"item_id": it["item_id"], "marks": it["marks"],
                          "topics": [{"topic_id": template.topics[t], "weight": w} for t, w in uniq.items()]})
        sittings.append(Sitting(tmpl.year, tmpl.root, items, tmpl.max_score, tmpl.max_score_all))
    return DatasetData(template.name, template.topics, sittings)


def one_run(args):
    name, template, delta, seed, n_perm, K = args
    rng = np.random.default_rng(seed)
    drawn = draw_appears(len(template.topics), len(template.sittings), rng, delta)
    data = assign_topics(template, drawn, rng)
    realised = data.appears
    mismatch = float(np.mean(drawn != realised))
    min_train = MIN_TRAIN[name]
    h2_obs, h2_per, chosen = h2_statistic(data, min_train, K=K)
    h3_obs = h3_beta([data])
    perm_rng = np.random.default_rng(seed + 991)
    h2_null, h3_null = [], []
    for _ in range(n_perm):
        shuffled = permute_sitting_order(data, perm_rng)
        h2_null.append(h2_statistic(shuffled, min_train, K=K)[0])
        h3_null.append(h3_beta([shuffled]))
    lo, hi = moving_block_bootstrap(h2_per)
    return {"dataset": name, "delta": delta, "seed": seed, "mismatch": mismatch, "chosen": chosen,
            "h2": h2_obs, "h2_p": permutation_p(h2_obs, h2_null),
            "h3": h3_obs, "h3_p": permutation_p(h3_obs, h3_null, one_sided=False),
            "h2_ci_lo": lo, "h2_ci_hi": hi, "h2_ci_covers_zero": bool(lo <= 0 <= hi)}


def recompute_ci(record, templates, K=100, method="t_blocks"):
    """Recompute one stored run's H2 interval under a different interval method (§15.7 rerun after EX-008).

    The seed regenerates exactly the same synthetic dataset, so the permutation results in the record still
    describe this run; only the interval changes, and the interval never depended on the permutations.
    """
    name, seed = record["dataset"], record["seed"]
    rng = np.random.default_rng(seed)
    template = templates[name]
    drawn = draw_appears(len(template.topics), len(template.sittings), rng, record["delta"])
    data = assign_topics(template, drawn, rng)
    h2_obs, h2_per, chosen = h2_statistic(data, MIN_TRAIN[name], K=K)
    if abs(h2_obs - record["h2"]) > 1e-9 or chosen != record["chosen"]:
        raise ValueError(f"replay mismatch for {name} seed {seed}: the stored run is not reproducible")
    lo, hi = moving_block_bootstrap(h2_per, method=method)
    return {**record, "h2_ci_lo": lo, "h2_ci_hi": hi, "h2_ci_covers_zero": bool(lo <= 0 <= hi),
            "ci_method": method}


def run(datasets=("PHY", "PM"), runs=500, deltas=(0.0, 0.5, 1.0, 1.5), n_perm=199, K=100, workers=8, out=None,
        shard=0, shards=1):
    # labelled=True: §15.7 requires the real leaf-topic structure, and only the labelled items carry the
    # weight shapes (a 60/40 leaf stays a 60/40 leaf). Unlabelled items would make every leaf single-topic.
    templates = {name: load_dataset(name, labelled=True) for name in datasets}
    jobs = []
    for name in datasets:
        for delta in deltas:
            for r in range(runs):
                jobs.append((name, templates[name], delta, SEEDS["simulator"] + 1000 * int(delta * 10) + r, n_perm, K))
    jobs = jobs[shard::shards]
    out = out or ANALYSIS / "sim" / (f"simulator_runs_{shard}.jsonl" if shards > 1 else "simulator_runs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    results, done = [], set()
    if out.exists():  # resumable: skip runs already on disk
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            results.append(r)
            done.add((r["dataset"], r["delta"], r["seed"]))
    jobs = [j for j in jobs if (j[0], j[2], j[3]) not in done]
    print(f"{len(done)} runs already done; {len(jobs)} to run", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        if workers <= 1:   # one process per shard: simplest and robust on macOS
            for n, job in enumerate(jobs, 1):
                r = one_run(job)
                results.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if n % 10 == 0 or n == len(jobs):
                    el = time.time() - t0
                    print(f"shard {shard}: {n}/{len(jobs)} runs, {el/60:.1f} min elapsed, "
                          f"{el/n*(len(jobs)-n)/60:.1f} min left", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for n, r in enumerate(pool.map(one_run, jobs, chunksize=1), 1):
                    results.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    if n % 25 == 0 or n == len(jobs):
                        el = time.time() - t0
                        print(f"{n}/{len(jobs)} runs, {el/60:.1f} min elapsed, {el/n*(len(jobs)-n)/60:.1f} min left",
                              flush=True)
    return results


def summarise(results):
    summary = {}
    for name in sorted({r["dataset"] for r in results}):
        for delta in sorted({r["delta"] for r in results}):
            sub = [r for r in results if r["dataset"] == name and r["delta"] == delta]
            if not sub:
                continue
            summary[f"{name}_delta{delta}"] = {
                "runs": len(sub),
                "H2_reject_rate": float(np.mean([r["h2_p"] <= ALPHA for r in sub])),
                "H3_reject_rate": float(np.mean([r["h3_p"] <= ALPHA for r in sub])),
                "H2_ci_covers_zero": float(np.mean([r["h2_ci_covers_zero"] for r in sub])),
                "mean_h2": float(np.mean([r["h2"] for r in sub])),
                "mean_h3_beta": float(np.mean([r["h3"] for r in sub])),
                "appears_mismatch": float(np.mean([r["mismatch"] for r in sub])),
            }
    checks = {}
    for name in sorted({r["dataset"] for r in results}):
        g0 = summary.get(f"{name}_delta0.0")
        g1 = summary.get(f"{name}_delta1.5")
        if g0:
            checks[f"{name}: G0 H2 false-positive 2-8%"] = 0.02 <= g0["H2_reject_rate"] <= 0.08
            checks[f"{name}: G0 H3 false-positive 2-8%"] = 0.02 <= g0["H3_reject_rate"] <= 0.08
            checks[f"{name}: G0 bootstrap coverage 90-98%"] = 0.90 <= g0["H2_ci_covers_zero"] <= 0.98
        if g1:
            checks[f"{name}: G1 delta=1.5 power >= 80%"] = g1["H3_reject_rate"] >= 0.80
    return summary, checks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--perm", type=int, default=199)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", default="PHY,PM")
    ap.add_argument("--deltas", default="0.0,0.5,1.0,1.5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--recompute-ci", default=None, metavar="METHOD",
                    help="replay the stored runs and rewrite only the H2 interval with this method")
    a = ap.parse_args()
    if a.recompute_ci:
        templates = {n: load_dataset(n, labelled=True) for n in a.datasets.split(",")}
        files = sorted((ANALYSIS / "sim").glob("simulator_runs*.jsonl"))
        done = 0
        for path in files:
            records = [json.loads(l) for l in open(path, encoding="utf-8")]
            fixed = [recompute_ci(r, templates, K=K_DEFAULT, method=a.recompute_ci) for r in records]
            path.write_text("".join(json.dumps(r) + "\n" for r in fixed), encoding="utf-8")
            done += len(fixed)
            print(f"{path.name}: {len(fixed)} runs rewritten ({done} total)", flush=True)
        summary, checks = summarise([json.loads(l) for p in files for l in open(p, encoding="utf-8")])
        print(json.dumps(checks, indent=1))
        raise SystemExit(0)
    res = run(tuple(a.datasets.split(",")), a.runs, tuple(float(d) for d in a.deltas.split(",")), a.perm,
              workers=a.workers, out=(ANALYSIS / "sim" / a.out) if a.out else None, shard=a.shard, shards=a.shards)
    summary, checks = summarise(res)
    print(json.dumps(summary, indent=1))
    print(json.dumps(checks, indent=1))
