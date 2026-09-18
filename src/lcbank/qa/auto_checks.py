"""Phase 6: automatic label QA checks A1-A5 and reports/label_qa.md (§13).

A1 validity            share of leaves with
A2 self-consistency    stratified 10% rerun (min 30)         pass overlap >= 0.85 and main-topic agreement >= 0.85
A3 second model        same sample with LLM2_* if configured pass overlap >= 0.80
A4 structure sanity    question number x main topic          report only, never change labels to fit
A5 distribution sanity topics never used; topics > 15% marks report only
Run: python -m lcbank.qa.auto_checks --run final [--rerun]
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from lcbank.common.paths import DATA_DERIVED, REPORTS
from lcbank.label.label_parts import load_items, run_labels
from lcbank.qa.consistency import main_topic, overlap

RERUN_SHARE, RERUN_MIN = 0.10, 30
THRESHOLDS = {"A1": 0.97, "A2_overlap": 0.85, "A2_main": 0.85, "A3_overlap": 0.80}
SEED = 20260107


def labels(dataset, run):
    path = DATA_DERIVED / "labels" / f"labels_raw_{dataset}.jsonl"
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["run"] == run:
                out[r["item_id"]] = r
    return out


def a1_validity(recs):
    n = len(recs)
    ok = sum(1 for r in recs.values() if r["
    return {"n": n, "ok": ok, "rate": (ok / n) if n else float("nan"),
            "pass": bool(n and ok / n >= THRESHOLDS["A1"]),
            "needs_human": [i for i, r in recs.items() if r["


def consistency_sample(dataset, recs, seed=SEED):
    """Stratified 10% sample (min 30) by year and question."""
    items = {it["item_id"]: it for it in load_items(dataset)}
    by_year = defaultdict(list)
    for iid in recs:
        if iid in items:
            by_year[items[iid]["year"]].append(iid)
    n_target = max(RERUN_MIN, int(round(RERUN_SHARE * len(recs))))
    rng = random.Random(seed)
    picked, years = [], sorted(by_year)
    while len(picked) < min(n_target, len(recs)):
        for y in years:
            pool = [i for i in by_year[y] if i not in picked]
            if pool:
                picked.append(rng.choice(pool))
            if len(picked) >= n_target:
                break
    return picked


def a2_self_consistency(dataset, run, rerun_run, do_rerun, version="v4s", pin_model=None):
    recs = labels(dataset, run)
    sample = consistency_sample(dataset, recs)
    if do_rerun:
        items = [it for it in load_items(dataset) if it["item_id"] in set(sample)]
        run_labels(dataset, items, rerun_run, version, workers=4, pin_model=pin_model)
    second = labels(dataset, rerun_run)
    common = [i for i in sample if i in second and recs[i]["
    if not common:
        return {"n": 0, "pass": None, "
    ov = [overlap(recs[i]["weights"], second[i]["weights"]) for i in common]
    ag = [main_topic(recs[i]["weights"]) == main_topic(second[i]["weights"]) for i in common]
    return {"n": len(common), "mean_overlap": sum(ov) / len(ov), "main_topic_agreement": sum(ag) / len(ag),
            "pass": bool(sum(ov) / len(ov) >= THRESHOLDS["A2_overlap"]
                         and sum(ag) / len(ag) >= THRESHOLDS["A2_main"]),
            "disagreements": [(i, main_topic(recs[i]["weights"]), main_topic(second[i]["weights"]))
                              for i, a in zip(common, ag) if not a][:10]}


def a4_structure(dataset, recs):
    """Question number x main topic cross-tab (report only)."""
    items = {it["item_id"]: it for it in load_items(dataset)}
    table = defaultdict(Counter)
    for iid, r in recs.items():
        if r["
            table[items[iid]["question"]][main_topic(r["weights"])] += 1
    return {q: dict(c.most_common(3)) for q, c in sorted(table.items(), key=lambda kv: (len(kv[0]), kv[0]))}


def a5_distribution(dataset, recs):
    """Topics never used and topics taking more than 15% of all labelled marks (report only)."""
    items = {it["item_id"]: it for it in load_items(dataset)}
    marks_by_topic, total = Counter(), 0.0
    for iid, r in recs.items():
        if r["
            continue
        m = items[iid]["marks"]
        total += m
        for w in r["weights"]:
            marks_by_topic[w["topic_id"]] += m * w["weight"] / 100.0
    cb_name = {"PHY": "PHY", "PM": "PM", "PMT": "PM", "AM2": "AM2"}[dataset]
    cb = json.loads((DATA_DERIVED / "codebooks" / f"{cb_name}_v1.json").read_text())
    all_topics = [t["topic_id"] for t in cb["topics"]]
    never = [t for t in all_topics if marks_by_topic.get(t, 0) == 0]
    heavy = {t: round(v / total, 4) for t, v in marks_by_topic.items() if total and v / total > 0.15}
    unclear = round(marks_by_topic.get("UNCLEAR", 0) / total, 4) if total else 0
    return {"topics_never_used": never, "topics_over_15pc": heavy, "unclear_share_of_marks": unclear}


def run_checks(datasets=("PHY", "PM", "PMT", "AM2"), run="final", do_rerun=False, version="v4s", pin_model=None):
    out = {}
    for ds in datasets:
        recs = labels(ds, run)
        if not recs:
            out[ds] = {"
            continue
        out[ds] = {"A1": a1_validity(recs),
                   "A2": a2_self_consistency(ds, run, f"{run}_rerun", do_rerun, version, pin_model),
                   "A4": a4_structure(ds, recs), "A5": a5_distribution(ds, recs)}
    write_report(out, run)
    summary = {ds: {"A1": {k: v for k, v in r["A1"].items() if k != "needs_human"},
                    "A2": {k: v for k, v in r["A2"].items() if k != "disagreements"},
                    "A5": r["A5"]}
               for ds, r in out.items() if not r.get("}
    (DATA_DERIVED / "labels" / "qa_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return out


def write_report(results, run):
    lines = ["# Label QA (Phase 6)", "",
             f"Generated by `python -m lcbank.qa.auto_checks --run {run}`. A1 and A2 have pass thresholds; "
             "A4 and A5 are reported only and never used to change labels (§13).", "",
             "| Dataset | A1 validity | A1 pass | A2 pairs | A2 overlap | A2 main-topic | A2 pass | UNCLEAR share |",
             "|---|---|---|---|---|---|---|---|"]
    for ds, r in results.items():
        if r.get(":
            lines.append(f"| {ds} | — | — | — | — | — | — | {r['
            continue
        a1, a2, a5 = r["A1"], r["A2"], r["A5"]
        lines.append(f"| {ds} | {a1['rate']:.3f} ({a1['ok']}/{a1['n']}) | {'yes' if a1['pass'] else 'NO'} | "
                     f"{a2.get('n', 0)} | {a2.get('mean_overlap', float('nan')):.2f} | "
                     f"{a2.get('main_topic_agreement', float('nan')):.2f} | "
                     f"{'yes' if a2.get('pass') else ('NO' if a2.get('pass') is False else '—')} | "
                     f"{a5['unclear_share_of_marks']:.2%} |")
    for ds, r in results.items():
        if r.get(":
            continue
        lines += ["", f"## {ds}", "",
                  f"- Leaves needing human review (A1): {len(r['A1']['needs_human'])}"
                  + (f" — e.g. {r['A1']['needs_human'][:5]}" if r['A1']['needs_human'] else ""),
                  f"- A5 topics never used: {r['A5']['topics_never_used'] or 'none'}",
                  f"- A5 topics above 15% of marks: {r['A5']['topics_over_15pc'] or 'none'}",
                  f"- A2 disagreements (sample): {r['A2'].get('disagreements', [])[:5]}"]
    (REPORTS / "label_qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final")
    ap.add_argument("--datasets", default="PHY,PM,PMT,AM2")
    ap.add_argument("--rerun", action="store_true", help="run the A2 sample again through the API")
    ap.add_argument("--version", default="v4s")
    ap.add_argument("--pin-model", default=None)
    a = ap.parse_args()
    res = run_checks(tuple(a.datasets.split(",")), a.run, a.rerun, a.version, a.pin_model)
    print(json.dumps({k: (v if v.get(" else {"A1": v["A1"]["rate"], "A2": v["A2"].get("mean_overlap")})
                      for k, v in res.items()}, indent=1))
