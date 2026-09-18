"""Phase 8.8-8.9: the real-data run — AS curves, decomposition, hypothesis tests, sensitivity analyses (§15.8-15.9).

Refuses to run until the analysis plan is locked (§15.1), the labels are frozen (R7) and the simulator controls have
passed (§15.7). Everything here is retrospective (R1): the last test sitting is the last available sitting.
Run: python -m lcbank.analysis.run_analysis [--quick]
"""
import argparse
import hashlib
import json
from statistics import mean

import numpy as np

from lcbank.analysis.data import load_dataset
from lcbank.analysis.evaluate import RHOS, SEEDS, decomposition, evaluate_dataset, rho80
from lcbank.analysis.inference import (h1_statistic, h2_statistic, h3_beta, holm, moving_block_bootstrap, negligible,
                                       permutation_p, permute_sitting_order, permute_within_sitting)
from lcbank.analysis.sim import summarise as sim_summarise
from lcbank.common.paths import ANALYSIS, DATA_DERIVED, REPORTS
from lcbank.structure.rules import all_compulsory, max_score

ROLES = {"PHY": "formal", "PM": "exploratory"}
MIN_TRAIN = {"PHY": 8, "PM": 5}
RESULTS = ANALYSIS / "results"
STATEMENT = "Retrospective analysis of past papers only. No prediction of any future examination."


def gate_checks():
    """Refuse to produce real-data results unless the preconditions hold (§15.1, §15.7, R7)."""
    problems = []
    lock = json.loads((ANALYSIS / "plan.lock").read_text())
    if hashlib.sha256((ANALYSIS / "plan.md").read_bytes()).hexdigest() != lock["sha256"]:
        problems.append("analysis/plan.md changed after the lock (record it in")
    frozen = DATA_DERIVED / "labels" / "FROZEN.json"
    if not frozen.exists():
        problems.append("labels are not frozen: run `python -m lcbank.label.merge_labels --freeze`")
    sim_files = sorted((ANALYSIS / "sim").glob("simulator_runs*.jsonl"))
    if not sim_files:
        problems.append("simulator has not been run (§15.7)")
    else:
        runs = [json.loads(l) for p in sim_files for l in open(p, encoding="utf-8")]
        summary, checks = sim_summarise(runs)
        short = {k: v["runs"] for k, v in summary.items() if v["runs"] < 500}
        if short:
            problems.append(f"simulator incomplete (500 runs per scenario required, §15.7): {short}")
        failed = [k for k, ok in checks.items() if not ok]
        if not checks:
            problems.append("simulator produced no usable control checks")
        elif failed:
            problems.append(f"simulator controls failed: {failed}")
    return problems


def dataset_results(name, credit="proportional", K=1000, threshold=None, exclude_years=(), drop_needs_human=False,
                    collapse_l1=False, random_draws=1000):
    data = load_dataset(name, labelled=True)
    if drop_needs_human:
        # The leaf stays in the rules tree - it is on the paper - but carries no marks and no topics, so it earns
        # nothing and counts for nothing in the denominator. Removing it outright breaks the tree (KeyError).
        kept = []
        for s in data.sittings:
            items = [i if i.get("label_ == "ok" else {**i, "marks": 0.0, "topics": []} for i in s.items]
            marks = {i["item_id"]: i["marks"] for i in items}
            kept.append(type(s)(s.year, s.root, items, max_score(s.root, marks),
                                max_score(all_compulsory(s.root), marks)))
        data = type(data)(data.name, data.topics, kept)
    if collapse_l1:
        cb = json.loads((DATA_DERIVED / "codebooks" / f"{'PM' if name == 'PMT' else name}_v1.json").read_text())
        l1 = {t["topic_id"]: t["L1"] for t in cb["topics"]}
        for s in data.sittings:
            for it in s.items:
                merged = {}
                for w in it["topics"] or []:
                    key = l1.get(w["topic_id"], w["topic_id"])
                    merged[key] = merged.get(key, 0) + w["weight"]
                it["topics"] = [{"topic_id": k, "weight": v} for k, v in merged.items()]
        data = type(data)(data.name, sorted({v for v in l1.values()}), data.sittings)
    if exclude_years:
        keep = [s for s in data.sittings if s.year not in set(exclude_years)]
        data = type(data)(data.name, data.topics, keep)
    if threshold:
        data = data.with_threshold(threshold)
    rows = evaluate_dataset(data, MIN_TRAIN[name], credit=credit, K=K, random_draws=random_draws)
    return data, rows


def curves_and_decomposition(rows):
    curves = {}
    for method in sorted({r["method"] for r in rows}):
        curve = {}
        for rho in RHOS:
            vals = [r["AS"] for r in rows if r["method"] == method and r["rho"] == rho]
            lo, hi = moving_block_bootstrap(vals) if len(vals) > 2 else (float("nan"), float("nan"))
            curve[rho] = {"mean": float(mean(vals)), "ci_lo": lo, "ci_hi": hi, "n_sittings": len(vals)}
        curves[method] = curve
    dec = decomposition(rows)
    per_sitting = {}
    for key, (a, b) in {"SQ1_question_choice": ("RANDOM", "RANDOM_ALL_COMPULSORY"),
                        "SQ2_frequency_gain": ("B1", "RANDOM"),
                        "SQ3_timing_gain": ("NESTED_TIMING", "B1")}.items():
        diffs = []
        for year in sorted({r["year"] for r in rows}):
            av = [r["AS"] for r in rows if r["method"] == a and r["year"] == year]
            bv = [r["AS"] for r in rows if r["method"] == b and r["year"] == year]
            if av and bv:
                diffs.append(mean(av) - mean(bv))
        per_sitting[key] = diffs
        lo, hi = moving_block_bootstrap(diffs) if len(diffs) > 2 else (float("nan"), float("nan"))
        dec[key] = {"mean": dec[key], "ci_lo": lo, "ci_hi": hi,
                    "practically_negligible": negligible(diffs) if len(diffs) > 2 else None}
    return curves, dec, per_sitting


def hypothesis_tests(datasets, n_perm=2000, K=100):
    """H1, H2 per formal dataset and H3 pooled, with permutation nulls and Holm correction (§15.8)."""
    out, pvalues = {}, {}
    formal = [name for name in datasets if ROLES.get(name) == "formal"]
    loaded = {name: load_dataset(name, labelled=True) for name in formal}
    for name in formal:
        data = loaded[name]
        obs1, per1 = h1_statistic(data, MIN_TRAIN[name], K=K)
        rng = np.random.default_rng(SEEDS["permutation"])
        null1 = [h1_statistic(permute_within_sitting(data, rng), MIN_TRAIN[name], K=K)[0] for _ in range(n_perm)]
        p1 = permutation_p(obs1, null1)
        obs2, per2, chosen = h2_statistic(data, MIN_TRAIN[name], K=K)
        rng2 = np.random.default_rng(SEEDS["permutation"] + 1)
        null2 = [h2_statistic(permute_sitting_order(data, rng2), MIN_TRAIN[name], K=K)[0] for _ in range(n_perm)]
        p2 = permutation_p(obs2, null2)
        out[f"H1_{name}"] = {"statistic": obs1, "p_raw": p1, "per_sitting": per1,
                             "ci": moving_block_bootstrap(per1)}
        out[f"H2_{name}"] = {"statistic": obs2, "p_raw": p2, "best_method": chosen, "per_sitting": per2,
                             "ci": moving_block_bootstrap(per2)}
        pvalues[f"H1_{name}"] = p1
        pvalues[f"H2_{name}"] = p2
    if formal:
        pooled = [loaded[n] for n in formal]
        obs3 = h3_beta(pooled)
        rng3 = np.random.default_rng(SEEDS["permutation"] + 2)
        null3 = [h3_beta([permute_sitting_order(d, rng3) for d in pooled]) for _ in range(n_perm)]
        p3 = permutation_p(obs3, null3, one_sided=False)
        out["H3_pooled"] = {"statistic": obs3, "p_raw": p3}
        pvalues["H3_pooled"] = p3
    corrected = holm(pvalues)
    for k, v in corrected.items():
        out[k]["p_holm"] = v
        out[k]["significant_holm_0.05"] = bool(v <= 0.05)
    return out


def run(datasets=("PHY", "PM"), quick=False, ignore_gates=False):
    problems = gate_checks()
    if problems and not ignore_gates:
        raise SystemExit("Cannot produce real-data results yet:\n- " + "\n- ".join(problems))
    RESULTS.mkdir(parents=True, exist_ok=True)
    K, n_perm, draws = (100, 200, 200) if quick else (1000, 2000, 1000)
    main, sens = {}, {}
    for name in datasets:
        data, rows = dataset_results(name, K=K, random_draws=draws)
        curves, dec, per_sitting = curves_and_decomposition(rows)
        main[name] = {"role": ROLES.get(name, "exploratory"), "sittings": data.years,
                      "test_sittings": data.years[MIN_TRAIN[name]:], "topics": len(data.topics),
                      "curves": curves, "decomposition": dec,
                      "AS_0.5": {m: curves[m][0.5]["mean"] for m in curves},
                      "rho80": {m: rho80({r: curves[m][r]["mean"] for r in RHOS}) for m in curves}}
        with open(RESULTS / f"as_curves_{name}.csv", "w", encoding="utf-8") as f:
            f.write("dataset,year,method,rho,AS\n")
            for r in rows:
                f.write(f"{r['dataset']},{r['year']},{r['method']},{r['rho']},{r['AS']:.6f}\n")
        for label, kwargs in {"strict_credit": {"credit": "strict"},
                              "threshold_0p75": {"threshold": 0.0075},
                              "threshold_3p0": {"threshold": 0.03},
                              "exclude_2020_2022": {"exclude_years": (2020, 2021, 2022)},
                              "drop_needs_human": {"drop_needs_human": True},
                              "collapse_L1": {"collapse_l1": True}}.items():
            _, srows = dataset_results(name, K=min(K, 200), random_draws=min(draws, 200), **kwargs)
            _, sdec, _ = curves_and_decomposition(srows)
            sens.setdefault(name, {})[label] = {k: (v["mean"] if isinstance(v, dict) else v) for k, v in sdec.items()}
    tests = hypothesis_tests(datasets, n_perm=n_perm, K=min(K, 100))
    payload = {"statement": STATEMENT, "main": main, "sensitivity": sens, "tests": tests,
               "plan_lock": json.loads((ANALYSIS / "plan.lock").read_text())}
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    write_report(payload)
    return payload


def write_report(payload):
    main, tests, sens = payload["main"], payload["tests"], payload["sensitivity"]
    lines = ["# Retrospective predictability analysis", "", f"**{payload['statement']}**", "",
             "Generated by `python -m lcbank.analysis.run_analysis`. "
             "Only the Holm table for H1-H3 on formal datasets is "
             "confirmatory; everything else is exploratory.", "",
             "## Attemptable share at ρ = 0.5 (mean over test sittings)", "",
             "| Dataset | Role | Test sittings | Random | B1 base rate | Nested best timing | ρ80 (B1) |",
             "|---|---|---|---|---|---|---|"]
    for name, m in main.items():
        lines.append(f"| {name} | {m['role']} | {len(m['test_sittings'])} ({m['test_sittings'][0]}-{m['test_sittings'][-1]}) | "
                     f"{m['AS_0.5'].get('RANDOM', float('nan')):.3f} | {m['AS_0.5'].get('B1', float('nan')):.3f} | "
                     f"{m['AS_0.5'].get('NESTED_TIMING', float('nan')):.3f} | {m['rho80'].get('B1')} |")
    lines += ["", "## Decomposition (mean over test sittings and ρ, 95% CI)", "",
              "| Dataset | SQ1 question choice | SQ2 frequency gain | SQ3 timing gain |", "|---|---|---|---|"]
    for name, m in main.items():
        cells = []
        for key in ("SQ1_question_choice", "SQ2_frequency_gain", "SQ3_timing_gain"):
            d = m["decomposition"][key]
            neg = " (negligible)" if d.get("practically_negligible") else ""
            cells.append(f"{d['mean']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]{neg}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Hypothesis tests (confirmatory, Holm over 3 tests)", "",
              "| Test | Statistic | p (permutation) | p (Holm) | Significant at 0.05 |", "|---|---|---|---|---|"]
    for k, v in tests.items():
        lines.append(f"| {k} | {v['statistic']:+.4f} | {v['p_raw']:.4f} | {v['p_holm']:.4f} | "
                     f"{'yes' if v['significant_holm_0.05'] else 'no'} |")
    lines += ["", "## Sensitivity analyses (exploratory)", "",
              "| Dataset | Variant | SQ1 | SQ2 | SQ3 |", "|---|---|---|---|---|"]
    for name, variants in sens.items():
        for label, d in variants.items():
            lines.append(f"| {name} | {label} | {d['SQ1_question_choice']:+.3f} | {d['SQ2_frequency_gain']:+.3f} | "
                         f"{d['SQ3_timing_gain']:+.3f} |")
    lines += ["", "## Limitations", "",
              "- Topic labels come from one language model.",
              "- PM is exploratory by design; AM2 and PMT are not analysed.",
              "- 2020-2022 sittings were adjusted for COVID; a sensitivity analysis excludes them.",
              "- The 2020 papers' month (June or the rescheduled November sitting) could not be determined (OI-008).",
              "", f"**{payload['statement']}**"]
    (REPORTS / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller K and fewer permutations (development only)")
    ap.add_argument("--ignore-gates", action="store_true", help="development only; never for reported results")
    ap.add_argument("--datasets", default="PHY,PM")
    a = ap.parse_args()
    res = run(tuple(a.datasets.split(",")), a.quick, a.ignore_gates)
    print(json.dumps({k: v["AS_0.5"] for k, v in res["main"].items()}, indent=1, default=float))
