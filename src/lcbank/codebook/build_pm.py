"""Phase 4 (PM, shared by PMT): codebook PM_v1 from the LC Mathematics syllabus (examination from 2015) headings.

Topics are the syllabus sub-topics (1.1-5.2); four large sub-topics are split along their own learning-outcome
bullets, and complex-number bullets from 3.1 join 4.4. Every OL/HL bullet is assigned to exactly one topic.
Designed without looking at topic frequencies (§11.1). Run: python -m lcbank.codebook.build_pm
"""
import csv
import hashlib
import json
import re

from lcbank.codebook.common import header, write_codebook
from lcbank.common.paths import DATA_DERIVED, MANIFEST

STRANDS = {"1": "Statistics and Probability", "2": "Geometry and Trigonometry", "3": "Number", "4": "Algebra",
           "5": "Functions"}
NAMES = {"1.1": "Counting", "1.2": "Concepts of probability", "1.3": "Outcomes of random processes",
         "1.4": "Statistical reasoning", "1.5": "Finding, collecting and organising data",
         "1.6": "Representing data graphically and numerically",
         "1.7": "Analysing, interpreting and drawing inferences from data", "2.1": "Synthetic geometry",
         "2.2": "Co-ordinate geometry", "2.3": "Trigonometry", "2.4": "Transformation geometry, enlargements",
         "3.1": "Number systems", "3.2": "Indices", "3.3": "Arithmetic", "3.4": "Length, area and volume",
         "4.1": "Expressions", "4.2": "Solving equations", "4.3": "Inequalities", "4.4": "Complex numbers",
         "5.1": "Functions", "5.2": "Calculus"}

# topic_id, L1 strand, name, syllabus codes, bullet selector (None = all bullets of those codes), excludes, rules
TOPICS = [
    ("PM.PRB.01", "1", "Counting: arrangements and selections", ["1.1"], None, [], []),
    ("PM.PRB.02", "1", "Concepts of probability", ["1.2"], None, ["Bernoulli/binomial and normal probabilities (PM.PRB.03)"],
     ["Sample spaces, set notation, rules of probability, conditional probability and independence belong here."]),
    ("PM.PRB.03", "1", "Bernoulli trials, binomial and normal distribution probabilities", ["1.3"], None,
     ["Confidence intervals and hypothesis tests (PM.STA.03)"],
     ["Reading probabilities from normal tables and expected value of random processes belong here."]),
    ("PM.STA.01", "1", "Statistical reasoning and data collection", ["1.4", "1.5"], None, [],
     ["Types of data, sampling methods, bias and survey design belong here."]),
    ("PM.STA.02", "1", "Representing and describing data", ["1.6"], None, ["Inference from samples (PM.STA.03)"],
     ["Graphs, measures of centre and spread, percentiles, scatter plots and correlation belong here."]),
    ("PM.STA.03", "1", "Statistical inference", ["1.7"], None, [],
     ["Empirical rule, z-scores used for,
    ("PM.GEO.01", "2", "Synthetic geometry: theorems, proofs and constructions", ["2.1"], None, [], []),
    ("PM.GEO.02", "2", "Co-ordinate geometry of the line", ["2.2"], ("not", r"circle|\(x-h\)|x2\s*\+\s*y2"),
     ["The circle (PM.GEO.03)"], []),
    ("PM.GEO.03", "2", "Co-ordinate geometry of the circle", ["2.2"], ("only", r"circle|\(x-h\)|x2\s*\+\s*y2"), [],
     ["Equations of circles, tangents and line-circle intersection belong here."]),
    ("PM.GEO.04", "2", "Trigonometry of triangles, sectors and 3D problems", ["2.3"],
     ("not", r"for all values|define tan|graph|equations|radian|formulae"), ["Trigonometric functions and equations (PM.GEO.05)"],
     ["Pythagoras, sine and cosine rules, area of a triangle, sectors/arcs and 3D problems belong here."]),
    ("PM.GEO.05", "2", "Trigonometric functions, identities and equations", ["2.3"],
     ("only", r"for all values|define tan|graph|equations|radian|formulae"), [],
     ["Unit-circle definitions, radians, graphs of trigonometric functions, identities and solving trigonometric equations belong here."]),
    ("PM.GEO.06", "2", "Transformation geometry and enlargements", ["2.4"], None, [], []),
    ("PM.NUM.01", "3", "Number systems, irrational numbers and proof", ["3.1"],
     ("not", r"complex|argand|conjugate|sequence|series|pattern|limit|sum to"), ["Sequences and series (PM.NUM.02)"],
     ["Proof by induction belongs here even when the identity is about a series."]),
    ("PM.NUM.02", "3", "Sequences, series and limits", ["3.1"], ("only", r"sequence|series|pattern|limit|sum to"),
     ["Financial applications (PM.NUM.04)"], ["Arithmetic and geometric sequences/series, sums to infinity and limits of sequences belong here."]),
    ("PM.NUM.03", "3", "Indices and logarithms", ["3.2"], None, [], []),
    ("PM.NUM.04", "3", "Arithmetic: applied and financial mathematics", ["3.3"], None, [],
     ["Percentages, error, compound interest, depreciation, present value, annuities and amortisation belong here, including their geometric-series algebra."]),
    ("PM.NUM.05", "3", "Length, area and volume", ["3.4"], None, ["Area under a curve by integration (PM.FUN.04)"],
     ["Plane figures, solids, nets and the trapezoidal rule belong here."]),
    ("PM.ALG.01", "4", "Algebraic expressions and polynomials", ["4.1"], None, [], []),
    ("PM.ALG.02", "4", "Solving equations", ["4.2"], None, [], ["Linear, quadratic, cubic, simultaneous and surd/exponential equations belong here when solving is the assessed step."]),
    ("PM.ALG.03", "4", "Inequalities", ["4.3"], None, [], []),
    ("PM.ALG.04", "4", "Complex numbers", ["3.1", "4.4"], ("only_3.1", r"complex|argand|conjugate"), [],
     ["All complex-number work (Argand diagram, modulus, conjugates, polar form, De Moivre, complex roots) belongs here."]),
    ("PM.FUN.01", "5", "Functions and their graphs", ["5.1"], None, ["Calculus (PM.FUN.02-04)"],
     ["Composition, inverses, transformations of graphs, exponential/log functions as functions and limits of functions belong here."]),
    ("PM.FUN.02", "5", "Differentiation", ["5.2"], ("only", r"^(differentiate|find (the )?(first|derivatives)|find first|associate derivatives|use differentiation to find the slope)"),
     ["Applications of differentiation (PM.FUN.03)", "Integration (PM.FUN.04)"], ["Rules of differentiation and first principles belong here."]),
    ("PM.FUN.03", "5", "Applications of differentiation", ["5.2"], ("only", r"apply differentiation|apply the differentiation"),
     [], ["Rates of change, maxima and minima, curve sketching and optimisation problems belong here."]),
    ("PM.FUN.04", "5", "Integration", ["5.2"], ("only", r"integrat|average value|areas of plane regions"), [],
     ["Antiderivatives, definite integrals, area under/between curves and average value belong here."]),
]

GLOBAL_RULES = [
    "Label each scoring step by the mathematics it assesses; a real-world context does not decide the topic.",
    "When a step combines strands, use the topic of the concept the scheme rewards in that step (e.g. finding a circle's equation → PM.GEO.03 even if a quadratic is solved on the way).",
    "Calculus applied to any function → PM.FUN.02-04; sketching or interpreting graphs without calculus → PM.FUN.01.",
    "Geometric series in a financial context (loans, annuities, present value) → PM.NUM.04; abstract sequences/series → PM.NUM.02.",
    "Normal-table probabilities → PM.PRB.03; empirical rule, confidence intervals and hypothesis tests → PM.STA.03.",
    "Complex numbers anywhere → PM.ALG.04; proof by induction → PM.NUM.01.",
    "PMT (2012-2013) papers use this codebook; content outside the 2015 syllabus is labelled with a topic only if it clearly matches its includes, otherwise UNCLEAR.",
    "Use UNCLEAR only when a step cannot be assigned to a single topic with reasonable confidence.",
]


def select(bullets, sel, code):
    if sel is None:
        return bullets
    kind, pat = sel
    rx = re.compile(pat, re.I)
    if kind == "only":
        return [b for b in bullets if rx.search(b)]
    if kind == "not":
        return [b for b in bullets if not rx.search(b)]
    if kind == "only_3.1":
        return [b for b in bullets if code != "3.1" or rx.search(b)]
    raise ValueError(kind)


def build():
    path = DATA_DERIVED / "codebooks" / "sources" / "PM_syllabus_outline.json"
    outline = {t["code"]: t for t in json.loads(path.read_text())}
    assigned = {}
    topics = []
    for tid, strand, name, codes, sel, excludes, rules in TOPICS:
        includes, refs = [], []
        for code in codes:
            o = outline[code]
            bullets = [("OL", b) for b in o["ol"]] + [("HL", b) for b in o["hl"]]
            chosen = [(lvl, b) for lvl, b in bullets if b in select([x for _, x in bullets], sel, code)]
            for lvl, b in chosen:
                assigned.setdefault((code, lvl, b), []).append(tid)
            includes += [b[:110] for _, b in chosen]
            refs.append(f"{code} {NAMES[code]} (p{o['page']})")
        topics.append({"topic_id": tid, "L1": STRANDS[strand], "name": name, "syllabus_ref": "; ".join(refs),
                       "includes": includes[:12] or [NAMES[codes[0]]], "excludes": excludes, "boundary_rules": rules,
                       "examples": []})
    every = {(c, lvl, b) for c, o in outline.items() for lvl, bs in (("OL", o["ol"]), ("HL", o["hl"])) for b in bs}
    missing = every - set(assigned)
    doubled = {k: v for k, v in assigned.items() if len(v) > 1}
    if missing or doubled:
        raise ValueError(f"bullets not assigned exactly once: missing {sorted(missing)[:5]} doubled {list(doubled.items())[:5]}")
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        syl = next(r for r in csv.DictReader(f) if r["doc_id"] == "PM-syllabus-2015")
    cb = header("PM", "v1", [{"doc_id": syl["doc_id"], "url": syl["url"], "sha256": syl["sha256"],
                              "pages": "17-43 (Ordinary and Higher level strand tables)",
                              "outline_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "applies_to": ["PM 2014-2026", "PMT 2012-2013 (version differences not documented, OI-007)"]}],
                GLOBAL_RULES)
    cb["topics"] = topics
    return cb


if __name__ == "__main__":
    cb = build()
    out, sha = write_codebook(cb)
    print(out, len(cb["topics"]), "topics", "sha256", sha)
