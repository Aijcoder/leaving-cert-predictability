"""Phase 3 (PM, PMT): items_structured_{PM,PMT}.jsonl, rules trees per paper and per sitting, crops, report.

Run: python -m lcbank.structure.build_maths
"""
import csv
import json
import re
from collections import Counter, defaultdict

from lcbank.common.paths import CROPS, DATA_DERIVED, MANIFEST, REPORTS, TEXT
from lcbank.structure import maths_paper as mp
from lcbank.structure import maths_scheme as ms
from lcbank.structure.build_phy import _band, _crop, _pages_method, text_quality
from lcbank.structure.rules import max_score, validate

SITTING_CODE = {"main": "", "deferred": "D", "project_maths_variant": "V"}


def scheme_excerpt(texts, paper, q, alt, path):
    if path and path[0] != "all":
        letter = path[0]
        if len(path) > 1:
            first_roman = re.split(r"[._]", path[1])[0]
            hit = texts.get((paper, q, alt, f"{letter}-{first_roman}"))
            if hit:
                return hit[:1500], "label"
        hit = texts.get((paper, q, alt, letter))
        if hit:
            return hit[:1500], "label" if len(path) == 1 else "parent_label"
    whole = "\n".join(v for k, v in texts.items() if k[:3] == (paper, q, alt))
    return (whole[:1500], "question") if whole else (None, "not_found")


def build(dataset, make_crops=True):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        docs = [r for r in csv.DictReader(f) if r["dataset"] == dataset and r["doc_type"] == "paper"]
    sittings = defaultdict(list)
    for d in docs:
        sittings[(int(d["year"]), d["sitting_type"])].append(d)
    out_items, report = [], []
    for (year, sitting), papers in sorted(sittings.items()):
        code = SITTING_CODE.get(sitting, sitting[:1].upper())
        scheme_doc = f"{dataset}-{year}-{sitting}-scheme"
        cands = mp.scheme_candidates(scheme_doc)
        texts = ms.scheme_texts(scheme_doc)
        if dataset == "AM2":  # AM2 schemes: text collected per label by the right-column reader
            texts = {}
            for e in cands.get("right_column", {}).get(1, []):
                key = (1, e["question"], None, "-".join(e["path"]) if e["path"] != ("all",) else "all")
                texts[key] = (texts.get(key, "") + "\n" + e["text"]).strip()
                if len(e["path"]) > 1:  # also index the letter, so "(a)" can find its sub-parts' text
                    texts[(1, e["question"], None, e["path"][0])] = (texts.get((1, e["question"], None, e["path"][0]), "")
                                                                   + "\n" + e["text"]).strip()
            if "right_column" in cands:
                cands["right_column"][1] = [e for e in cands["right_column"][1] if e["marks"] > 0]
        paper_nodes, sitting_ok = [], True
        for d in sorted(papers, key=lambda r: int(r["paper_no"])):
            k = int(d["paper_no"])
            prefix = f"{dataset}-{year}{code}-P{k}"
            info = mp.parse_paper(d["doc_id"])
            src, entries, bad = mp.choose_leaves(info, cands, k) or (None, [], [])
            # questions whose scheme leaves do not add up to the printed question marks become one leaf with the
            # printed marks (marks stay official; granularity is coarser) —
            fallback = sorted({b[0] for b in mp.mismatches(info, entries)
                               if isinstance(b[0], int) and b[0] in info["questions"] and info["questions"][b[0]]["marks"]})
            if fallback:
                entries = [e for e in entries if e["question"] not in fallback] + [
                    {"paper": k, "question": n, "alt": None, "spec": "printed question total", "path": ("all",),
                     "marks": info["questions"][n]["marks"], "scale": "question_total_fallback"} for n in fallback]
                bad = mp.mismatches(info, entries)
            root, items = mp.build_rules(info, entries, prefix)
            marks = {i["item_id"]: i["marks"] for i in items}
            printed = info["paper_total"] or sum((s["marks"] or 0) for s in info["sections"])
            problems = [f"Q{b[0]} scheme {b[1]} != printed {b[2]}" for b in bad]
            try:
                validate(root)
                mx = max_score(root, marks)
            except ValueError as e:
                mx = None
                problems.append(f"rules invalid: {e}")
            if mx != printed:
                problems.append(f"tree max {mx} != printed total {printed}")
            sitting_ok &= not problems
            rules_doc = {"paper_id": prefix, "dataset": dataset, "year": year, "sitting_type": sitting, "paper": k,
                         "doc_id": d["doc_id"], "scheme_doc": scheme_doc, "marks_source": src,
                         "printed_total": printed, "max_score": mx, "valid": not problems, "problems": problems,
                         "root": root}
            (DATA_DERIVED / "rules" / f"{prefix}.json").write_text(json.dumps(rules_doc, indent=1, ensure_ascii=False))
            paper_nodes.append(root)
            lines, methods = info["lines"], _pages_method(d["doc_id"])
            width = max((l["bbox"][2] for l in lines), default=560) + 20
            by_q = defaultdict(list)
            for it in items:
                by_q[(int(it["question"]), it["alt"])].append(it)
            for (qn, alt), its in by_q.items():
                q = info["questions"][qn]
                span_start = q["start"]
                span_end = q["end"]
                if alt:
                    a = next((x for x in q["alts"] if x["alt"] == alt), None)
                    if a:
                        span_start, span_end = a["start"], a["end"]
                starts = [mp.label_index(q, alt, tuple(it["scheme_path"])) for it in its]
                first = min([s_ for s_ in starts if s_ is not None], default=None)
                stem_text = "\n".join(l["text"] for l in lines[span_start:first]) if first and first > span_start else None
                for n_it, it in enumerate(its):
                    s = starts[n_it]
                    scope = "leaf"
                    if s is None:
                        s, e, scope = span_start, span_end, "question"
                    else:
                        later = [x for x in starts[n_it + 1:] if x is not None and x > s]
                        e = (min(later) - 1) if later else span_end
                    regions = _band(lines, s, e, width)
                    text = "\n".join(l["text"] for l in lines[s:e + 1])
                    excerpt, how = scheme_excerpt(texts, k, qn, alt, tuple(it["scheme_path"]))
                    crops = _crop(d["doc_id"], regions, CROPS / dataset / it["item_id"]) if make_crops else []
                    out_items.append({
                        "item_id": it["item_id"], "dataset": dataset, "year": year, "sitting_type": sitting,
                        "paper": k, "section": it["section"], "question": it["question"], "part_path": it["part_path"],
                        "marks": it["marks"],
                        "marks_source": "paper_question_total" if it["scale"] == "question_total_fallback" else f"scheme_{src}",
                        "scale": it["scale"],
                        "stem_ref": f"{prefix}-Q{qn}{alt or ''}-stem" if stem_text else None, "stem_text": stem_text,
                        "source": {"doc_id": d["doc_id"], "url": d["url"],
                                   "page": regions[0]["page"] if regions else None,
                                   "bbox": regions[0]["bbox"] if regions else None, "regions": regions},
                        "crop_scope": scope, "text": text, "scheme_text": excerpt, "scheme_align": how,
                        "scheme_steps": None, "crop_paths": crops, "text_quality": text_quality(text),
                        "extraction_method": "text" if all(methods.get(r["page"]) == "text" for r in regions) else "mixed",
                        "topics": None, "label_
                        "prompt_sha256": None, "codebook_version": None,
                        "analysis_excluded": bool(problems) or sitting != "main" or dataset in ("PMT", "AM2")})
            report.append({"paper_id": prefix, "marks_source": src, "fallback_questions": fallback,
                           "questions": len(info["questions"]),
                           "leaves": len(items), "printed_total": printed, "max_score": mx, "valid": not problems,
                           "problems": problems})
            print(f"{prefix:<14} src={str(src):<16} Q={len(info['questions']):<2} leaves={len(items):<3} max={mx} "
                  f"printed={printed} {'OK' if not problems else 'INVALID ' + '; '.join(problems)[:80]}", flush=True)
        sitting_id = f"{dataset}-{year}{code}"
        sitting_root = {"type": "all", "children": paper_nodes, "source_text": "Paper 1 and Paper 2 form one sitting"}
        (DATA_DERIVED / "rules" / f"{sitting_id}.json").write_text(json.dumps(
            {"sitting_id": sitting_id, "dataset": dataset, "year": year, "sitting_type": sitting,
             "valid": sitting_ok, "root": sitting_root}, indent=1, ensure_ascii=False))
    with open(DATA_DERIVED / "items" / f"items_structured_{dataset}.jsonl", "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return out_items, report


def write_report(dataset, items, report):
    lines = [f"# Structure report — {dataset} (Phase 3)", "",
             "Generated by `python -m lcbank.structure.build_maths`. Leaves and marks from the marking scheme "
             "(summary table, inline scales or old-format part rows), chosen per paper by agreement with the printed "
             "question marks.", "",
             f"- Papers: {len(report)}; valid: {sum(r['valid'] for r in report)}",
             f"- Leaves: {len(items)}; with marks: {sum(1 for i in items if i['marks'])} "
             f"({sum(1 for i in items if i['marks']) / max(1, len(items)):.1%})",
             f"- Text quality: {dict(Counter(i['text_quality'] for i in items))}",
             f"- Crop scope: {dict(Counter(i['crop_scope'] for i in items))}",
             f"- Scheme excerpt: {dict(Counter(i['scheme_align'] for i in items))}", "",
             "| Paper | Marks source | Question-level fallback | Questions | Leaves | Printed total | Tree max | Valid | Problems |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in report:
        lines.append(f"| {r['paper_id']} | {r['marks_source']} | {', '.join(map(str, r['fallback_questions']))} | "
                     f"{r['questions']} | {r['leaves']} | {r['printed_total']} | "
                     f"{r['max_score']} | {'yes' if r['valid'] else 'NO'} | {'; '.join(r['problems'])} |")
    (REPORTS / f"structure_{dataset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    for ds in (sys.argv[1:] or ["PM", "PMT", "AM2"]):
        items, report = build(ds)
        write_report(ds, items, report)
