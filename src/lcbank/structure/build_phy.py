"""Phase 3 (PHY): items_structured_PHY.jsonl, rules trees, leaf/stem crops, validation report.

Run: python -m lcbank.structure.build_phy
"""
import csv
import json
import re
from collections import Counter

import pymupdf
import yaml

from lcbank.common.paths import CONFIG, CROPS, DATA_DERIVED, MANIFEST, PAGES, REPORTS, TEXT
from lcbank.structure.phy_parser import build_rules, parse_paper
from lcbank.structure.rules import max_score, validate
from lcbank.structure.scheme_align import align, label_fallback, scheme_flat

DPI = 150
SCALE = DPI / 72.0
PAD = 4  # points
ODD_CHARS = re.compile(r"[-�]")


def _pages_method(doc_id):
    s = json.loads((TEXT / doc_id / "pages.json").read_text())
    return {p["page"]: p["extraction_method"] for p in s["pages"]}


def _band(lines, i, j, width):
    """Vertical bands (one per page) covering lines[i..j], full content width."""
    bands = {}
    for l in lines[i:j + 1]:
        b = bands.setdefault(l["page"], [l["bbox"][1], l["bbox"][3]])
        b[0], b[1] = min(b[0], l["bbox"][1]), max(b[1], l["bbox"][3])
    return [{"page": p, "bbox": [20.0, round(y0 - PAD, 1), round(width - 15.0, 1), round(y1 + PAD, 1)]}
            for p, (y0, y1) in sorted(bands.items())]


def _crop(doc_id, regions, out_base):
    paths = []
    for k, r in enumerate(regions, 1):
        png = PAGES / doc_id / f"p{r['page']}.png"
        pix = pymupdf.Pixmap(str(png))
        x0, y0, x1, y1 = [int(v * SCALE) for v in r["bbox"]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(pix.width, x1), min(pix.height, y1)
        if x1 - x0 < 10 or y1 - y0 < 5:
            continue
        doc = pymupdf.open()
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, pixmap=pix)
        clip = pymupdf.Rect(x0, y0, x1, y1)
        sub = page.get_pixmap(clip=clip, dpi=72)  # page units are already pixels
        out = out_base.parent / f"{out_base.name}_{k}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.save(out)
        doc.close()
        paths.append(str(out.relative_to(CROPS.parents[1])))
    return paths


def text_quality(text):
    if len(text.strip()) < 15:
        return "poor"
    return "poor" if len(ODD_CHARS.findall(text)) / max(1, len(text)) > 0.02 else "good"


def build(make_crops=True):
    overrides = yaml.safe_load((CONFIG / "mark_overrides.yaml").read_text()) or {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        docs = [r for r in csv.DictReader(f) if r["dataset"] == "PHY" and r["doc_type"] == "paper"]
    (DATA_DERIVED / "rules").mkdir(parents=True, exist_ok=True)
    (DATA_DERIVED / "items").mkdir(parents=True, exist_ok=True)
    out_items, report = [], []
    for d in sorted(docs, key=lambda r: (int(r["year"]), r["sitting_type"])):
        year, sitting = int(d["year"]), d["sitting_type"]
        prefix = f"PHY-{year}-P1" if sitting == "main" else f"PHY-{year}{sitting[0].upper()}-P1"
        info = parse_paper(d["doc_id"])
        root, items = build_rules(info, prefix)
        by_id = {i["item_id"]: i for i in items}
        applied = []
        for item_id, o in overrides.items():
            if item_id in by_id:
                by_id[item_id]["marks_printed"] = by_id[item_id]["marks"]
                by_id[item_id]["marks"] = o["marks"]
                by_id[item_id]["marks_source"] = "scheme_override"
                applied.append(item_id)
        marks = {i["item_id"]: i["marks"] for i in items}
        printed = sum((s["marks"] or 0) for s in info["sections"].values())
        problems = []
        try:
            validate(root)
            mx = max_score(root, marks)
        except ValueError as e:
            mx = None
            problems.append(f"rules invalid: {e}")
        if mx != printed:
            problems.append(f"tree max {mx} != printed total {printed}")
        for sec in root["children"]:
            k = sec.get("k") or len(sec["children"])
            per = (sec["printed_marks"] or 0) / k if k else None
            for qn in sec["children"]:
                v = max_score(qn, marks) if mx is not None else None
                if v != per:
                    problems.append(f"Q{qn['question']} max {v} != {per:g}")
        excluded = bool(problems)
        rules_doc = {"paper_id": prefix, "dataset": "PHY", "year": year, "sitting_type": sitting, "paper": 1,
                     "doc_id": d["doc_id"], "printed_total": printed, "max_score": mx, "valid": not problems,
                     "problems": problems, "root": root}
        (DATA_DERIVED / "rules" / f"{prefix}.json").write_text(json.dumps(rules_doc, indent=1, ensure_ascii=False))
        methods = _pages_method(d["doc_id"])
        lines = info["lines"]
        width = max((l["width"] for l in lines), default=595)
        qmap = {q.number: q for q in info["questions"]}
        for it in items:
            q = qmap[int(it["question"])]
            u = next(u for u in q.units if "-".join(u.path) == "-".join(it["part_path"]))
            stem_id = f"{prefix}-Q{q.number}-stem"
            stem_end = q.first_label - 1 if q.first_label > q.start else -1
            unit_lines = lines[u.start:u.end + 1]
            regions = _band(lines, u.start, u.end, width)
            text = "\n".join(l["text"] for l in unit_lines)
            crop_paths = _crop(d["doc_id"], regions, CROPS / "PHY" / it["item_id"]) if make_crops else []
            out_items.append({
                "item_id": it["item_id"], "dataset": "PHY", "year": year, "sitting_type": sitting, "paper": 1,
                "section": it["section"], "question": it["question"], "part_path": it["part_path"],
                "marks": it["marks"], "marks_source": it.get("marks_source", "paper_printed"),
                "marks_printed": it.get("marks_printed", it["marks"]), "mark_text": it["mark_text"],
                "stem_ref": stem_id if stem_end >= q.start else None,
                "source": {"doc_id": d["doc_id"], "url": d["url"], "page": regions[0]["page"] if regions else None,
                           "bbox": regions[0]["bbox"] if regions else None, "regions": regions},
                "text": text, "scheme_text": None, "scheme_steps": None, "crop_paths": crop_paths,
                "text_quality": text_quality(text),
                "extraction_method": "text" if all(methods.get(r["page"]) == "text" for r in regions) else "mixed",
                "topics": None, "label_
                "codebook_version": None, "analysis_excluded": excluded or sitting != "main",
            })
        # stems
        for q in info["questions"]:
            if q.first_label > q.start + 0:
                regions = _band(lines, q.start, q.first_label - 1, width)
                stem_id = f"{prefix}-Q{q.number}-stem"
                stem_text = "\n".join(l["text"] for l in lines[q.start:q.first_label])
                paths = _crop(d["doc_id"], regions, CROPS / "PHY" / stem_id) if make_crops else []
                (DATA_DERIVED / "items" / "stems_PHY.jsonl").open("a").write(json.dumps(
                    {"stem_id": stem_id, "doc_id": d["doc_id"], "text": stem_text, "regions": regions,
                     "crop_paths": paths}, ensure_ascii=False) + "\n")
        report.append({"paper_id": prefix, "doc_id": d["doc_id"], "questions": len(info["questions"]),
                       "leaves": len(items), "printed_total": printed, "max_score": mx, "valid": not problems,
                       "overrides": applied, "problems": problems})
        print(f"{prefix:<14} Q={len(info['questions']):<2} leaves={len(items):<3} max={mx} printed={printed} "
              f"{'OK' if not problems else 'INVALID ' + '; '.join(problems)[:80]}", flush=True)
    attach_scheme_text(out_items)
    path = DATA_DERIVED / "items" / "items_structured_PHY.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return out_items, report


def attach_scheme_text(items):
    """scheme_text per leaf: prompt anchors, then label/question-block fallback (scheme_align)."""
    by_doc = {}
    for it in items:
        by_doc.setdefault(it["source"]["doc_id"], []).append(it)
    for doc_id, its in by_doc.items():
        scheme_id = doc_id.replace("-paper-P1", "-scheme")
        if not (TEXT / scheme_id / "pages.json").exists():
            for it in its:
                it["scheme_text"], it["scheme_align"] = None, "no_scheme"
            continue
        res, _ = align(its, scheme_id)
        res = label_fallback(its, scheme_flat(scheme_id)[0], res)
        for it, (excerpt, in zip(its, res):
            it["scheme_text"], it["scheme_align"] = excerpt,


def write_report(items, report):
    tq = Counter(i["text_quality"] for i in items)
    lines = ["# Structure report — PHY (Phase 3)", "",
             "Generated by `python -m lcbank.structure.build_phy`. Leaves are mark units printed on the paper "
             "; overrides from `config/mark_overrides.yaml` (D-021).", "",
             f"- Papers: {len(report)}; valid rules trees: {sum(r['valid'] for r in report)}",
             f"- Leaves: {len(items)}; with marks: {sum(1 for i in items if i['marks'])} "
             f"({sum(1 for i in items if i['marks']) / max(1, len(items)):.1%})",
             f"- Text quality: {dict(tq)}",
             f"- Scheme excerpt alignment: {dict(Counter(i.get('scheme_align') for i in items))}", "",
             "| Paper | Questions | Leaves | Printed total | Tree max | Valid | Overrides | Problems |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report:
        lines.append(f"| {r['paper_id']} | {r['questions']} | {r['leaves']} | {r['printed_total']} | {r['max_score']} | "
                     f"{'yes' if r['valid'] else 'NO'} | {', '.join(r['overrides']) or ''} | {'; '.join(r['problems'])} |")
    (REPORTS / "structure_PHY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    stems = DATA_DERIVED / "items" / "stems_PHY.jsonl"
    stems.unlink(missing_ok=True)
    items, report = build()
    write_report(items, report)
