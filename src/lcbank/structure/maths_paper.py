"""Phase 3 (PM, PMT): parse a Maths paper (sections, rubrics, question headers + printed marks, 6A/6B
alternatives, part labels) and combine it with scheme leaves into items and a rules tree.

Leaves and marks come from the scheme (maths_scheme): the summary table, the inline "Scale" notes, or the old
"Part (a) N marks" rows, whichever agrees with the paper's printed question marks.
"""
import re
from collections import defaultdict

from lcbank.structure import maths_scheme as ms
from lcbank.structure.layout import n_pages, page_lines

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "all": None}
ROMAN = r"(?:i{1,3}|iv|vi{0,3}|ix|x)"


def read_lines(doc_id):
    out = []
    for p in range(1, n_pages(doc_id) + 1):
        for l in page_lines(doc_id, p):
            if l["bbox"][1] > 790 or re.fullmatch(r"(Page \d+ of \d+|\d{1,2}|\[?Turn over\]?)", l["text"].strip(), re.I) \
                    and (l["bbox"][1] > 760 or l["bbox"][1] < 45):
                continue
            out.append(l)
    return out


def parse_paper(doc_id):
    lines = read_lines(doc_id)
    info = {"doc_id": doc_id, "lines": lines, "sections": [], "questions": {}, "paper_total": None}
    sec, q, alt, expected_old = None, None, None, 1
    old_style = False
    for idx, l in enumerate(lines):
        t, x, y = l["text"].strip(), l["bbox"][0], l["bbox"][1]
        row = [o for o in lines if o["page"] == l["page"] and abs(o["bbox"][1] - y) <= 4]
        m = re.search(r"\(\s*(\d{3})\s*marks\s*\)|^(\d{3})\s*marks$", t)
        if m and info["paper_total"] is None and l["page"] <= 2:
            info["paper_total"] = int(m.group(1) or m.group(2))
        m = re.search(r"Attempt\s+(\w+)\s+QUESTIONS\s*\((\d+)\s*marks each\)", t, re.I)
        if m:
            old_style = True
            sec = {"name": "A", "marks": None, "k": WORDS.get(m.group(1).lower()), "rubric": t, "questions": [],
                   "each": int(m.group(2))}
            info["sections"].append(sec)
            continue
        m = re.search(r"Each question carries (\d+) marks", t, re.I)
        if m:
            info["each"] = int(m.group(1))
        m = re.match(r"^Answer any (\w+) questions\.?$", t, re.I)
        if m and sec is None:
            sec = {"name": "A", "marks": None, "k": WORDS.get(m.group(1).lower()), "rubric": t, "questions": [],
                   "each": info.get("each")}
            info["sections"].append(sec)
            continue
        m = re.match(r"^Section\s+([A-C])\b(?:\s*\((\d+)\))?", t)
        if m and l["bold"] and x < 200:
            marks = int(m.group(2)) if m.group(2) else None
            for o in row:
                mm = re.fullmatch(r"(\d+)\s*marks", o["text"].strip())
                if mm:
                    marks = int(mm.group(1))
            sec = {"name": m.group(1), "marks": marks, "k": None, "rubric": None, "questions": []}
            info["sections"].append(sec)
            continue
        m = re.match(r"^Answer\s+(?:(all|any)\s+)?(\w+)\s+questions?\s+from this section", t, re.I)
        if m and sec is not None:
            sec["k"] = None if (m.group(1) or "").lower() == "all" else WORDS.get(m.group(2).lower())
            sec["rubric"] = t
            continue
        m = re.match(r"^Question\s+(\d+)\s*([AB])?$", t)
        mo = re.match(r"^(\d{1,2})\s*\.$", t) if old_style else None
        if (m and l["bold"] and x < 100) or (mo and l["bold"] and x < 80 and int(mo.group(1)) == expected_old):
            n = int((m or mo).group(1))
            letter = m.group(2) if m else None
            if letter and q is not None and q["number"] == n:
                if q["alts"]:
                    q["alts"][-1]["end"] = idx - 1
                q["alts"].append({"alt": letter, "start": idx, "end": idx})
                alt = letter
                continue
            if q is not None:
                q["end"] = idx - 1
                if q["alts"]:
                    q["alts"][-1]["end"] = idx - 1
            printed = None
            for o in row:
                mm = re.fullmatch(r"\((\d+)\s*marks\)", o["text"].strip(), re.I)
                if mm:
                    printed = int(mm.group(1))
            if printed is None and sec is not None and sec.get("each"):
                printed = sec["each"]  # "Attempt SIX QUESTIONS (50 marks each)"
            q = {"number": n, "section": sec["name"] if sec else None, "marks": printed, "start": idx, "end": idx,
                 "alts": [], "labels": [], "rubric": None}
            if old_style:
                expected_old += 1
            info["questions"][n] = q
            if sec is not None:
                sec["questions"].append(n)
            alt = None
            continue
        if q is None:
            continue
        q["end"] = idx
        if q["alts"]:
            q["alts"][-1]["end"] = idx
        if re.match(r"^Answer either\s+\d+A\s+or\s+\d+B", t, re.I):
            q["rubric"] = t
            continue
        lm = re.match(r"^((?:\([a-z]\)|\(" + ROMAN + r"\))+)", t)
        if lm and l["bold"] and x < 140:
            tokens = re.findall(r"\(([a-z]+)\)", lm.group(1))
            q["labels"].append({"tokens": tokens, "idx": idx, "alt": alt})
    if q is not None and q["alts"]:
        q["alts"][-1]["end"] = q["end"]
    return info


def label_index(q, alt, path):
    """Line index of the first label line matching path (letter, then roman) inside the question/alt."""
    letter = path[0] if path and path[0] != "all" else None
    roman = re.split(r"[._]", path[1])[0] if len(path) > 1 else None
    cur_letter, found_letter = None, None
    for lab in q["labels"]:
        if lab["alt"] != alt:
            continue
        for tok in lab["tokens"]:
            if re.fullmatch(ROMAN, tok) and cur_letter is not None and not (tok == "i" and cur_letter == "h"):
                if cur_letter == letter and roman == tok:
                    return lab["idx"]
            else:
                cur_letter = tok
                if tok == letter and roman is None:
                    return lab["idx"]
                if tok == letter:
                    found_letter = lab["idx"]
    return found_letter


def scheme_candidates(scheme_doc):
    """{source: {paper: [leaf entries]}} for each scheme layout that yields leaves."""
    cands = {}
    pages = ms.summary_pages(scheme_doc)
    if pages:
        cands["summary"] = {k + 1: [e for e in ms.parse_summary_page(scheme_doc, p) if "marks" in e]
                            for k, p in enumerate(pages[:2])}
    if scheme_doc.startswith("AM2"):
        return {"right_column": {1: ms.right_column_leaves(scheme_doc)}}
    inline = [e for e in ms.inline_scale_leaves(scheme_doc) if "marks" in e]
    if inline:
        cands["inline"] = {k: [e for e in inline if e["paper"] == k] for k in (1, 2)}
    old = ms.old_format_leaves(scheme_doc)
    if old:
        cands["old_format"] = {1: old}
    return cands


def question_sums(entries):
    sums = defaultdict(int)
    for e in entries:
        sums[(e["question"], e.get("alt"))] += e["marks"]
    return sums


def mismatches(info, entries):
    """Questions whose scheme leaves disagree with the paper's printed question marks."""
    sums = question_sums(entries)
    bad = []
    for n, q in info["questions"].items():
        if q["marks"] is None:
            continue
        keys = [k for k in sums if k[0] == n]
        alts = {k[1] for k in keys}
        if not keys:
            bad.append((n, None, q["marks"]))
        elif alts == {None}:
            if sums[(n, None)] != q["marks"]:
                bad.append((n, sums[(n, None)], q["marks"]))
        else:
            for k in keys:
                if sums[k] != q["marks"]:
                    bad.append((n, sums[k], q["marks"]))
    extra = {k[0] for k in sums} - set(info["questions"])
    bad += [(n, "not on paper", None) for n in sorted(extra)]
    return bad


def choose_leaves(info, cands, paper_no):
    """Pick the scheme source whose leaves agree with the paper's printed question marks; ties -> summary."""
    best = None
    for src in ("summary", "inline", "old_format", "right_column"):
        entries = cands.get(src, {}).get(paper_no)
        if not entries:
            # a summary page might belong to the other paper: try both pages
            continue
        bad = mismatches(info, entries)
        if best is None or len(bad) < len(best[2]):
            best = (src, entries, bad)
    for src in ("summary",):  # summary pages assigned by order may be swapped (e.g. only a P2 table exists)
        for k, entries in cands.get(src, {}).items():
            if k != paper_no and entries:
                bad = mismatches(info, entries)
                if best is None or len(bad) < len(best[2]):
                    best = (f"{src}_page{k}", entries, bad)
    return best


def build_rules(info, entries, item_prefix):
    items, sec_nodes = [], []
    by_q = defaultdict(list)
    for e in ms.dedupe_paths([dict(e) for e in entries]):
        by_q[e["question"]].append(e)
    for sec in info["sections"]:
        q_nodes = []
        for n in sec["questions"]:
            q = info["questions"][n]
            leaves_by_alt = defaultdict(list)
            for e in by_q.get(n, []):
                alt = e.get("alt")
                path = ((alt,) if alt else ()) + tuple(e["path"])
                item_id = f"{item_prefix}-Q{n}-" + "-".join(path)
                items.append({"item_id": item_id, "question": str(n), "section": sec["name"], "alt": alt,
                              "part_path": list(path), "scheme_path": list(e["path"]), "marks": e["marks"],
                              "scale": e.get("scale"), "scheme_spec": e.get("spec")})
                leaves_by_alt[alt].append({"type": "part", "item_id": item_id, "children": []})
            if set(leaves_by_alt) - {None}:
                node = {"type": "best_k", "k": 1, "source_text": q["rubric"] or "Answer either A or B",
                        "children": [{"type": "all", "children": v, "source_text": None}
                                     for a, v in sorted(leaves_by_alt.items(), key=lambda kv: str(kv[0])) if v]}
            else:
                node = {"type": "all", "children": leaves_by_alt[None], "source_text": None}
            node["question"] = n
            node["printed_marks"] = q["marks"]
            q_nodes.append(node)
        k = sec["k"]
        sec_nodes.append({"type": "best_k" if k else "all", "k": k, "children": q_nodes, "source_text": sec["rubric"],
                          "section": sec["name"], "printed_marks": sec["marks"]})
    return {"type": "all", "children": sec_nodes, "source_text": None}, items
