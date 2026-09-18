"""Phase 3 (PM, PMT, AM2): leaves and marks from the marking scheme (§10.2).

Two scheme layouts:
- summary table ("Summary of mark allocations and scales to be applied"): two columns of
  "Question N" / "(a)(i)  10C" rows; one table per paper;
- inline scales (2021+): "Scale 10D (0, 3, 5, 7, 10)" in a Marking Notes column beside the model solution,
  following the part label it belongs to.
A leaf is one scale row: its marks are the sum of the numbers in the row ("5B, 5B, 5B" = 15).
"""
import re

from lcbank.structure.layout import n_pages, page_lines

ROMAN = r"(?:i{1,3}|iv|vi{0,3}|ix|x)"
LABEL_TOKEN = re.compile(r"\(([a-z]|" + ROMAN + r")\)")
SCALE_TOKEN = re.compile(r"(?<![\w(])(\d{1,2})\s*([A-E])?(\*?)(?![\w)])")


def _rows(lines, tol=4):
    rows = []
    for l in sorted(lines, key=lambda l: l["bbox"][1]):
        if rows and abs(l["bbox"][1] - rows[-1][0]["bbox"][1]) <= tol:
            rows[-1].append(l)
        else:
            rows.append([l])
    return [" ".join(x["text"] for x in sorted(r, key=lambda x: x["bbox"][0])) for r in rows]


def label_path(spec):
    """'(a)(i)+(ii)' -> ('a', 'i_ii'); '(b)' -> ('b',); '' -> ('all',)."""
    tokens = LABEL_TOKEN.findall(spec)
    if not tokens:
        return ("all",)
    path, romans = [], []
    for tok in tokens:
        if re.fullmatch(ROMAN, tok) and path:
            romans.append(tok)
        elif not path:
            path.append(tok)
        else:
            romans.append(tok)
    if romans:
        path.append(romans[0] if len(romans) == 1 else f"{romans[0]}_{romans[-1]}")
    return tuple(path)


def summary_pages(doc_id):
    pages = []
    for p in range(1, n_pages(doc_id) + 1):
        ls = page_lines(doc_id, p)
        head = [l for l in ls if re.search(r"summary of mark allocations", l["text"], re.I)]
        if head and any(re.match(r"^Question\s+\d+", l["text"]) for l in ls) and len(parse_summary_page(doc_id, p)) >= 5:
            pages.append(p)
    return pages


def parse_summary_page(doc_id, page):
    """Rows of a summary table. Columns are found from the x positions of the "Question N" headers."""
    ls = [l for l in page_lines(doc_id, page) if l["bbox"][1] < 780 and not re.fullmatch(r"\[\d+\]", l["text"].strip())]
    heads = sorted(l["bbox"][0] for l in ls if re.match(r"^Question\s+\d+", l["text"].strip()))
    cols = []
    for x in heads:
        if not cols or x - cols[-1] > 40:
            cols.append(x)
    if not cols:
        return []

    def col_of(l):
        x = l["bbox"][0]
        return max([i for i, c in enumerate(cols) if c <= x + 12] or [0])

    out = []
    for ci in range(len(cols)):
        q, alt = None, None
        for row in _rows([l for l in ls if col_of(l) == ci and l["bbox"][1] > min(
                (h["bbox"][1] for h in ls if re.search(r"summary of mark", h["text"], re.I)), default=0)]):
            t = row.strip()
            if re.match(r"^Section\s+[A-C]", t):
                continue
            m = re.match(r"^Question\s+(\d+)\s*([AB])?\b\s*(?:\((\d+)(?:\s*marks)?\))?\s*(.*)$", t)
            if m:
                q, alt = int(m.group(1)), m.group(2)
                if m.group(3):
                    out.append({"question": q, "alt": alt, "header_total": int(m.group(3))})
                t = m.group(4).strip()
                if not t:
                    continue
            if q is None:
                continue
            label_part = t[:max((mm.end() for mm in LABEL_TOKEN.finditer(t)), default=0)]
            rest = t[len(label_part):]
            scales = [(int(a), b, star) for a, b, star in SCALE_TOKEN.findall(rest) if int(a) <= 60]
            if not scales:
                continue
            out.append({"question": q, "alt": alt, "spec": t, "path": label_path(label_part),
                        "marks": sum(s[0] for s in scales),
                        "scale": ", ".join(f"{a}{b}{star}" for a, b, star in scales)})
    return out


def dedupe_paths(entries):
    seen = {}
    for e in entries:
        key = (e["question"], e["path"])
        n = seen.get(key, 0)
        seen[key] = n + 1
        if n:
            e["path"] = e["path"][:-1] + (f"{e['path'][-1]}.{n + 1}",)
    return entries


SCALE_LETTER = {2: "A", 3: "B", 4: "C", 5: "D", 6: "E"}


def inline_scale_leaves(doc_id):
    """Inline layout: bold "Qn" / "P1 Qn" + "Model Solution – N Marks", bold part labels in the left column, and
    "Scale NNX (...)" notes. Paper 2 starts where the question number drops back (e.g. Q10 -> Q1).
    "Scale (0, 4, 6, 10)" without a label is read as 10 marks, letter from the number of steps (inferred)."""
    out, q, path, paper, last_q, last, alt = [], None, ("all",), 1, 0, None, None
    for p in range(1, n_pages(doc_id) + 1):
        ls = page_lines(doc_id, p)
        for l in ls:
            t = l["text"].strip()
            x, y = l["bbox"][0], l["bbox"][1]
            m = re.match(r"^(?:P([12])\s*)?(Q)?\s*(\d{1,2})([AB])?\b", t)
            model_row = [o for o in ls if abs(o["bbox"][1] - y) <= 4 and re.search(r"Model Solution", o["text"])]
            if m and l["bold"] and x < 110 and (m.group(2) or (model_row and re.fullmatch(r"\d{1,2}", t))):
                n = int(m.group(3))
                if m.group(1):
                    paper = int(m.group(1))
                elif n < last_q:
                    paper += 1
                if n != q or m.group(4) != alt:
                    path = ("all",)
                q, last_q, alt = n, n, m.group(4)
                tot = re.search(r"(\d+)\s*Marks", model_row[0]["text"]) if model_row else None
                out.append({"paper": paper, "question": q, "alt": alt, "header_total": int(tot.group(1)) if tot else None})
                continue
            lm = re.match(r"^((?:\([a-z]\)|\(" + ROMAN + r"\))+)(?:\s*$|\s+(?!Scale)\S)", t)
            if lm and l["bold"] and x < 110 and q is not None:
                new_path = label_path(lm.group(1))
                if len(new_path) == 1 and re.fullmatch(ROMAN, new_path[0]) and path and path != ("all",):
                    new_path = (path[0], new_path[0])  # "(i)" under the current letter
                    if last and last["path"] == path[:1] and last["page"] == p and y - last["y"] < 30:
                        last["path"] = new_path  # "(a)" + scale, then "(i)" on the next line: the scale was (a)(i)
                path = new_path
                continue
            pre = re.match(r"^((?:\([a-z]\)|\(" + ROMAN + r"\))+)\s*(?=Scale\s*\d)", t)
            if pre and q is not None:  # "(d)(i) Scale 10D (...)": the label sits on the scale's line
                path = label_path(pre.group(1))
                t = t[pre.end():]
            sm = re.match(r"^Scale\s*(\d{1,2})\s*([A-E])", t)
            sn = None if sm else re.match(r"^Scale\s*\(([\d,\s]+)\)", t)
            if (sm or sn) and q is not None:
                if sm:
                    marks, scale = int(sm.group(1)), f"{sm.group(1)}{sm.group(2)}"
                else:
                    steps = [int(v) for v in re.findall(r"\d+", sn.group(1))]
                    marks, scale = max(steps), f"{max(steps)}{SCALE_LETTER.get(len(steps), '?')} (inferred)"
                last = {"paper": paper, "question": q, "alt": alt, "spec": t, "path": path, "marks": marks, "scale": scale,
                        "page": p, "y": y}
                out.append(last)
    return out


def old_format_leaves(doc_id):
    """Pre-Project-Maths layout (PMT 2012 Paper 1): "QUESTION N" then "Part (a)  10 (5, 5) marks" rows.
    Stops at the first "Paper 2" heading. The first occurrence of each part is used."""
    out, q, seen = [], None, set()
    for p in range(1, n_pages(doc_id) + 1):
        ls = page_lines(doc_id, p)
        if any(re.search(r"paper\s*2", l["text"], re.I) and l["bold"] and l["bbox"][1] < 200 for l in ls) and out:
            break
        for row in _rows(ls):
            m = re.match(r"^QUESTION\s+(\d+)", row.strip())
            if m:
                q = int(m.group(1))
                continue
            m = re.match(r"^Part\s+\(([a-z])\)\s+(\d{1,2})\s*(\([\d,\s]+\))?\s*marks", row.strip())
            if m and q is not None and (q, m.group(1)) not in seen:
                seen.add((q, m.group(1)))
                out.append({"paper": 1, "question": q, "alt": None, "spec": row.strip()[:60], "path": (m.group(1),),
                            "marks": int(m.group(2)), "scale": (m.group(3) or "").strip()})
    return out


def scheme_texts(doc_id):
    """Scheme text per (paper, question, alt, path) from the detailed solutions: text between a part label
    (or question header) and the next label/header. Paper 2 starts at "P2 Qn" or when question numbers reset."""
    texts, key, paper, last_q, q, alt = {}, None, 1, 0, None, None
    letter = None
    for p in range(1, n_pages(doc_id) + 1):
        ls = page_lines(doc_id, p)
        for l in ls:
            t, x, y = l["text"].strip(), l["bbox"][0], l["bbox"][1]
            model_row = any(abs(o["bbox"][1] - y) <= 4 and "Model Solution" in o["text"] for o in ls)
            m = re.match(r"^(?:P([12])\s*)?(Q|Question\s+)?\s*(\d{1,2})([AB])?\b", t)
            if m and l["bold"] and x < 110 and (m.group(2) or (model_row and re.fullmatch(r"\d{1,2}[AB]?", t))):
                n = int(m.group(3))
                if m.group(1):
                    paper = int(m.group(1))
                elif n < last_q:
                    paper += 1
                if n != q or m.group(4) != alt:
                    key, letter = (paper, n, m.group(4), "all"), None
                q, last_q, alt = n, n, m.group(4)
                continue
            lm = re.match(r"^((?:\([a-z]\)|\(" + ROMAN + r"\))+)", t)
            if lm and l["bold"] and x < 110 and q is not None:
                toks = LABEL_TOKEN.findall(lm.group(1))
                for tok in toks:
                    if re.fullmatch(ROMAN, tok) and letter is not None and not (tok == "i" and letter == "h"):
                        key = (paper, q, alt, f"{letter}-{tok}")
                    else:
                        letter = tok
                        key = (paper, q, alt, tok)
            if key is not None:
                texts.setdefault(key, []).append(t)
    return {k: "\n".join(v) for k, v in texts.items()}


def right_column_leaves(doc_id):
    """AM2 layout: bold labels "1(a) (i)", "2(b)" or "3" + "(i)" at the left; step marks "5" / "5, 5" in the right
    column. A leaf's marks are the sum of right-column marks until the next label."""
    out, cur, q = [], None, None
    for p in range(1, n_pages(doc_id) + 1):
        ls = page_lines(doc_id, p)
        rows = {}
        for l in ls:
            rows.setdefault(round(l["bbox"][1] / 5), []).append(l)
        for key in sorted(rows):
            row = sorted(rows[key], key=lambda l: l["bbox"][0])
            left = " ".join(l["text"].strip() for l in row if l["bold"] and l["bbox"][0] < 130)
            m = re.match(r"^(\d{1,2})\s*((?:\([a-z]+\)\s*)*)$", left)
            if m and (m.group(2) or len(row) == 1 or any(re.fullmatch(r"\(\w+\)", l["text"].strip()) for l in row)):
                n = int(m.group(1))
                if 1 <= n <= 12 and (q is None or n >= q):
                    q = n
                    toks = LABEL_TOKEN.findall(m.group(2))
                    letter = next((x for x in toks if not re.fullmatch(ROMAN, x)), None)
                    roman = next((x for x in toks if re.fullmatch(ROMAN, x)), None)
                    if letter is None and cur and cur["question"] == n and roman:
                        letter = cur["path"][0] if cur["path"][0] not in ("all",) and not re.fullmatch(ROMAN, cur["path"][0]) else None
                    path = tuple(x for x in (letter, roman) if x) or ("all",)
                    cur = {"paper": 1, "question": n, "alt": None, "spec": left, "path": path, "marks": 0,
                           "scale": "step marks", "text": []}
                    out.append(cur)
            if cur is not None and row[0]["bbox"][1] < 770:
                cur["text"].append(" ".join(l["text"].strip() for l in row))
            for l in row:
                tx = l["text"].strip()
                if cur and l["bbox"][0] > 480 and l["bbox"][1] < 770 and re.fullmatch(r"\d{1,2}(\s*,\s*\d{1,2})*", tx):
                    vals = [int(v) for v in re.findall(r"\d+", tx)]
                    if all(v <= 20 for v in vals):
                        cur["marks"] += sum(vals)
    for e in out:
        e["text"] = "\n".join(e["text"])[:1500]
    return out  # entries with 0 marks keep their text; build_maths drops them from the leaves
