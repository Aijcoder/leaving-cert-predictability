"""Phase 3 (PHY): parse a Physics paper into questions, mark units (leaves) and a rules tree.

Design:
- Questions: lines starting "N." (or a bare bold "N") at the left margin with N the next expected number;
  sections from "SECTION A (120 marks)".
- Labels: (a)-(j), (i)-(x) and "Part (a)" at the start of a line; nesting level = rank of the label's indentation
  among the question's label indentations ("Part (x)" labels sit one level below the question's top labels).
- Leaves are *mark units*: every mark printed in the right margin, e.g. "(7)", "(4 + 3)", "(8 × 7)", closes a unit
  covering the labels opened since the previous mark. "(n × k)" over >= n labels gives k marks per label.
- Choices come from the paper's own rubric sentences, applied at the label level where they appear, and are
  quoted in each rules node's source_text.
"""
import re
from dataclasses import dataclass, field

from lcbank.structure.layout import n_pages, page_lines

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
MARK_RE = re.compile(r"^\(\s*(\d+)\s*(?:([×xX*])\s*(\d+)|((?:\s*\+\s*\d+)+))?\s*\)$")
TRAIL_MARK_RE = re.compile(r"(?:\s|(?<=[.?:;,]))\(\s*(\d+)\s*(?:([×xX*])\s*(\d+))?\s*\)\s*$")
LABEL_RE = re.compile(r"^(Part\s+)?\(([a-j]|i{1,3}|iv|vi{0,3}|ix|x)\)\s*")
QNUM_RE = re.compile(r"^(\d{1,2})(?:\s*\.(?:\s+(.*))?)?$")
HEADER_FOOTER_RE = re.compile(r"(page \d+ of \d+|\d{1,2}|\d{4}\.\s*M\d+[A-Z]?|Leaving Certificate.*|\[?Turn over\]?"
                              r"|.*Examination Number.*)", re.I)


@dataclass
class Unit:
    q: int
    labels: list          # label paths (tuples) opened since the previous mark
    marks: int
    mark_text: str
    lines: list = field(default_factory=list)   # (page, bbox) of text lines belonging to the unit
    path: tuple = ()
    start: int = 0        # index in read_lines() of the first line of the unit
    end: int = 0          # index of the mark line that closes the unit


@dataclass
class Question:
    number: int
    section: str
    page: int
    y: float
    units: list = field(default_factory=list)
    rubric: list = field(default_factory=list)    # dicts: kind, k, text, context (path tuple), labels
    stem_lines: list = field(default_factory=list)
    or_points: list = field(default_factory=list)  # unit indexes i where units[i-1] "or" units[i]
    start: int = 0        # index of the question-number line
    first_label: int = -1  # index of the first label line (end of the stem), -1 if none
    end: int = 0          # index of the last line before the next question/section


def _mark_value(m):
    base = int(m.group(1))
    if m.group(2):
        return base, int(m.group(3))            # (n × k)
    if m.lastindex and m.lastindex >= 4 and m.group(4):
        return base + sum(int(v) for v in re.findall(r"\d+", m.group(4))), None
    return base, None


def read_lines(doc_id):
    lines = []
    for p in range(1, n_pages(doc_id) + 1):
        ls = page_lines(doc_id, p)
        width = max((l["bbox"][2] for l in ls), default=595)
        for l in ls:
            if HEADER_FOOTER_RE.fullmatch(l["text"]) and (l["bbox"][1] < 50 or l["bbox"][1] > 760):
                continue
            l["width"] = max(width, 560)
            l["rightmost"] = not any(o is not l and abs(o["bbox"][1] - l["bbox"][1]) <= 3 and o["bbox"][0] >= l["bbox"][2]
                                     for o in ls)
            lines.append(l)
    return lines


def _question_start(l, expected):
    t, x = l["text"], l["bbox"][0]
    mq = QNUM_RE.match(t)
    if mq and int(mq.group(1)) == expected and x < 90 and (l["bold"] or re.fullmatch(r"\d{1,2}\s*\.", t)):
        return mq
    return None


def _labels_in(t, x):
    """Labels at the start of a line: [(label, is_part_label, x_estimate)], plus the rest of the text."""
    out, rest = [], t
    while True:
        ml = LABEL_RE.match(rest)
        if not ml:
            return out, rest
        out.append((ml.group(2), bool(ml.group(1)), round(x)))
        rest = rest[ml.end():]
        x = x + 28


def _label_levels(lines):
    """Pre-pass: per question number, sorted indentation clusters of its labels and whether it has Part labels."""
    levels, expected, qn = {}, 1, None
    for l in lines:
        mq = _question_start(l, expected)
        t, x = l["text"], l["bbox"][0]
        if mq:
            qn, expected = expected, expected + 1
            levels[qn] = {"xs": [], "part_labels": False}
            t, x = (mq.group(2) or "").strip(), x + 25
        if qn is None:
            continue
        for lab, is_part, lx in _labels_in(t, x)[0]:
            if is_part:
                levels[qn]["part_labels"] = True
            elif not any(abs(c - lx) <= 10 for c in levels[qn]["xs"]):
                levels[qn]["xs"].append(lx)
    for v in levels.values():
        v["xs"].sort()
    return levels


def parse_paper(doc_id):
    lines = read_lines(doc_id)
    info = {"doc_id": doc_id, "sections": {}, "paper_rubric": None}
    questions, q, section = [], None, None
    expected = 1
    state = {"pending": [], "path": [], "or_pending": False, "last_at_level": {}, "kind_at_level": {}, "opened_at": {}}
    label_levels = _label_levels(lines)

    def level_for(lx, is_part):
        lv_info = label_levels[q.number]
        if is_part:
            return 2
        rank = min(range(len(lv_info["xs"])), key=lambda i: abs(lv_info["xs"][i] - lx)) + 1
        return rank + 1 if lv_info["part_labels"] and rank >= 2 else rank

    unit_start = 0
    for idx, l in enumerate(lines):
        t, x, page, y = l["text"], l["bbox"][0], l["page"], l["bbox"][1]
        if q is not None:
            q.end = idx
        m = re.match(r"^SECTION\s+([A-Z])\b(?:.*?\((\d+)\s*marks\))?", t, re.I)
        if m and l["bold"]:
            section = m.group(1).upper()
            if q is not None:
                q.end = idx - 1
            s = info["sections"].setdefault(section, {"marks": None, "answer": None, "questions": []})
            if m.group(2):
                s["marks"] = int(m.group(2))
            continue
        if re.search(r"Answer (\w+) questions from Section A and (\w+) questions from Section B", t, re.I):
            info["paper_rubric"] = info["paper_rubric"] or t
            continue
        m = re.match(r"^Answer (\w+) questions? from this section", t, re.I)
        if m and section:
            info["sections"][section]["answer"] = (WORDS.get(m.group(1).lower()), t)
            continue
        mq = _question_start(l, expected)
        if mq:
            if q is not None:
                q.end = idx - 1
            q = Question(number=expected, section=section, page=page, y=y, start=idx, end=idx)
            unit_start = idx
            questions.append(q)
            if section:
                info["sections"][section]["questions"].append(expected)
            expected += 1
            state.update(pending=[], path=[], or_pending=False, last_at_level={}, kind_at_level={}, opened_at={})
            t = (mq.group(2) or "").strip()
            if not t:
                continue
            x = x + 25
        if q is None:
            continue
        # rubric inside a question
        mr = re.search(r"Answer any (\w+) of the following parts", t, re.I)
        if mr:
            q.rubric.append({"kind": "best_k", "k": WORDS.get(mr.group(1).lower()), "text": t,
                             "context": tuple(state["path"][:-1]) if state["pending"] else tuple(state["path"]),
                             "labels": None})
        me = re.search(r"Answer (?:either )?part \((\w+)\) or part \((\w+)\)", t, re.I)
        if me:
            q.rubric.append({"kind": "either", "k": 1, "text": t, "context": tuple(state["path"]),
                             "labels": (me.group(1), me.group(2))})
        if re.fullmatch(r"or|OR", t):
            state["or_pending"] = True
            continue
        # labels at the start of the line (possibly several, e.g. "(d) (i) ...")
        found_labels = _labels_in(t, x)[0]
        if found_labels and q.first_label < 0:
            q.first_label = idx
        for lab, is_part, lx in found_labels:
            lv = level_for(lx, is_part)
            prev = state["kind_at_level"].get(lv)
            last = state["last_at_level"].get((tuple(state["path"][:lv - 1]), lv))
            kind = ("letter" if last == "h" else "roman") if lab == "i" else \
                ("roman" if re.fullmatch(r"i{2,3}|iv|vi{0,3}|ix|x", lab) else "letter")
            if prev and kind != prev and len(state["path"]) >= lv and \
                    (lab == ("i" if kind == "roman" else "a") or state["kind_at_level"].get(lv + 1) == kind):
                lv += 1  # e.g. "(a)" then "(i)", "(ii)" at the same indentation: they are children of (a)
            state["kind_at_level"].setdefault(lv, kind)
            parent = tuple(state["path"][:lv - 1])
            if state["or_pending"] and state["last_at_level"].get((parent, lv)):
                # "or" between two sibling labels: the whole labels are alternatives
                q.rubric.append({"kind": "either", "k": 1, "text": "or", "context": parent,
                                 "labels": (state["last_at_level"][(parent, lv)], lab)})
                state["or_pending"] = False
            state["path"] = list(parent) + [lab]
            state["last_at_level"][(parent, lv)] = lab
            state["pending"].append(tuple(state["path"]))
            state["opened_at"].setdefault(tuple(state["path"]), idx)
        # marks: a whole margin line, or a trailing "(n)" on a line reaching the right margin
        # figure labels may sit to the right of a mark, so a mark-only line needs only to be right of centre-left
        mm = MARK_RE.match(t) if l["bbox"][0] > 0.3 * l["width"] else None
        mt = None if mm else (TRAIL_MARK_RE.search(t) if l["bbox"][2] > 0.8 * l["width"] or l["rightmost"] else None)
        if mm or mt:
            val, per = _mark_value(mm or mt)
            mark_text = (mm or mt).group(0).strip()
            opened = state["pending"] or ([tuple(state["path"])] if state["path"] else [])
            if state["or_pending"] and q.units:
                q.or_points.append(len(q.units))
                state["or_pending"] = False
            if per is not None:
                top = min((len(p) for p in opened), default=0)
                heads = [p for i, p in enumerate(opened) if len(p) == top and p not in opened[:i]]
                if opened and len(heads) >= val:
                    starts = [state["opened_at"].get(h, unit_start) for h in heads]
                    for n_h, h in enumerate(heads):
                        end = starts[n_h + 1] - 1 if n_h + 1 < len(heads) else idx
                        q.units.append(Unit(q.number, [h], per, mark_text, [(page, l["bbox"])], start=starts[n_h], end=end))
                else:
                    q.units.append(Unit(q.number, opened, val * per, mark_text, [(page, l["bbox"])], start=unit_start, end=idx))
            else:
                q.units.append(Unit(q.number, opened, val, mark_text, [(page, l["bbox"])], start=unit_start, end=idx))
            if state["pending"]:  # only units that opened new labels start at their first label
                for u in q.units:
                    if u.end == idx and u.start == unit_start:
                        u.start = min(state["opened_at"].get(p_, unit_start) for p_ in u.labels)
            state["pending"] = []
            unit_start = idx + 1
            if mt:
                q.units[-1].lines.append((page, l["bbox"]))
            continue
        if not q.units and not state["path"]:
            q.stem_lines.append((page, l["bbox"], t))
    info["questions"] = questions
    info["lines"] = lines
    return info


def unit_paths(q):
    """Part path per unit: one label -> its path; several labels at the shallowest level -> range "i_iii";
    a second unit under the same path -> "a.2"; no label -> "u{n}"."""
    seen = {}
    for idx, u in enumerate(q.units, 1):
        if not u.labels:
            path = ("u%d" % idx,)
        else:
            leaves = []
            for p in u.labels:
                if p not in leaves and not any(o != p and o[:len(p)] == p for o in u.labels):
                    leaves.append(p)
            if len(leaves) == 1:
                path = leaves[0]
            else:
                common = 0
                while all(len(p) > common for p in leaves) and len({p[common] for p in leaves}) == 1:
                    common += 1
                first, last = leaves[0], leaves[-1]
                path = first[:common] + (f"{first[common]}_{last[common]}",)
        n = seen.get(path, 0)
        seen[path] = n + 1
        u.path = path if n == 0 else path[:-1] + (f"{path[-1]}.{n + 1}",)
    return q


def _head(element):
    return re.split(r"[._]", element)[0]


def _build_node(q, entries, prefix):
    """entries: list of (unit, leaf). Applies a rubric whose context == prefix; recurses where deeper rubrics exist."""
    d = len(prefix)
    rub = next((r for r in q.rubric if r["context"] == prefix), None)
    deeper = [r for r in q.rubric if len(r["context"]) > d and r["context"][:d] == prefix]
    here, groups = [], {}
    for u, leaf in entries:
        if len(u.path) > d and (rub or deeper):
            groups.setdefault(_head(u.path[d]), []).append((u, leaf))
        else:
            here.append(leaf)
    children = list(here)
    group_nodes = {h: (_build_node(q, g, prefix + (h,)) if any(r["context"][:d + 1] == prefix + (h,) for r in deeper)
                       else {"type": "all", "children": [lf for _, lf in g], "source_text": None})
                   for h, g in groups.items()}
    if rub:
        names = rub["labels"]
        chosen = [n for h, n in group_nodes.items() if not names or h in names]
        if len(chosen) <= (rub["k"] or 0):
            rub = None  # the choice's labels carry no separate marks here, so it cannot change the score
    if rub:
        others = [n for h, n in group_nodes.items() if names and h not in names]
        choice = {"type": "best_k", "k": rub["k"], "children": chosen, "source_text": rub["text"]}
        children += others + [choice]
    else:
        children += list(group_nodes.values())
    if len(children) == 1 and children[0]["type"] != "part":
        return children[0]
    return {"type": "all", "children": children, "source_text": None}


def build_rules(info, item_prefix):
    """Rules tree for one paper from its own rubric. Returns (root, items)."""
    items, section_nodes = [], []
    for sec, s in info["sections"].items():
        q_nodes = []
        for q in [q for q in info["questions"] if q.section == sec]:
            unit_paths(q)
            entries = []
            for u in q.units:
                item_id = f"{item_prefix}-Q{q.number}-" + "-".join(u.path)
                items.append({"item_id": item_id, "question": str(q.number), "section": sec,
                              "part_path": list(u.path), "marks": u.marks, "mark_text": u.mark_text,
                              "labels": ["-".join(p) for p in u.labels]})
                entries.append((u, {"type": "part", "item_id": item_id, "children": []}))
            for i in sorted(set(q.or_points), reverse=True):
                if 0 < i < len(entries):
                    (u1, l1), (u2, l2) = entries[i - 1], entries[i]
                    entries[i - 1:i + 1] = [(u1, {"type": "best_k", "k": 1, "children": [l1, l2], "source_text": "or"})]
            node = _build_node(q, entries, ()) if entries else {"type": "all", "children": [], "source_text": None}
            node["question"] = q.number
            q_nodes.append(node)
        answer = s["answer"]
        sec_node = {"type": "best_k" if answer and answer[0] else "all", "k": answer[0] if answer else None,
                    "children": q_nodes, "source_text": answer[1] if answer else None,
                    "section": sec, "printed_marks": s["marks"]}
        section_nodes.append(sec_node)
    root = {"type": "all", "children": section_nodes, "source_text": info["paper_rubric"]}
    return root, items
