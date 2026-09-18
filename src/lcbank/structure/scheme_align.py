"""Phase 3.2: attach a marking-scheme excerpt to each leaf by anchoring the leaf's prompt text in the scheme.

The scheme text is flattened (NFKC, lower case, whitespace collapsed). For each leaf, candidate anchors are its
prompt lines (labels and marks stripped, >= 20 characters); the first 30 characters of the first prompt found
after the previous leaf's anchor fixes the leaf's position. A leaf's excerpt runs to the next found anchor.
Leaves without an anchor get the excerpt between their neighbours' anchors (marked `approximate`).
"""
import re
import unicodedata

from lcbank.structure.layout import n_pages, page_lines

STRIP_RE = re.compile(r"^(?:\d{1,2}\s*\.\s*)?(?:(?:Part\s+)?\([a-j]|\((?:i{1,3}|iv|vi{0,3}|ix|x)\)\s*)*")


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()


def scheme_flat(doc_id):
    parts, spans = [], []
    pos = 0
    for p in range(1, n_pages(doc_id) + 1):
        for l in page_lines(doc_id, p):
            t = norm(l["text"]) + " "
            parts.append(t)
            spans.append((pos, p))
            pos += len(t)
    return "".join(parts), spans


def prompts(text):
    out = []
    for line in text.split("\n"):
        s = re.sub(r"\(\s*\d+\s*(?:[×x+]\s*\d+\s*)*\)\s*$", "", line)
        s = re.sub(r"^\s*(?:\d{1,2}\s*\.\s*)?(?:(?:part\s+)?\((?:[a-j]|i{1,3}|iv|vi{0,3}|ix|x)\)\s*)+", "", s, flags=re.I)
        s = norm(s)
        if len(s) >= 20 and re.search(r"[a-z]{4}", s):
            out.append(s)
    return out


def align(items, scheme_doc_id):
    """items: leaves of one paper in paper order (dicts with 'text'). Returns list of (excerpt,."""
    flat, _ = scheme_flat(scheme_doc_id)
    pos, anchors = 0, []
    for it in items:
        found = None
        for pr in prompts(it["text"]):
            k = flat.find(pr[:30], pos)
            if k >= 0:
                found = k
                break
        anchors.append(found)
        if found is not None:
            pos = found + 1
    out = []
    known = [a for a in anchors if a is not None]
    for i, a in enumerate(anchors):
        nxt = next((b for b in anchors[i + 1:] if b is not None), None)
        if a is not None:
            end = nxt if nxt is not None else min(len(flat), a + 1500)
            out.append((flat[a:end][:1500], "anchored"))
        else:
            prev = next((b for b in reversed(anchors[:i]) if b is not None), None)
            if prev is not None and nxt is not None and nxt - prev < 3000:
                out.append((flat[prev:nxt][:1500], "approximate"))
            else:
                out.append((None, "not_found"))
    return out, len(known)


def question_blocks(flat, n_questions):
    """Start offsets of questions 1..n in the flattened scheme ("question N" or " N. "), in order."""
    starts, pos = {}, 0
    for n in range(1, n_questions + 1):
        m = re.compile(rf"(?:question\s+{n}\b|(?:^|\s){n}\.\s)").search(flat, pos)
        if not m:
            continue
        starts[n] = m.start()
        pos = m.end()
    return starts


def label_fallback(items, flat, results):
    """For leaves not anchored by prompt text: slice the scheme by question block and label path."""
    n_q = max(int(it["question"]) for it in items)
    starts = question_blocks(flat, n_q)
    ordered = sorted(starts.items())
    ends = {n: (ordered[i + 1][1] if i + 1 < len(ordered) else len(flat)) for i, (n, _) in enumerate(ordered)}
    out = []
    for it, (excerpt, in zip(items, results):
        if
            out.append((excerpt,)
            continue
        qn = int(it["question"])
        if qn not in starts:
            out.append((excerpt,)
            continue
        block = flat[starts[qn]:ends[qn]]
        path = [re.split(r"[._]", p)[0] for p in it["part_path"] if not re.fullmatch(r"u\d+", p)]
        pos, ok = 0, bool(path)
        for lab in path:
            k = block.find(f"({lab})", pos)
            if k < 0:
                ok = False
                break
            pos = k
        last = it["part_path"][-1] if it["part_path"] else ""
        if ok and "_" in last:  # range "i_iv": slice through the end label
            k = block.find("(" + re.split(r"[.]", last.split("_")[1])[0] + ")", pos + 3)
            pos_end_label = k if k >= 0 else pos
        else:
            pos_end_label = pos
        if ok:
            nxt = re.search(r"\((?:[a-j]|i{1,3}|iv|vi{0,3}|ix|x)\)", block[pos_end_label + 3:])
            end = pos_end_label + 3 + nxt.start() if nxt else len(block)
            out.append((block[pos:end][:1500], "label_slice"))
        else:
            out.append((block[:1500], "question_block"))
    return out
