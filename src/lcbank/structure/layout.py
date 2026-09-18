"""Line-level view of the per-page text JSON (Phase 2 output): text, bbox, font size, bold."""
import json
import unicodedata

from lcbank.common.paths import TEXT


def page_lines(doc_id, page):
    d = json.loads((TEXT / doc_id / f"p{page}.json").read_text(encoding="utf-8"))
    out = []
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = unicodedata.normalize("NFKC", "".join(s["text"] for s in ln["spans"])).strip()
            text = text.replace("\u019f", "ti")  # some SEC fonts encode the "ti" ligature as "Ɵ" (e.g. "QuesƟon")
            out.append({"page": page, "text": text, "bbox": ln["bbox"],
                        "size": round(max(s["size"] for s in spans), 1),
                        "bold": any((s["flags"] & 16) or "Bold" in s["font"] for s in spans),
                        "font": spans[0]["font"]})
    # reading order: cluster lines into rows (|dy| <= 3 pt), then left to right
    out.sort(key=lambda l: l["bbox"][1])
    rows, row_y = [], None
    for l in out:
        if row_y is None or l["bbox"][1] - row_y > 3:
            rows.append([])
            row_y = l["bbox"][1]
        rows[-1].append(l)
    return [l for row in rows for l in sorted(row, key=lambda l: l["bbox"][0])]


def n_pages(doc_id):
    return len(json.loads((TEXT / doc_id / "pages.json").read_text())["pages"])


def dump(doc_id, pages, width=70):
    for p in pages:
        for l in page_lines(doc_id, p):
            x0, y0 = l["bbox"][0], l["bbox"][1]
            print(f"p{p} x{x0:5.0f} y{y0:4.0f} s{l['size']:4.1f}{'B' if l['bold'] else ' '} {l['text'][:width]}")
