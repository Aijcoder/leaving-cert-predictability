"""Phase 2: per-page text (layout blocks + bboxes), 150 dpi page PNGs, scanned-page detection (resumable).

Outputs per document:
  data_private/text/{doc_id}/p{n}.json   page.get_text("dict") without image bytes
  data_private/pages/{doc_id}/p{n}.png   150 dpi render
  data_private/text/{doc_id}/pages.json  per-page chars, blank flag, extraction_method
"""
import csv
import json
import shutil
import sys

import pymupdf

from lcbank.common.paths import MANIFEST, PAGES, RAW_PDFS, TEXT

DPI = 150
SCANNED_MIN_CHARS = 150        # §9.3
SCAN_IMAGE_COVERAGE = 0.30     # D-019: a low-text page counts as scanned only if an image covers > 30% of it
MIN_FREE_BYTES = 3 * 1024**3   #


class DiskGuard(RuntimeError):
    pass


def _strip_images(d):
    for block in d.get("blocks", []):
        if block.get("type") == 1:
            block.pop("image", None)
            block["image_omitted"] = True
    return d


def _no_bytes(o):
    if isinstance(o, (bytes, bytearray)):
        return None  # binary payloads (images, masks) are not kept in the text JSON
    raise TypeError(type(o).__name__)


def is_blank(page):
    """Blank if a low-resolution grey render has (almost) no dark pixels."""
    pix = page.get_pixmap(dpi=20, colorspace=pymupdf.csGRAY)
    dark = sum(1 for b in pix.samples if b < 200)
    return dark < 0.001 * pix.width * pix.height


def image_coverage(page):
    area = page.rect.width * page.rect.height
    cov = 0.0
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        cov = max(cov, max(0.0, x1 - x0) * max(0.0, y1 - y0) / area)
    return round(min(cov, 1.0), 3)


def classify(chars, blank, coverage):
    if blank or chars >= SCANNED_MIN_CHARS or coverage <= SCAN_IMAGE_COVERAGE:
        return "text"
    return "needs_ocr"


def extract_doc(row, root=RAW_PDFS.parents[1]):
    out_text = TEXT / row["doc_id"]
    out_pages = PAGES / row["doc_id"]
    summary_path = out_text / "pages.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("sha256") == row["sha256"]:
        return json.loads(summary_path.read_text()), False
    out_text.mkdir(parents=True, exist_ok=True)
    out_pages.mkdir(parents=True, exist_ok=True)
    pages = []
    with pymupdf.open(root / row["local_path"]) as doc:
        for i, page in enumerate(doc, 1):
            if shutil.disk_usage(PAGES).free < MIN_FREE_BYTES:
                raise DiskGuard(f"free disk below {MIN_FREE_BYTES // 1024**3} GB at {row['doc_id']} p{i}")
            d = _strip_images(page.get_text("dict"))
            (out_text / f"p{i}.json").write_text(json.dumps(d, ensure_ascii=False, default=_no_bytes), encoding="utf-8")
            page.get_pixmap(dpi=DPI).save(out_pages / f"p{i}.png")
            chars = len("".join(page.get_text().split()))
            blank = chars == 0 and is_blank(page)
            cov = image_coverage(page)
            method = classify(chars, blank, cov)
            pages.append({"page": i, "chars": chars, "blank": blank, "images": len(page.get_images()),
                          "image_coverage": cov,
                          "extraction_method": method, "width": page.rect.width, "height": page.rect.height})
    summary = {"doc_id": row["doc_id"], "sha256": row["sha256"], "dpi": DPI, "pages": pages}
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary, True


def reclassify():
    """Re-apply classify() to existing pages.json files without re-rendering (keeps ocr/vision results)."""
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["sha256"]]
    for row in rows:
        path = TEXT / row["doc_id"] / "pages.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        with pymupdf.open(RAW_PDFS.parents[1] / row["local_path"]) as doc:
            for p in summary["pages"]:
                if p["extraction_method"] in ("ocr", "vision") or p.get("visual_check"):
                    continue
                p["image_coverage"] = image_coverage(doc[p["page"] - 1])
                p["extraction_method"] = classify(p["chars"], p["blank"], p["image_coverage"])
        path.write_text(json.dumps(summary, indent=1), encoding="utf-8")


def run(doc_types=("paper", "scheme", "syllabus")):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["sha256"] and r["title_check"] == "pass" and r["doc_type"] in doc_types]
    done = 0
    for n, row in enumerate(rows, 1):
        summary, new = extract_doc(row)
        done += new
        if new:
            methods = {}
            for p in summary["pages"]:
                methods[p["extraction_method"]] = methods.get(p["extraction_method"], 0) + 1
            print(f"[{n}/{len(rows)}] {row['doc_id']} pages={len(summary['pages'])} {methods}", flush=True)
    print(f"extracted {done} new documents; {len(rows)} eligible", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["reclassify"]:
        reclassify()
    else:
        run(tuple(sys.argv[1:]) or ("paper", "scheme", "syllabus"))
