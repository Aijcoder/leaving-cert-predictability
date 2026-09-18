"""Phase 4.1 (PHY): outline of the Higher Level syllabus content table (pages 29-48 of the 1999 syllabus).

Columns: Content (bold headings, x < 170), Depth of Treatment (x 175-300), Activities (x 300-425), STS (x > 425).
Output: data_derived/codebooks/sources/PHY_syllabus_outline.json — [{section, subsection, heading, depth, page}].
"""
import json
import re

from lcbank.common.paths import DATA_DERIVED
from lcbank.structure.layout import page_lines

DOC = "PHY-syllabus-1999"
SKIP = {"Content", "Depth of Treatment", "Activities", "STS"}


def outline(first=29, last=48):
    out, section, subsection, cur = [], None, None, None
    pending_heading = None
    for p in range(first, last + 1):
        ls = [l for l in page_lines(DOC, p) if 60 < l["bbox"][1] < 790 and l["text"].strip() not in SKIP]
        for l in ls:
            t, x0, x1 = l["text"].strip(), l["bbox"][0], l["bbox"][2]
            if l["bold"] and 200 < x0 < 320 and t.isupper():
                section = t.replace("(CONTINUED)", "").strip()
                continue
            if l["bold"] and x0 < 170 and x1 < 200:
                if t.isupper() and not re.match(r"^\d", t):
                    subsection = t if not t.startswith("OPTION") else t
                    continue
                if re.match(r"^\d+\.\s", t):
                    cur = {"section": section, "subsection": subsection, "heading": t, "depth": [], "page": p}
                    out.append(cur)
                    pending_heading = cur
                    continue
                if cur is not None and not t.isupper():
                    cur["heading"] += " " + t  # wrapped heading line (may follow the first depth line)
                    continue
            if cur is not None and 175 <= x0 < 300:
                cur["depth"].append(t)
                pending_heading = None
    for o in out:
        o["heading"] = re.sub(r"\s+", " ", o["heading"].replace("- ", "")).strip()
        o["depth"] = re.sub(r"\s+", " ", " ".join(o["depth"]).replace("- ", "")).strip()
    return out


if __name__ == "__main__":
    items = outline()
    path = DATA_DERIVED / "codebooks" / "sources" / "PHY_syllabus_outline.json"
    path.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    for i, o in enumerate(items):
        print(f"{i:2d} [{(o['section'] or '')[:12]}/{(o['subsection'] or '')[:14]}] {o['heading'][:38]} :: {o['depth'][:70]}")
