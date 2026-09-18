"""Phase 4.1 (PM): outline of the Leaving Certificate Mathematics syllabus (examination from 2015), Ordinary/Higher
level strand tables (pages 17-43): topic headings and learning-outcome bullets (OL column and HL column).
Output: data_derived/codebooks/sources/PM_syllabus_outline.json — [{strand, topic, page, ol:[...], hl:[...]}].
"""
import json
import re

from lcbank.common.paths import DATA_DERIVED
from lcbank.structure.layout import page_lines

DOC = "PM-syllabus-2015"


def outline(first=16, last=43):
    topics, cur, strand = [], None, None
    for p in range(first, last + 1):
        ls = page_lines(DOC, p)
        title = " ".join(l["text"] for l in ls if l["size"] >= 16)
        if "Foundation" in title or "Strand" not in title:
            continue
        strand = re.sub(r"\s+", " ", re.sub(r"[\x07\t]", " ", title.split("–")[0])).strip()
        for l in sorted(ls, key=lambda l: (l["bbox"][1], l["bbox"][0])):
            t = re.sub(r"[\x07\t]", " ", l["text"]).strip()
            x0 = l["bbox"][0]
            if l["bbox"][1] < 85 or l["size"] >= 16:
                continue
            if x0 < 200 and l["bold"]:
                m = re.match(r"^(\d\.\d{1,2})\s+(.*)$", t)
                if m:
                    cur = next((c for c in topics if c["code"] == m.group(1)), None)
                    if cur is None:
                        cur = {"strand": strand, "code": m.group(1), "topic": m.group(2), "page": p,
                               "ol": [], "hl": []}
                        topics.append(cur)
                    cur["_last"] = "name"
                elif cur is not None and cur.get("_last") == "name":
                    cur["topic"] += " " + t
                continue
            if cur is None:
                continue
            col = "ol" if 200 <= x0 < 370 else "hl" if x0 >= 370 else None
            if col is None:
                continue
            cur["_last"] = col
            if re.match(r"^[–-]\s", t) or not cur[col]:
                cur[col].append(re.sub(r"^[–-]\s*", "", t))
            else:
                cur[col][-1] += " " + t
    for c in topics:
        c.pop("_last", None)
        for col in ("ol", "hl"):
            c[col] = [re.sub(r"\s+", " ", b).strip() for b in c[col] if b.strip()]
    return topics


if __name__ == "__main__":
    import sys
    tops = outline()
    (DATA_DERIVED / "codebooks" / "sources" / "PM_syllabus_outline.json").write_text(json.dumps(tops, indent=1, ensure_ascii=False))
    show = set(sys.argv[1:])
    for t in tops:
        print(f"{t['code']} {t['topic'][:50]} (p{t['page']}) OL={len(t['ol'])} HL={len(t['hl'])}")
        if t["code"] in show:
            for col in ("ol", "hl"):
                for b in t[col]:
                    print(f"     {col}: {b[:95]}")
