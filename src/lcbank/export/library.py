"""Phase 9.2: pure library functions for the practice app (§16.2).

Everything here is retrospective (R1): historical facts and a what-if calculator over past papers. Nothing ranks,
scores or predicts topics for a future sitting, and no function takes a future sitting as a target.
"""
import json
import sqlite3
from collections import defaultdict

from lcbank.common.paths import EXPORT_PUBLIC
from lcbank.structure.rules import attemptable_share

WHATIF_LABEL = "On past papers, this revision set would have covered …"


def connect(db=None):
    con = sqlite3.connect(db or (EXPORT_PUBLIC / "lc_bank_public.sqlite"))
    con.row_factory = sqlite3.Row
    return con


def filter_items(con, dataset, topics=None, years=None, marks_range=None, paper=None):
    """Items matching the filters, metadata only, ordered by year, paper, question."""
    sql = ["SELECT i.* FROM items i WHERE i.dataset = ?"]
    args = [dataset]
    if years:
        sql.append(f"AND CAST(i.year AS INTEGER) IN ({','.join('?' * len(years))})")
        args += list(years)
    if paper is not None:
        sql.append("AND CAST(i.paper AS INTEGER) = ?")
        args.append(int(paper))
    if marks_range:
        sql.append("AND CAST(i.marks AS REAL) BETWEEN ? AND ?")
        args += [float(marks_range[0]), float(marks_range[1])]
    if topics:
        sql.append(f"AND i.item_id IN (SELECT item_id FROM item_topics WHERE topic_id IN "
                   f"({','.join('?' * len(topics))}))")
        args += list(topics)
    sql.append("ORDER BY CAST(i.year AS INTEGER), CAST(i.paper AS INTEGER), i.question, i.item_id")
    return [dict(r) for r in con.execute(" ".join(sql), args)]


def topic_history(con, dataset, topic_id):
    """Historical facts for one topic: appearances, mean share, longest gap, first and last year (§16.2)."""
    row = con.execute("SELECT * FROM topic_history WHERE dataset = ? AND topic_id = ?", (dataset, topic_id)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["statement"] = (f"appeared in {out['appearance_count']} of {out['sittings_present']} past sittings; "
                        f"mean share {float(out['mean_share']):.1%}")
    return out


def _sitting_items(con, dataset, year):
    items = [dict(r) for r in con.execute(
        "SELECT item_id, marks, paper FROM items WHERE dataset = ? AND CAST(year AS INTEGER) = ? AND sitting_type = 'main'",
        (dataset, year))]
    weights = defaultdict(list)
    ids = [i["item_id"] for i in items]
    for chunk in range(0, len(ids), 500):
        part = ids[chunk:chunk + 500]
        for r in con.execute(f"SELECT item_id, topic_id, weight FROM item_topics WHERE item_id IN "
                             f"({','.join('?' * len(part))})", part):
            weights[r["item_id"]].append({"topic_id": r["topic_id"], "weight": float(r["weight"])})
    for it in items:
        it["marks"] = float(it["marks"])
        it["topics"] = weights.get(it["item_id"], [])
    return items


def _sitting_rules(con, dataset, year):
    roots = [json.loads(r["rules_json"]) for r in con.execute(
        "SELECT r.rules_json FROM rules r JOIN papers p USING (paper_id) "
        "WHERE p.dataset = ? AND p.year = ? AND p.sitting_type = 'main' ORDER BY p.paper", (dataset, year))]
    if not roots:
        return None
    return roots[0] if len(roots) == 1 else {"type": "all", "children": roots, "source_text": "one sitting"}


def past_paper_whatif(con, dataset, revised_topic_ids, credit="proportional"):
    """For each past sitting, the share of that paper a student revising these topics could have attempted.

    Retrospective only: it never extrapolates to a future sitting (R1).
    """
    years = [int(r["year"]) for r in con.execute(
        "SELECT DISTINCT year FROM papers WHERE dataset = ? AND sitting_type = 'main' ORDER BY CAST(year AS INTEGER)",
        (dataset,))]
    revised = set(revised_topic_ids)
    per_sitting = {}
    for year in years:
        root = _sitting_rules(con, dataset, year)
        items = _sitting_items(con, dataset, year)
        if not root or not items or not any(it["topics"] for it in items):
            continue
        per_sitting[year] = round(attemptable_share(root, items, revised, credit), 4)
    values = sorted(per_sitting.values())
    summary = {}
    if values:
        mid = len(values) // 2
        summary = {"median": values[mid] if len(values) % 2 else round((values[mid - 1] + values[mid]) / 2, 4),
                   "min": values[0], "max": values[-1]}
    return {"label": WHATIF_LABEL, "dataset": dataset, "credit": credit, "topics": sorted(revised),
            "per_sitting": per_sitting, **summary}


def practice_set(con, dataset, topic_ids, n, seed, stratify_by_year=True):
    """Random or year-stratified sample of items for practice. Never uses timing-method scores (§16.2)."""
    import random
    items = filter_items(con, dataset, topics=topic_ids)
    rng = random.Random(seed)
    if not stratify_by_year:
        return rng.sample(items, min(n, len(items)))
    by_year = defaultdict(list)
    for it in items:
        by_year[it["year"]].append(it)
    years, out = sorted(by_year), []
    while len(out) < min(n, len(items)):
        for y in years:
            pool = [i for i in by_year[y] if i not in out]
            if pool:
                out.append(rng.choice(pool))
            if len(out) >= min(n, len(items)):
                break
    return out
