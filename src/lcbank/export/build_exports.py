"""Phase 9: SQLite + JSON exports, public and full tiers (§16, R1, R2).

export_public/  metadata only: datasets, papers, rules trees, items (no text), topics, item_topics, topic_history,
                sources. No exam text, scheme text or images (R2).
export_full/    the same plus text, stem_text, scheme_text, scheme_steps and crop paths; git-ignored, carries the
                R2 README, never shipped without written SEC permission.
A test scans every table and column name for forbidden forecast fields (R1).
Run: python -m lcbank.export.build_exports
"""
import csv
import json
import sqlite3
from datetime import date

from lcbank.common import config
from lcbank.common.paths import DATA_DERIVED, EXPORT_FULL, EXPORT_PUBLIC, MANIFEST

FORBIDDEN = ("prob_", "predicted_", "due", "likely", "next_", "forecast", "will_appear", "expected_topic")
PUBLIC_ITEM_FIELDS = ["item_id", "dataset", "year", "sitting_type", "paper", "section", "question", "part_path",
                      "marks", "marks_source", "stem_ref", "text_quality", "extraction_method", "label_
                      "label_model", "prompt_sha256", "codebook_version", "analysis_excluded"]
PRIVATE_ITEM_FIELDS = ["text", "stem_text", "scheme_text", "scheme_steps", "crop_paths", "mark_text", "scale",
                       "label_confidence", "label_basis", "needs_diagram", "scheme_align", "crop_scope"]
DATASETS = ("PHY", "PM", "PMT", "AM2")


def check_names(names):
    bad = [n for n in names if any(f in n.lower() for f in FORBIDDEN)]
    if bad:
        raise ValueError(f"forbidden forecast-like field names (R1): {bad}")
    return True


def load_items(dataset):
    path = DATA_DERIVED / "items" / f"items_labelled_{dataset}.jsonl"
    if not path.exists():
        path = DATA_DERIVED / "items" / f"items_structured_{dataset}.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_history(dataset):
    path = DATA_DERIVED / "stats" / f"topic_history_{dataset}.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def create_schema(con, full):
    con.executescript("""
    CREATE TABLE datasets (dataset TEXT PRIMARY KEY, subject TEXT, level TEXT, syllabus TEXT, year_first INTEGER,
        year_last INTEGER, role TEXT, codebook_version TEXT, papers_per_sitting INTEGER);
    CREATE TABLE papers (paper_id TEXT PRIMARY KEY, dataset TEXT, year INTEGER, sitting_type TEXT, paper INTEGER,
        printed_total REAL, max_score REAL, rules_valid INTEGER, source_url TEXT, doc_id TEXT);
    CREATE TABLE rules (paper_id TEXT PRIMARY KEY, rules_json TEXT);
    CREATE TABLE topics (topic_id TEXT PRIMARY KEY, codebook TEXT, L1 TEXT, name TEXT, syllabus_ref TEXT,
        includes TEXT, excludes TEXT, boundary_rules TEXT);
    CREATE TABLE item_topics (item_id TEXT, topic_id TEXT, weight REAL, PRIMARY KEY (item_id, topic_id));
    CREATE TABLE topic_history (dataset TEXT, topic_id TEXT, sittings_present INTEGER, appearance_count INTEGER,
        appearance_rate REAL, mean_share REAL, sd_share REAL, longest_gap_sittings INTEGER, first_year INTEGER,
        last_year INTEGER, PRIMARY KEY (dataset, topic_id));
    CREATE TABLE sources (item_id TEXT PRIMARY KEY, doc_id TEXT, url TEXT, page INTEGER, bbox TEXT);
    CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT);
    """)
    cols = ", ".join(f"{c} TEXT" for c in PUBLIC_ITEM_FIELDS if c != "item_id")
    extra = (", " + ", ".join(f"{c} TEXT" for c in PRIVATE_ITEM_FIELDS)) if full else ""
    con.execute(f"CREATE TABLE items (item_id TEXT PRIMARY KEY, {cols}{extra})")
    names = [r[1] for t in ("datasets", "papers", "rules", "topics", "item_topics", "topic_history", "sources",
                            "items", "build_info")
             for r in con.execute(f"PRAGMA table_info({t})")]
    check_names(names + ["datasets", "papers", "rules", "topics", "item_topics", "topic_history", "sources", "items"])


def build(full, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / ("lc_bank_full.sqlite" if full else "lc_bank_public.sqlite")
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    create_schema(con, full)
    cfg = config.load("datasets")["datasets"]
    manifest = {r["doc_id"]: r for r in csv.DictReader(open(MANIFEST, newline="", encoding="utf-8"))}
    json_out = {"datasets": [], "papers": [], "topics": [], "items": [], "item_topics": [], "topic_history": []}

    for ds in DATASETS:
        d = cfg[ds]
        row = (ds, d["subject"], d["level"], d["syllabus"], d["year_first"], d["year_last"],
               d.get("role_resolved") or d["role"], f"{'PM' if ds == 'PMT' else ds}_v1", d["papers_per_sitting"])
        con.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?)", row)
        json_out["datasets"].append(dict(zip([c[1] for c in con.execute("PRAGMA table_info(datasets)")], row)))

    for path in sorted((DATA_DERIVED / "rules").glob("*.json")):
        doc = json.loads(path.read_text())
        if "root" not in doc or "paper" not in doc:
            continue
        src = manifest.get(doc["doc_id"], {})
        row = (doc["paper_id"], doc["dataset"], doc["year"], doc["sitting_type"], doc["paper"],
               doc.get("printed_total"), doc.get("max_score"), int(bool(doc.get("valid"))), src.get("url"),
               doc["doc_id"])
        con.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)", row)
        con.execute("INSERT INTO rules VALUES (?,?)", (doc["paper_id"], json.dumps(doc["root"], ensure_ascii=False)))
        json_out["papers"].append(dict(zip([c[1] for c in con.execute("PRAGMA table_info(papers)")], row)))

    for cb_name in ("PHY", "PM", "AM2"):
        cb = json.loads((DATA_DERIVED / "codebooks" / f"{cb_name}_v1.json").read_text())
        for t in cb["topics"]:
            row = (t["topic_id"], cb["version"], t["L1"], t["name"], t["syllabus_ref"], " | ".join(t["includes"]),
                   " | ".join(t["excludes"]), " | ".join(t["boundary_rules"]))
            con.execute("INSERT OR IGNORE INTO topics VALUES (?,?,?,?,?,?,?,?)", row)
            json_out["topics"].append(dict(zip([c[1] for c in con.execute("PRAGMA table_info(topics)")], row)))

    item_cols = PUBLIC_ITEM_FIELDS + (PRIVATE_ITEM_FIELDS if full else [])
    for ds in DATASETS:
        for it in load_items(ds):
            values = []
            for c in item_cols:
                v = it.get(c)
                values.append(json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
            con.execute(f"INSERT INTO items VALUES ({','.join('?' * len(item_cols))})", values)
            json_out["items"].append({c: it.get(c) for c in item_cols})
            for w in it.get("topics") or []:
                con.execute("INSERT OR REPLACE INTO item_topics VALUES (?,?,?)",
                            (it["item_id"], w["topic_id"], w["weight"]))
                json_out["item_topics"].append({"item_id": it["item_id"], "topic_id": w["topic_id"],
                                                "weight": w["weight"]})
            s = it["source"]
            con.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?)",
                        (it["item_id"], s["doc_id"], s["url"], s.get("page"), json.dumps(s.get("bbox"))))
        for h in load_history(ds):
            con.execute("INSERT OR REPLACE INTO topic_history VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (h["dataset"], h["topic_id"], h["sittings_present"], h["appearance_count"],
                         h["appearance_rate"], h["mean_share"], h["sd_share"], h["longest_gap_sittings"],
                         h["first_year"] or None, h["last_year"] or None))
            json_out["topic_history"].append(h)

    frozen = DATA_DERIVED / "labels" / "FROZEN.json"
    info = {"built": date.today().isoformat(), "tier": "full" if full else "public",
            "codebooks": json.loads((DATA_DERIVED / "codebooks" / "FROZEN.json").read_text()),
            "labels": json.loads(frozen.read_text()) if frozen.exists() else "labels not frozen yet",
            "last_available_sitting": config.last_available_sitting(),
            "rule": "Retrospective data only. No forecast fields, no predictions for any future sitting (R1)."}
    for k, v in info.items():
        con.execute("INSERT OR REPLACE INTO build_info VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
    con.commit()
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("datasets", "papers", "rules", "topics", "items", "item_topics", "topic_history", "sources")}
    con.close()

    (out_dir / "json").mkdir(exist_ok=True)
    for name, rows in json_out.items():
        (out_dir / "json" / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (out_dir / "json" / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return db, counts


def write_full_readme():
    (EXPORT_FULL / "README.md").write_text(
        "DO NOT SHIP OR PUBLISH WITHOUT WRITTEN PERMISSION FROM THE SEC.\n\n"
        "This export contains State Examinations Commission exam text, marking-scheme text and page crops. SEC "
        "material is provided for personal, non-commercial use; reproducing or publishing any part of it needs the "
        "SEC's written permission (the build specification R2).\n\n"
        "This folder is git-ignored and must not be uploaded anywhere. The public tier "
        "(`export_public/`) holds only derived metadata and is the one to share.\n", encoding="utf-8")


if __name__ == "__main__":
    pub_db, pub_counts = build(full=False, out_dir=EXPORT_PUBLIC)
    full_db, full_counts = build(full=True, out_dir=EXPORT_FULL)
    write_full_readme()
    print("public:", pub_db.name, pub_counts)
    print("full:  ", full_db.name, full_counts)
