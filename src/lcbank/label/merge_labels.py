"""Phase 5.4: merge labels into items_labelled_{dataset}.jsonl and freeze them (§12.4).

Each leaf gets topics[{topic_id, weight}] (Appendix B.1 weights computed from the model's scoring steps),
label_, label_model, prompt_sha256 and codebook_version.
Run: python -m lcbank.label.merge_labels --run final
"""
import argparse
import hashlib
import json
from collections import Counter

from lcbank.common.paths import DATA_DERIVED
from lcbank.label.label_parts import CODEBOOK_FOR, load_items


def merge(dataset, run="final"):
    items = load_items(dataset)
    labels = {}
    path = DATA_DERIVED / "labels" / f"labels_raw_{dataset}.jsonl"
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["run"] == run:
                labels[r["item_id"]] = r
    out, stats = [], Counter()
    for it in items:
        it.pop("qa_   # the human-QA field was removed from the project (EX-011)
        r = labels.get(it["item_id"])
        if r is None:
            stats["unlabelled"] += 1
            it.update(topics=None, label_
                      prompt_sha256=None, codebook_version=None, scheme_steps=None, label_confidence=None,
                      label_basis=None, needs_diagram=None)
        else:
            stats[r["
            resp = r.get("response") or {}
            it.update(topics=r.get("weights"), label_
                      label_model=r["model"], prompt_sha256=r["prompt_sha256"],
                      codebook_version=r.get("codebook_version") or f"{CODEBOOK_FOR[dataset]}_v1",
                      scheme_steps=resp.get("steps"), label_confidence=resp.get("confidence"),
                      label_basis=resp.get("basis"), needs_diagram=resp.get("needs_diagram"))
        out.append(it)
    dest = DATA_DERIVED / "items" / f"items_labelled_{dataset}.jsonl"
    with open(dest, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"dataset": dataset, "items": len(out), **dict(stats), "file": str(dest.name), "sha256": sha}


def freeze(results, run):
    path = DATA_DERIVED / "labels" / "FROZEN.json"
    payload = {r["dataset"]: {k: r[k] for k in ("file", "sha256", "items", "ok", "needs_human", "unlabelled")
                              if k in r} for r in results}
    path.write_text(json.dumps({"run": run, "datasets": payload}, indent=1) + "\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final")
    ap.add_argument("--datasets", default="PHY,PM,PMT,AM2")
    ap.add_argument("--freeze", action="store_true", help="write data_derived/labels/FROZEN.json")
    a = ap.parse_args()
    res = [merge(ds, a.run) for ds in a.datasets.split(",")]
    for r in res:
        print(r)
    if a.freeze:
        print("frozen:", freeze(res, a.run))
