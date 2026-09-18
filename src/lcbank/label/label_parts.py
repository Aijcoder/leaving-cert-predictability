"""Phase 5: LLM topic labelling of leaves (§12.1 binding design; token economy D-015).

One leaf per call. System message = fixed prefix per dataset (instructions + compact codebook JSON), so provider
prefix caching can apply. User message = compact JSON of the leaf. The model returns scoring steps with one
topic_id each; this script validates them and computes weights (Appendix B.1). Resumable; every call is logged by
LLMClient. Run: python -m lcbank.label.label_parts --dataset PHY --items dev_sample --run v1
"""
import argparse
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from lcbank.common import config
from lcbank.common.paths import DATA_DERIVED
from lcbank.label.llm_client import AllModelsUnavailable, BudgetExceeded, LLMClient
from lcbank.label.weights import compute_weights

LABELS = DATA_DERIVED / "labels"
PROMPTS = LABELS / "prompts"
SUBJECT = {"PHY": "Physics", "PM": "Mathematics", "PMT": "Mathematics", "AM2": "Applied Mathematics"}
CODEBOOK_FOR = {"PHY": "PHY", "PM": "PM", "PMT": "PM", "AM2": "AM2"}
LIMITS = {"stem": 700, "text": 1200, "scheme": 1000}

# Symbol-font private-use characters in SEC PDFs: U+F020-U+F07E carry the Symbol font's ASCII slots
SYMBOL_GREEK = dict(zip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        "αβχδεφγηιϕκλμνοπθρστυϖωξψζΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ"))
SYMBOL_HIGH = {0xA3: "≤", 0xA5: "∞", 0xAE: "→", 0xB0: "°", 0xB1: "±", 0xB3: "≥", 0xB4: "×", 0xB8: "÷", 0xB9: "≠",
               0xBB: "≈", 0xCE: "∈", 0xD6: "√", 0xF2: "∫", 0xD7: "·", 0xA2: "′", 0xB6: "∂", 0xE5: "∑", 0xC5: "⊕",
               0xC6: "∅", 0xC7: "∩", 0xC8: "∪", 0xD0: "∠", 0xDE: "⇒", 0xDB: "⇔", 0xBE: "—", 0xA4: "⁄"}

PROMPT_V1 = """You label Leaving Certificate Higher Level {subject} exam parts with syllabus topics.
Task: split the part's marks into scoring steps as the marking scheme awards them, and give each step exactly one topic_id from CODEBOOK, or "UNCLEAR".
Rules:
- Step marks are numbers that add up exactly to the part's "marks".
- If "scheme" shows how marks are awarded, follow it (basis "marking_scheme"); otherwise split by judgement (basis "judgement").
- Use as few steps as needed; merge adjacent steps with the same topic.
- Follow CODEBOOK.rules and the topics' boundary rules. Never output percentages.
- Maths symbols in the text may be garbled. Set needs_diagram true if the part depends on a figure or graph that is not described in the text.
Return one JSON object only:
{{"part_id":"...","steps":[{{"what":"<=8 words","marks":0,"topic_id":"..."}}],"basis":"marking_scheme|judgement","confidence":0.0,"needs_diagram":false,"reason":"<=20 words"}}
CODEBOOK={codebook}"""

PROMPT_V2 = PROMPT_V1.replace(
    "- Follow CODEBOOK.rules and the topics' boundary rules. Never output percentages.",
    "- Follow CODEBOOK.rules and the topics' boundary rules. For each step choose the topic whose includes and "
    "boundary rules best match what the step rewards; if the concept is listed in a topic's excludes, use the topic "
    "named there. Never output percentages.")
# generation settings are part of a prompt version
PROMPT_V3 = PROMPT_V2  # same instructions; lean codebook (see compact_codebook) to halve tokens per call
PROMPT_V4 = PROMPT_V2.replace(
    "choose the topic whose includes and boundary rules best match what the step rewards; if the concept is listed in a "
    "topic's excludes, use the topic named there.",
    "choose the topic whose name best matches what the step rewards in the marking scheme.")

# v5 (owner's option B, test only): main + up to two secondary topics; fixed weights, not the B.1 step design
PROMPT_V5 = """You label Leaving Certificate Higher Level {subject} exam parts with syllabus topics.
Task: read the part and its marking scheme. Choose the main topic (the topic most of the marks reward) and at most two secondary topics that also clearly earn marks, from CODEBOOK, or "UNCLEAR".
Rules:
- Base the choice on what "scheme" awards marks for (basis "marking_scheme"); if there is no scheme, use the question (basis "judgement").
- List a secondary topic only if it earns a clear share of the marks. Follow CODEBOOK.rules.
- Maths symbols in the text may be garbled. Set needs_diagram true if the part depends on a figure or graph that is not described in the text.
Return one JSON object only:
{{"part_id":"...","main_topic":"...","secondary_topics":[],"basis":"marking_scheme|judgement","confidence":0.0,"needs_diagram":false,"reason":"<=20 words"}}
CODEBOOK={codebook}"""
FIXED_WEIGHTS = {0: [100], 1: [70, 30], 2: [60, 20, 20]}

SETTINGS = {"v1": {}, "v2": {"temperature": 0}, "v3": {"temperature": 0, "lean_codebook": True},
            "v4": {"temperature": 0, "lean_codebook": "names"},
            "v5": {"temperature": 0, "lean_codebook": "names", "output": "main_secondary"},
            # v4s: v4 plus JSON-schema structured output (topic_id constrained to the codebook) — owner request
            "v4s": {"temperature": 0, "lean_codebook": "names", "structured": True}}


def clean(text, limit):
    if not text:
        return None
    out = []
    for ch in text:
        o = ord(ch)
        if 0xF020 <= o <= 0xF07E:
            c = chr(o - 0xF000)
            out.append(SYMBOL_GREEK.get(c, c) if c.isalpha() else ("−" if c == "-" else c))
        elif 0xF080 <= o <= 0xF0FF:
            out.append(SYMBOL_HIGH.get(o - 0xF000, ""))
        elif 0xE000 <= o <= 0xF8FF:
            out.append("")
        else:
            out.append(ch)
    s = re.sub(r"[ \t]+", " ", "".join(out))
    s = re.sub(r"\n\s*\n+", "\n", s).strip()
    return s[:limit]


def compact_codebook(dataset, lean=False):
    """Codebook JSON for the prompt. lean=True (v3): up to 3 includes of <= 70 characters per topic;
    lean="names" (v4, v5): topic_id and name only (the question and marking scheme carry the detail)."""
    cb = json.loads((DATA_DERIVED / "codebooks" / f"{CODEBOOK_FOR[dataset]}_v1.json").read_text())
    topics = []
    for t in cb["topics"]:
        row = {k: t[k] for k in ("topic_id", "name", "includes", "excludes", "boundary_rules")}
        if lean == "names":
            row = {"topic_id": t["topic_id"], "name": t["name"]}
        elif lean:
            row["includes"] = [s[:70] for s in t["includes"][:3]]
            row = {k: v for k, v in row.items() if v}
        topics.append(row)
    return cb, json.dumps({"topics": topics, "rules": cb["global_boundary_rules"]}, ensure_ascii=False,
                          separators=(",", ":"))


def system_prompt(dataset, version="v1"):
    _, cbjson = compact_codebook(dataset, lean=SETTINGS[version].get("lean_codebook", False))
    template = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3, "v4": PROMPT_V4, "v4s": PROMPT_V4,
                "v5": PROMPT_V5}[version]
    text = template.format(subject=SUBJECT[dataset], codebook=cbjson)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256((template + json.dumps(SETTINGS[version], sort_keys=True)).encode("utf-8")).hexdigest()
    (PROMPTS / f"prompt_{version}.txt").write_text(template, encoding="utf-8")
    return text, sha


def load_items(dataset):
    items = [json.loads(l) for l in open(DATA_DERIVED / "items" / f"items_structured_{dataset}.jsonl", encoding="utf-8")]
    if dataset == "PHY":
        stems = {}
        path = DATA_DERIVED / "items" / "stems_PHY.jsonl"
        for line in open(path, encoding="utf-8"):
            s = json.loads(line)
            stems[s["stem_id"]] = s["text"]
        for it in items:
            it["stem_text"] = stems.get(it.get("stem_ref"))
    return items


def user_message(it):
    msg = {"part_id": it["item_id"], "year": it["year"], "paper": it["paper"], "question": it["question"],
           "part": "-".join(it["part_path"]), "marks": it["marks"],
           "stem": clean(it.get("stem_text"), LIMITS["stem"]), "text": clean(it.get("text"), LIMITS["text"]),
           "scheme": clean(it.get("scheme_text"), LIMITS["scheme"])}
    return json.dumps({k: v for k, v in msg.items() if v not in (None, "")}, ensure_ascii=False, separators=(",", ":"))


def validate(obj, it, topic_ids):
    if not isinstance(obj, dict):
        return "output is not a JSON object"
    if obj.get("part_id") != it["item_id"]:
        return f"part_id must be {it['item_id']}"
    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return "steps must be a non-empty list"
    total = 0.0
    for s in steps:
        if not isinstance(s, dict) or not isinstance(s.get("marks"), (int, float)) or s["marks"] < 0:
            return "every step needs numeric non-negative marks"
        if s.get("topic_id") not in topic_ids and s.get("topic_id") != "UNCLEAR":
            return f"invalid topic_id {s.get('topic_id')!r}; use a CODEBOOK topic_id or UNCLEAR"
        total += float(s["marks"])
    if abs(total - float(it["marks"])) > 1e-6:
        return f"step marks add up to {total:g} but the part has {it['marks']} marks"
    if obj.get("basis") not in ("marking_scheme", "judgement"):
        return "basis must be marking_scheme or judgement"
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return "confidence must be a number from 0 to 1"
    if not isinstance(obj.get("needs_diagram"), bool):
        return "needs_diagram must be true or false"
    return None


def steps_schema(topic_ids):
    """JSON schema for structured output: topic_id is limited to the codebook ids plus UNCLEAR."""
    return {"name": "part_label", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["part_id", "steps", "basis", "confidence", "needs_diagram", "reason"],
        "properties": {
            "part_id": {"type": "string"},
            "steps": {"type": "array", "minItems": 1, "items": {
                "type": "object", "additionalProperties": False, "required": ["what", "marks", "topic_id"],
                "properties": {"what": {"type": "string"}, "marks": {"type": "number"},
                               "topic_id": {"type": "string", "enum": sorted(topic_ids) + ["UNCLEAR"]}}}},
            "basis": {"type": "string", "enum": ["marking_scheme", "judgement"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "needs_diagram": {"type": "boolean"},
            "reason": {"type": "string"}}}}


def validate_main_secondary(obj, it, topic_ids):
    if not isinstance(obj, dict):
        return "output is not a JSON object"
    if obj.get("part_id") != it["item_id"]:
        return f"part_id must be {it['item_id']}"
    valid = topic_ids | {"UNCLEAR"}
    if obj.get("main_topic") not in valid:
        return f"invalid main_topic {obj.get('main_topic')!r}; use a CODEBOOK topic_id or UNCLEAR"
    sec = obj.get("secondary_topics")
    if not isinstance(sec, list) or len(sec) > 2 or any(s not in valid for s in sec):
        return "secondary_topics must be a list of at most two CODEBOOK topic_ids"
    if obj["main_topic"] in sec or len(set(sec)) != len(sec):
        return "secondary_topics must be distinct and differ from main_topic"
    if obj.get("basis") not in ("marking_scheme", "judgement"):
        return "basis must be marking_scheme or judgement"
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return "confidence must be a number from 0 to 1"
    if not isinstance(obj.get("needs_diagram"), bool):
        return "needs_diagram must be true or false"
    return None


def fixed_weights(obj):
    topics = [obj["main_topic"]] + list(obj["secondary_topics"])
    return [{"topic_id": t, "weight": w} for t, w in zip(topics, FIXED_WEIGHTS[len(topics) - 1])]


def label_one(client, sys_text, it, topic_ids, dataset, run, max_tokens=400, settings=None):
    messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_message(it)}]
    response_format = ({"type": "json_schema", "json_schema": steps_schema(topic_ids)}
                       if (settings or {}).get("structured") else {"type": "json_object"})
    record = {"item_id": it["item_id"], "dataset": dataset, "run": run, "attempts": 0, "
              "tokens_in": 0, "tokens_out": 0, "cny": 0.0, "model": None}
    for attempt in (1, 2):
        record["attempts"] = attempt
        resp, cny = client.chat(messages, log_name=f"{dataset}", purpose=f"label {run}", max_tokens=max_tokens,
                                response_format=response_format,
                                **{k: v for k, v in (settings or {}).items()
                                   if k not in ("lean_codebook", "output", "structured")})
        record["tokens_in"] += resp.usage.prompt_tokens if resp.usage else 0
        record["tokens_out"] += resp.usage.completion_tokens if resp.usage else 0
        record["cny"] += cny
        record["model"] = resp.model
        content = resp.choices[0].message.content or ""
        main_secondary = (settings or {}).get("output") == "main_secondary"
        try:
            obj = json.loads(content)
            err = (validate_main_secondary if main_secondary else validate)(obj, it, topic_ids)
        except json.JSONDecodeError as e:
            obj, err = None, f"invalid JSON: {e.msg}"
        if err is None:
            record.update
                          weights=fixed_weights(obj) if main_secondary else compute_weights(obj["steps"]))
            return record
        record["error"] = err
        messages += [{"role": "assistant", "content": content[:2000]},
                     {"role": "user", "content": f"Invalid output: {err}. Return the corrected JSON object only."}]
    record.update
    return record


def run_labels(dataset, items, run, version="v1", workers=4, pin_model=None):
    config.load_dotenv()
    client = LLMClient()
    client.pin = pin_model
    sys_text, prompt_sha = system_prompt(dataset, version)
    cb, _ = compact_codebook(dataset)
    topic_ids = {t["topic_id"] for t in cb["topics"]}
    LABELS.mkdir(parents=True, exist_ok=True)
    out_path = LABELS / f"labels_raw_{dataset}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            if r["run"] == run:
                done.add(r["item_id"])
    todo = [it for it in items if it["item_id"] not in done]
    lock = threading.Lock()
    stats = {"ok": 0, "needs_human": 0, "cny": 0.0, "tin": 0, "tout": 0}
    print(f"{dataset} run={run} prompt={version} sha={prompt_sha[:12]} todo={len(todo)} (done {len(done)})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(label_one, client, sys_text, it, topic_ids, dataset, run, 400, SETTINGS[version]): it
                   for it in todo}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except (BudgetExceeded, AllModelsUnavailable) as e:
                print(f"STOP: {e}", flush=True)
                pool.shutdown(cancel_futures=True)
                break
            except Exception as e:  # any other API/runtime error: report it and stop cleanly (resumable)
                print(f"STOP (error on {futures[fut]['item_id']}): {type(e).__name__}: {str(e)[:200]}", flush=True)
                pool.shutdown(cancel_futures=True)
                break
            rec.update(prompt_version=version, prompt_sha256=prompt_sha, codebook_version=cb["version"],
                       ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            with lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[rec["
                stats["cny"] += rec["cny"]
                stats["tin"] += rec["tokens_in"]
                stats["tout"] += rec["tokens_out"]
    n = max(1, stats["ok"] + stats["needs_human"])
    print(f"{dataset} run={run}: ok={stats['ok']} needs_human={stats['needs_human']} tokens_in={stats['tin']} "
          f"tokens_out={stats['tout']} mean_in={stats['tin'] / n:.0f} mean_out={stats['tout'] / n:.0f} "
          f"cny={stats['cny']:.4f}", flush=True)
    return stats


def dev_sample(dataset, n, seed=2026):
    """Stratified random sample of leaves from the dataset's development sittings (config/datasets.yaml)."""
    dev = config.load("datasets")["datasets"][dataset]["dev_sittings"] or []
    items = [it for it in load_items(dataset) if it["year"] in dev and it["sitting_type"] in ("main", "project_maths_variant")]
    rng = random.Random(seed)
    by_year = {}
    for it in items:
        by_year.setdefault(it["year"], []).append(it)
    out = []
    years = sorted(by_year)
    for k in range(n):
        pool = by_year[years[k % len(years)]]
        pick = rng.choice([x for x in pool if x not in out])
        out.append(pick)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--items", default="dev_sample")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--run", default="v1")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pin-model", default=None)
    ap.add_argument("--skip-model-of-run", default=None,
                    help="with --pin-model: skip leaves that this earlier run already labelled with the pinned model")
    a = ap.parse_args()
    its = dev_sample(a.dataset, a.n) if a.items == "dev_sample" else load_items(a.dataset)
    if a.skip_model_of_run:
        already = set()
        for line in open(LABELS / f"labels_raw_{a.dataset}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["run"] == a.skip_model_of_run and r["
                already.add(r["item_id"])
        its = [it for it in its if it["item_id"] not in already]
    run_labels(a.dataset, its, a.run, a.version, a.workers, a.pin_model)
