"""Codebook helpers: A.4 schema, validation, phrase extraction from syllabus text, freezing (§11)."""
import hashlib
import json
import re
from datetime import date

import jsonschema

from lcbank.common.paths import DATA_DERIVED

TOPIC_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["topic_id", "L1", "name", "syllabus_ref", "includes", "excludes", "boundary_rules", "examples"],
    "additionalProperties": False,
    "properties": {
        "topic_id": {"type": "string", "pattern": r"^[A-Z0-9]+\.[A-Z]{3}\.\d{2}$"},
        "L1": {"type": "string", "minLength": 1},
        "name": {"type": "string", "minLength": 3},
        "syllabus_ref": {"type": "string", "minLength": 3},
        "includes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "excludes": {"type": "array", "items": {"type": "string"}},
        "boundary_rules": {"type": "array", "items": {"type": "string"}},
        "examples": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
}

CODEBOOK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["dataset", "version", "source_documents", "global_boundary_rules", "unclear", "topics"],
    "properties": {
        "dataset": {"type": "string"},
        "version": {"type": "string"},
        "topics": {"type": "array", "items": TOPIC_SCHEMA, "minItems": 15, "maxItems": 40},
    },
}


def phrases(text, max_n=8, max_len=110):
    """Short phrases from syllabus 'depth of treatment' text (page numbers dropped)."""
    text = re.sub(r"\s+\d{1,2}$", "", text.strip())
    parts = [p.strip(" .;") for p in re.split(r"(?<=[.;])\s+|•", text) if p.strip(" .;")]
    out = []
    for p in parts:
        if len(p) < 4 or re.fullmatch(r"[\d\s.]+", p):
            continue
        out.append(p[:max_len])
        if len(out) >= max_n:
            break
    return out


def validate_codebook(cb):
    jsonschema.validate(cb, CODEBOOK_SCHEMA)
    ids = [t["topic_id"] for t in cb["topics"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate topic_id")
    if any(i == "UNCLEAR" for i in ids):
        raise ValueError("UNCLEAR is implicit and must not be listed as a topic")
    return True


def write_codebook(cb, version="v1"):
    validate_codebook(cb)
    out = DATA_DERIVED / "codebooks" / f"{cb['dataset']}_{version}.json"
    data = json.dumps(cb, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    out.write_text(data, encoding="utf-8")
    return out, hashlib.sha256(data.encode("utf-8")).hexdigest()


def header(dataset, version, sources, global_rules):
    return {"dataset": dataset, "version": f"{dataset}_{version}", "created": date.today().isoformat(),
            "source_documents": sources, "global_boundary_rules": global_rules,
            "unclear": "UNCLEAR is always a valid topic_id: use it when a scoring step cannot be assigned to one "
                       "topic with reasonable confidence, or when the content is outside the syllabus. "
                       "UNCLEAR weight never earns credit in the analysis."}
