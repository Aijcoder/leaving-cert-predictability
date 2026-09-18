"""Rules trees (the build specification §10.4, Appendix A.3, B.2): schema, scoring, max score, all-compulsory variant."""
import copy
from itertools import combinations

import jsonschema
import numpy as np

RULES_NODE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "rules_node",
    "type": "object",
    "required": ["type", "children"],
    "properties": {
        "type": {"enum": ["all", "best_k", "best_k_constrained", "part"]},
        "k": {"type": ["integer", "null"], "minimum": 1},
        "item_id": {"type": ["string", "null"]},
        "groups": {"type": ["array", "null"], "items": {
            "type": "object", "required": ["name", "children_idx", "min"],
            "properties": {"name": {"type": "string"},
                           "children_idx": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                           "min": {"type": "integer", "minimum": 0}}}},
        "children": {"type": "array", "items": {"$ref": "rules_node"}},
        "source_text": {"type": ["string", "null"]},
    },
    "allOf": [
        {"if": {"properties": {"type": {"const": "part"}}},
         "then": {"required": ["item_id"], "properties": {"item_id": {"type": "string"}, "children": {"maxItems": 0}}}},
        {"if": {"properties": {"type": {"enum": ["best_k", "best_k_constrained"]}}},
         "then": {"required": ["k"], "properties": {"k": {"type": "integer"}}}},
        {"if": {"properties": {"type": {"const": "best_k_constrained"}}},
         "then": {"required": ["groups"], "properties": {"groups": {"type": "array", "minItems": 1}}}},
        {"if": {"properties": {"type": {"enum": ["all", "best_k", "best_k_constrained"]}}},
         "then": {"properties": {"children": {"minItems": 1}}}},
    ],
}


def validate(root):
    """JSON Schema check plus semantic checks. Raises ValueError with a readable message."""
    try:
        jsonschema.validate(root, RULES_NODE_SCHEMA)
    except jsonschema.ValidationError as e:
        raise ValueError(f"rules schema: {e.message} at {list(e.absolute_path)}") from None
    seen = set()

    def walk(node, path):
        t = node["type"]
        if t == "part":
            if node["item_id"] in seen:
                raise ValueError(f"item_id used twice: {node['item_id']}")
            seen.add(node["item_id"])
            return
        n = len(node["children"])
        if t in ("best_k", "best_k_constrained") and not 1 <= node["k"] <= n:
            raise ValueError(f"k={node['k']} out of range for {n} children at {path}")
        if t == "best_k_constrained":
            for g in node["groups"]:
                if any(i >= n for i in g["children_idx"]) or g["min"] > len(g["children_idx"]):
                    raise ValueError(f"bad group {g['name']} at {path}")
            if sum(g["min"] for g in node["groups"]) > node["k"]:
                raise ValueError(f"group minimums exceed k at {path}")
        for i, c in enumerate(node["children"]):
            walk(c, f"{path}/{i}")

    walk(root, "")
    return seen


def best_subset_with_group_minimums(vals, k, groups):
    best = None
    for combo in combinations(range(len(vals)), k):
        chosen = set(combo)
        if all(len(chosen & set(g["children_idx"])) >= g["min"] for g in groups):
            s = sum(vals[i] for i in combo)
            best = s if best is None or s > best else best
    if best is None:
        raise ValueError("no subset satisfies the group minimums")
    return best


def score(node, credit):
    """credit: item_id -> achievable marks (Appendix B.2)."""
    t = node["type"]
    if t == "part":
        return credit[node["item_id"]]
    vals = [score(c, credit) for c in node["children"]]
    if t == "all":
        return sum(vals)
    if t == "best_k":
        return sum(sorted(vals, reverse=True)[:node["k"]])
    if t == "best_k_constrained":
        return best_subset_with_group_minimums(vals, node["k"], node["groups"])
    raise ValueError(t)


def item_ids(node):
    if node["type"] == "part":
        return [node["item_id"]]
    return [i for c in node["children"] for i in item_ids(c)]


def max_score(node, marks):
    """The tree evaluated with full marks on every leaf; marks: item_id -> marks."""
    return score(node, marks)


def all_compulsory(node):
    """Copy of the tree with every best_k / best_k_constrained replaced by `all` (SQ1, §15.6)."""
    new = copy.deepcopy(node)

    def walk(n):
        if n["type"] in ("best_k", "best_k_constrained"):
            n["type"], n["k"], n["groups"] = "all", None, None
        for c in n.get("children", []):
            walk(c)

    walk(new)
    return new


def attemptable_share(root, items, revised, credit="proportional"):
    """AS for one sitting (Appendix B.2). items: dicts with item_id, marks, topics[{topic_id, weight}].

    proportional: marks x sum of revised weights / 100; strict: full marks only if every topic is revised.
    UNCLEAR weight never earns credit.
    """
    full = {i["item_id"]: i["marks"] for i in items}
    cred = {}
    for i in items:
        topics = [w for w in i["topics"] if w["topic_id"] != "UNCLEAR"]
        if credit == "proportional":
            cred[i["item_id"]] = i["marks"] * sum(w["weight"] for w in topics if w["topic_id"] in revised) / 100
        elif credit == "strict":
            ok = bool(i["topics"]) and all(w["topic_id"] in revised and w["topic_id"] != "UNCLEAR" for w in i["topics"])
            cred[i["item_id"]] = i["marks"] if ok else 0
        else:
            raise ValueError(credit)
    return score(root, cred) / score(root, full)


def compile_tree(root, item_ids_order):
    """Flatten a rules tree for vectorised scoring: returns a callable f(credit_matrix) -> array of scores.

    credit_matrix has shape (K, n_items) with columns in `item_ids_order`. Semantics match score() exactly.
    """
    index = {i: k for k, i in enumerate(item_ids_order)}

    def build(node):
        t = node["type"]
        if t == "part":
            col = index[node["item_id"]]
            return lambda c: c[:, col]
        children = [build(ch) for ch in node["children"]]
        if t == "all":
            return lambda c: sum(f(c) for f in children)
        if t == "best_k":
            k = node["k"]
            def best_k(c, children=children, k=k):
                vals = np.stack([f(c) for f in children], axis=1)   # (K, n_children)
                part = np.sort(vals, axis=1)[:, ::-1][:, :k]
                return part.sum(axis=1)
            return best_k
        if t == "best_k_constrained":
            k, groups = node["k"], node["groups"]
            combos = [c for c in combinations(range(len(children)), k)
                      if all(len(set(c) & set(g["children_idx"])) >= g["min"] for g in groups)]
            if not combos:
                raise ValueError("no subset satisfies the group minimums")
            def best_k_constrained(c, children=children, combos=combos):
                vals = np.stack([f(c) for f in children], axis=1)
                sums = np.stack([vals[:, list(combo)].sum(axis=1) for combo in combos], axis=1)
                return sums.max(axis=1)
            return best_k_constrained
        raise ValueError(t)

    return build(root)
