"""Evidence-set interventions and original-source field supervision."""
from __future__ import annotations


def candidate_sets(unit_ids):
    """Nested evidence and removal interventions for complementary/redundant material."""
    ids = tuple(unit_ids)
    variants = ((), ids[:1], ids[:2], ids[:4], ids, ids[1:])
    return tuple(dict.fromkeys(variants))


def source_bindings(row, annotations):
    """Translate literal source selections into the shared field-head labels."""
    units = {unit["unit_id"]: unit for unit in row["units"]}
    result = []
    for annotation in annotations:
        unit = units[annotation["unit_id"]]
        source = unit["source_spans"][0]
        quoted = annotation["text"]
        start = 0
        for _ in range(annotation.get("occurrence", 0)+1):
            found = unit["text"].index(quoted, start)
            start = found+len(quoted)
        span = {"span_id": f"{unit['unit_id']}:{found}:{start}", "source_key": source["source_key"],
                "start": source["start"]+found, "end": source["start"]+start}
        result.append({"need_id": annotation["need_id"], "role": annotation["role"], "source_span": span,
            "condition_id": annotation.get("condition_id", ""), "relation_group": annotation.get("relation_group", "")})
    return result
