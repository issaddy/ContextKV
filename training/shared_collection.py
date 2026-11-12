"""Evidence-set interventions and original-source field supervision."""
from __future__ import annotations


def candidate_sets(unit_ids):
    """Nested evidence and removal interventions for complementary/redundant material."""
    ids = tuple(unit_ids)
    variants = ((), ids[:1], ids[:2], ids[:4], ids, ids[1:])
    return tuple(dict.fromkeys(variants))


def numbered_text(text):
    import re
    words = tuple(re.finditer(r"[\u3400-\u9fff]|\w+|[^\w\s]", text))
    return " ".join(f"{index}:{word.group()}" for index, word in enumerate(words)), tuple(word.span() for word in words)


def numbered_sources(units):
    shown, locations = [], {}
    for index, unit in enumerate(units):
        identity = f"S{index+1}"
        words, offsets = numbered_text(unit["text"])
        shown.append({"source_id": identity, "title": unit["heading"],
                      "version": unit["source_spans"][0]["source_key"]["version"], "words": words})
        locations[identity] = unit, offsets
    return shown, locations


def word_reference_schema(locations, id_field):
    return {"oneOf": [{"type": "object", "properties": {
        id_field: {"const": identity},
        "word_boundaries": {"type": "array", "minItems": 2, "maxItems": 2,
            "items": {"type": "integer", "minimum": 0, "maximum": len(offsets)-1}}},
        "required": [id_field, "word_boundaries"], "additionalProperties": False}
        for identity, (_, offsets) in locations.items() if offsets]}


def word_offsets(reference, offsets):
    first, last = sorted(reference["word_boundaries"])
    return offsets[first][0], offsets[last][1]


def source_bindings(locations, annotations):
    """Restore exact source offsets from the teacher's numbered word selections."""
    result = []
    for annotation in annotations:
        unit, offsets = locations[annotation["source"]["source_id"]]
        start, end = word_offsets(annotation["source"], offsets)
        source = unit["source_spans"][0]
        span = {"span_id": f"{unit['unit_id']}:{start}:{end}", "source_key": source["source_key"],
                "start": source["start"]+start, "end": source["start"]+end}
        result.append({"need_id": annotation["need_id"], "role": annotation["role"], "source_span": span,
            "condition_id": annotation["condition_id"], "relation_group": annotation["relation_group"]})
    return result
