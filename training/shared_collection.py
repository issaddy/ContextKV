"""Evidence-set interventions and original-source field supervision."""
from __future__ import annotations

import json

from controller.source_windows import numbered_text


def candidate_sets(frontier, selected):
    """Measure the serving choice, adjacent affordable sets, and the empty control."""
    pivot = min(range(len(frontier)), key=lambda i: abs(len(frontier[i]) - len(selected)))
    indices = sorted({0, min(1, len(frontier)-1), max(0, pivot-1), pivot,
                      min(pivot+1, len(frontier)-1), len(frontier)-1})
    sets = {frozenset(frontier[i]): frontier[i] for i in indices}
    sets[frozenset(selected)] = tuple(selected)
    return tuple(sets.values())


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


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


async def reference_coverage(client, endpoint, model, question, reference, answers):
    async def ask(instruction, payload, schema):
        response = await client.post(endpoint.rstrip("/") + "/v1/chat/completions", json={
            "model": model, "temperature": 0, "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": schema}},
            "messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]})
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

    facts = (await ask("Extract only the necessary facts or requested actions explicitly expressed in the "
        "reference. Keep qualifications with their fact. For a clarification, identify the information "
        "the reference asks the user to supply. Do not add background information.",
        {"question": question, "reference": reference},
        object_schema({"facts": {"type": "array", "minItems": 1, "items": {"type": "string"}}})))["facts"]
    # Blind the grader to the selection policy and collapse identical answers.
    candidates = {f"A{i}": text for i, text in enumerate(sorted(set(answers)))}
    criteria = {f"F{i}": fact for i, fact in enumerate(facts)}
    marks = await ask("For each criterion, identify the answers that convey it correctly, including "
        "its conditions. Accept paraphrases. A clarification must ask for the relevant information.",
        {"question": question, "criteria": criteria, "answers": candidates},
        object_schema({key: object_schema({a: {"type": "boolean"} for a in candidates}) for key in criteria}))
    targets = {text: [marks[f][a] for f in criteria] for a, text in candidates.items()}
    return facts, [targets[text] for text in answers]
