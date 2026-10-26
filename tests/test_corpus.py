import json

import pytest

from adapters.corpus import CorpusIndex
from controller.candidates import build_candidates
from controller.evidence import EvidenceError
from controller.schema import SessionSnapshot


def test_corpus_conditions_headers_and_code_anchors_keep_original_text():
    rows = [
        {"_id": "heading", "text": "Parameter | limit", "metadata": {"version": "v2"}},
        {"_id": "conditions", "text": "Only enterprise accounts support this option.",
         "metadata": {"version": "v2", "header_passage_ids": ["heading"]}},
        {"_id": "row", "document_id": "api", "text": "  `connect_timeout` = 30\n",
         "source": "https://example.org/api", "title": "API settings",
         "metadata": {"version": "v2", "condition_passage_ids": ["conditions"],
                      "parent_section": "settings", "span": [20, 45]},
         "reference": "GOLD_SECRET", "answer": "ANSWER_SECRET"},
    ]
    index = CorpusIndex.from_records(rows, corpus_id="manual", snapshot_id="2026-10-23")
    selected = index.resolve(("row",))
    assert {block.key.passage_id for block in selected} == {"row", "conditions", "heading"}
    block = selected[0]
    assert block.text == rows[-1]["text"] and block.key.document_id == "api"
    assert block.parent_section == "settings" and block.span == (20, 45)
    assert block.anchors == ("connect_timeout",)
    candidate = build_candidates(SessionSnapshot("s"), "How do I configure this?", selected)[0]
    prompt = "\n".join(message.content for message in candidate.messages)
    assert "Only enterprise" in prompt and "Parameter | limit" in prompt
    assert "GOLD_SECRET" not in prompt and "ANSWER_SECRET" not in prompt


def test_reranking_context_uses_original_metadata_without_expanding_the_pool():
    from adapters.mtrag import OnlineContext
    from adapters.retrieval import rerank_document

    index = CorpusIndex.from_records([
        {"_id":"value","text":"The limit is three requests.","title":"Atlas limits",
         "url":"https://example.org/atlas/limits","dependencies":["condition"]},
        {"_id":"condition","text":"Only the enterprise plan uses this limit."},
    ],corpus_id="manuals",snapshot_id="v1")
    original = (OnlineContext("value","The limit is three requests.",score=9),
                OnlineContext("unannotated","An original candidate without metadata.",score=8))
    scoped = index.contextualize(original)
    assert len(scoped) == len(original)
    assert [(c.document_id,c.text,c.score) for c in scoped] == [(c.document_id,c.text,c.score) for c in original]
    assert scoped[0].source == "https://example.org/atlas/limits" and scoped[0].title == "Atlas limits"
    assert scoped[1] is original[1] and original[0].source == ""
    assert rerank_document(scoped[0]) == "Source: https://example.org/atlas/limits\nAtlas limits\nThe limit is three requests."
    assert rerank_document(original[0]) == "\nThe limit is three requests."
    assert {b.key.passage_id for b in index.enrich(scoped[:1])} == {"value","condition"}
    mismatched = OnlineContext("value","A different version's text.")
    assert index.contextualize((mismatched,)) == (mismatched,)
    with pytest.raises(EvidenceError,match="exact retrieved body"):
        index.enrich((mismatched,))


def test_selection_groups_preserve_version_and_condition_scope():
    from adapters.hybrid_retrieval import select_ranked_evidence
    from adapters.mtrag import OnlineContext

    rows = [{"_id":pid,"text":"Limit is three.","title":"Plan limits","version":version,
             "dependencies":dependencies,"parent_section":parent}
            for pid,version,dependencies,parent in (
                ("a","v1",[],None),("copy","v1",[],None),("revision","v2",[],None),
                ("conditional","v1",["condition"],None),("regional","v1",[],"East region"))]
    rows += [{"_id":"condition","text":"Only when offline.","version":"v1"}]
    index = CorpusIndex.from_records(rows,corpus_id="manual",snapshot_id="snapshot")
    ranked = tuple(OnlineContext(row["_id"],row["text"],title=row["title"]) for row in rows[:-1])
    selected, trace = select_ranked_evidence(ranked,group_key=index.selection_group)
    assert [c.document_id for c in selected] == ["a","revision","conditional","regional"]
    assert trace["collapsed"] == [{"passage_id":"copy","representative":"a"}]
    # Condition closure remains available to both execution candidates after selection.
    blocks = index.enrich(selected)
    assert {b.key.passage_id for b in blocks} == {"a","revision","conditional","regional","condition"}
    plan, = build_candidates(SessionSnapshot("s"),"Compare v1 and v2 limits",blocks)
    assert plan.feasible and {b.key for b in blocks} == plan.required_blocks


def test_corpus_dependency_resolves_exact_version_and_requires_explicit_ambiguity_choice():
    rows = [{"_id": passage, "version": version, "text": f"{passage} {version}",
             "dependencies": ["condition"] if passage == "value" else []}
            for version in ("v1", "v2") for passage in ("condition", "value")]
    index = CorpusIndex.from_records(rows, corpus_id="c", snapshot_id="snapshot")
    with pytest.raises(EvidenceError, match="ambiguous"):
        index.resolve(("value",))
    value = next(block for block in index.blocks
                 if block.key.passage_id == "value" and block.key.version == "v1")
    selected = index.resolve((value.key,))
    assert len(selected) == 2 and all(block.key.version == "v1" for block in selected)


def test_missing_condition_is_reported_at_corpus_boundary():
    with pytest.raises(EvidenceError, match="missing or ambiguous dependency"):
        CorpusIndex.from_records(({"_id": "row", "text": "value", "dependencies": ["missing"]},),
                                 corpus_id="c", snapshot_id="v1")


def test_enrich_preserves_retrieved_bodies_with_separate_document_and_passage_identity():
    from adapters.mtrag import OnlineContext

    index = CorpusIndex.from_records([
        {"_id": "passage", "document_id": "document", "version": "v2", "text": "original",
         "condition_passage_ids": ["condition"]},
        {"_id": "condition", "version": "v2", "text": "only if enabled"},
    ], corpus_id="manual", snapshot_id="snapshot")
    blocks = index.enrich((OnlineContext("passage", "original"),))
    assert blocks[0].key.document_id == "document" and blocks[0].key.passage_id == "passage"
    assert blocks[0].key.version == "v2" and len(blocks) == 2
    with pytest.raises(EvidenceError, match="exact retrieved body/version"):
        index.enrich((OnlineContext("passage", "different retrieval text"),))


def test_contextpilot_annotation_ranks_retrieved_passages_without_supporting_conditions():
    from adapters.mtrag import OnlineContext
    from controller.evidence import EvidencePolicy

    index = CorpusIndex.from_records([
        {"_id": "passage", "text": "original", "condition_passage_ids": ["condition"]},
        {"_id": "condition", "text": "only if enabled"},
    ], corpus_id="manual", snapshot_id="snapshot")
    keys = index.retrieved_keys((OnlineContext("passage", "original"),
                                OnlineContext("passage", "original")))
    assert len(keys) == 1
    blocks = index.resolve(keys)
    candidate = build_candidates(SessionSnapshot("s"), "q", blocks,
                                 EvidencePolicy(layout="contextpilot"), retrieval_priority=keys)[0]
    refs = {key: ref for ref, key in candidate.reference_map}
    annotation = next(line for line in candidate.messages[-1].content.splitlines()
                      if line.startswith("Retrieval priority"))
    assert refs[keys[0]] in annotation
    condition = next(block for block in blocks if block.key.passage_id == "condition")
    assert refs[condition.key] not in annotation
    assert condition.key in candidate.required_blocks <= candidate.visible_blocks


def test_duplicate_passage_version_is_not_silently_overwritten():
    with pytest.raises(EvidenceError, match="duplicate corpus passage/version"):
        CorpusIndex.from_records(({"_id": "same", "text": text} for text in ("one", "two")),
                                 corpus_id="c", snapshot_id="v1")


def test_source_windows_keep_originals_and_do_not_cross_source_version_or_section():
    from adapters.mtrag import OnlineContext

    rows = [{"_id": str(i), "document_id": "manual", "text": f"Original {i}",
             "span": [i*10, i*10+10], "version": "v1", "url": "https://example.org/manual"}
            for i in range(5)]
    rows += [{**rows[2], "_id": "other-"+field, field: value}
             for field, value in (("version", "v2"), ("url", "https://mirror.org/manual"),
                                  ("parent_section", "another"), ("document_id", "different"))]
    index = CorpusIndex.from_records(reversed(rows), corpus_id="c", snapshot_id="v1")
    hits = (OnlineContext("2", "Original 2", score=9.0), OnlineContext("3", "Original 3", score=8.0))
    support = index.source_window(hits)
    assert [(c.document_id, c.text, c.score) for c in support] == [("1", "Original 1", None), ("4", "Original 4", None)]
    assert "0" not in {c.document_id for c in support}  # One hop, not transitive closure.
    candidates = build_candidates(SessionSnapshot("s"), "q", index.enrich((*hits, *support)),
                                  retrieval_priority=index.retrieved_keys(hits))
    assert len(candidates[0].required_blocks) == 4
    assert candidates[0].required_blocks <= candidates[0].visible_blocks


def test_mtrag_windows_use_offsets_only_for_order_not_for_text_stitching():
    from adapters.mtrag import OnlineContext

    rows = [{"_id": pid, "text": body, "url": "https://example.org/manual"}
            for pid, body in (("doc-0-9", "LEFT"), ("doc-5-14", "MIDDLE"),
                              ("doc-10-19", "RIGHT"), ("other-6-9", "OTHER"))]
    index = CorpusIndex.from_records(rows, corpus_id="mtrag", snapshot_id="mtrag-fixed")
    support = index.source_window((OnlineContext("doc-5-14", "MIDDLE"),))
    assert [c.text for c in support] == ["LEFT", "RIGHT"]
    assert [b.key.document_id for b in index.enrich(support)] == ["doc-0-9", "doc-10-19"]
    # Unknown schemas do not turn arbitrary numeric suffixes into source relations.
    other = CorpusIndex.from_records(rows, corpus_id="other", snapshot_id="snapshot")
    assert other.source_window((OnlineContext("doc-5-14", "MIDDLE"),)) == ()


def test_corpus_roundtrip_jsonl_and_dependency_cycle(tmp_path):
    rows = [{"passage_id": name, "text": name, "dependencies": [other]}
            for name, other in (("a", "b"), ("b", "a"))]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    index = CorpusIndex.from_jsonl(path, corpus_id="c", snapshot_id="v1")
    assert {block.text for block in index.resolve(("a",))} == {"a", "b"}


@pytest.mark.asyncio
async def test_corpus_metadata_survives_service_commit_and_revisit(tmp_path):
    from controller.mock_engine import MockEngine
    from controller.planner import PlannerKnobs
    from controller.service import Controller
    from controller.session_store import SessionStore

    index = CorpusIndex.from_records([
        {"_id": "condition", "text": "The condition is an enterprise plan."},
        {"_id": "row", "text": "`max_retries` is 3", "condition_passage_ids": ["condition"]},
        {"_id": "other", "text": "Avatars accept PNG files."},
    ], corpus_id="manual", snapshot_id="v1")
    store = SessionStore(tmp_path / "state.sqlite3")
    controller = Controller(store, MockEngine(), knobs=PlannerKnobs(strategy="always_rebuild"))
    try:
        await controller.serve_turn("s", "What is max_retries?", index.resolve(("row",)))
        await controller.serve_turn("s", "How do avatars work?", index.resolve(("other",)))
        await controller.serve_turn("s", "What about that condition?", index.resolve(("other",)))
        snapshot = store.read_snapshot("s")
        condition = index.resolve(("condition",))[0]
        assert condition.key in snapshot.execution.visible_blocks
        assert condition.text in snapshot.execution.messages[-2].content
        assert len(snapshot.raw_history) == 3
    finally:
        await controller.aclose()
        store.close()
