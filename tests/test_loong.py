"""Author-format fixtures verify input fidelity, source resolution and label isolation."""
from copy import deepcopy
from dataclasses import asdict
import json
import struct
import zipfile

import pytest

from adapters import loong
from adapters.mtrag import load_rows, online_turn
from adapters.retrieval import to_evidence
from controller.candidates import build_candidates
from controller.schema import SessionSnapshot


def archive(tmp_path, members):
    path = tmp_path/"doc.zip"
    with zipfile.ZipFile(path, "w") as output:
        for name, text in members.items():
            output.writestr("doc/"+name, text)
    return path


def example(**changes):
    return {"id": "published-id", "type": "paper", "language": "en", "level": 3,
            "set": 1, "length": 40000, "shuffle_doc": True, "doc": ["first.md", "second.md"],
            "question": "Compare these papers; retain literal {instruction}.",
            "instruction": "List their respective conditions.",
            "prompt_template": "#Papers:\n{docs}\n\n{instruction}\nQuestion: {question}",
            "answer": {"PRIVATE_REFERENCE": ["not an input fact"]}, **changes}


def test_complete_sources_order_and_structured_answers_survive_p1_boundary(tmp_path):
    originals = {"first.md": "# First paper\nA\n\nOnly under condition X.🙂\n",
                 "second.md": "# Second paper\nB\n\nExcept under condition Y.\n"}
    path = archive(tmp_path, {"paper/"+name: text for name, text in originals.items()})
    row = example(doc=["second.md", "first.md", "second.md"])
    before = deepcopy(row)
    with loong.Documents(path) as docs:
        result = loong.task(row, docs, revision="fixture-v1")
    assert row == before and result["source_task"]["doc"] == row["doc"]
    assert len(result["contexts"]) == 3
    assert [c["metadata"]["published_position"] for c in result["contexts"]] == [0, 1, 2]
    assert len({c["document_id"] for c in result["contexts"]}) == 3
    for c, name in zip(result["contexts"], row["doc"]):
        body = originals[name]
        assert c["text"] == body.split("\n")[0].strip("#").strip()+"\n"+body+"\n\n"
    assert json.loads(result["targets"][0]["text"]) == row["answer"] == result["reference_value"]
    assert row["question"] in result["input"][0]["text"]  # Braces were not re-expanded.
    turn = online_turn(result, "P1")
    visible = json.dumps(asdict(turn), ensure_ascii=False)
    assert "PRIVATE_REFERENCE" not in visible and "not an input fact" not in visible
    assert not {"reference_value", "targets", "source_task"} & asdict(turn).keys()
    blocks = tuple(to_evidence(c, "fixture-v1", corpus_id="loong") for c in turn.contexts)
    candidate, = build_candidates(SessionSnapshot(turn.conversation_id), turn.query, blocks)
    assert candidate.feasible and len(candidate.required_blocks) == 3
    rendered = "".join(message.content for message in candidate.messages)
    assert all(body in rendered for body in originals.values())
    assert rendered.index(originals["second.md"]) < rendered.index(originals["first.md"])
    assert "PRIVATE_REFERENCE" not in rendered


def test_task_without_question_placeholder_does_not_append_unused_question(tmp_path):
    path = archive(tmp_path, {"paper/first.md": "# First\nOriginal"})
    row = example(doc=["first.md"], question="Unused author field",
                  prompt_template="{instruction}\n#Papers:\n{docs}")
    with loong.Documents(path) as docs:
        result = loong.task(row, docs, revision="fixture-v1")
    assert "Unused author field" not in result["input"][0]["text"]
    assert result["source_task"]["question"] == row["question"]


@pytest.mark.parametrize("matching", [True, False])
def test_legal_fact_result_separation_and_numbered_aliases(tmp_path, matching):
    cases = {"甲案": {"content": "甲事实\n", "result": "甲裁决\n"},
             "乙案": {"content": "乙事实\n", "result": "乙裁决\n"}}
    path = archive(tmp_path, {"legal/legal.json": json.dumps(cases, ensure_ascii=False)})
    row = example(type="legal", language="zh", level=4, doc=["乙案", "甲案"],
                  question=(loong.LINK_RESULTS+"候选结果A/B" if matching else "按案件类型排序"),
                  instruction="仅根据上述看到的判决文书回答以下问题。")
    with loong.Documents(path) as docs:
        result = loong.task(row, docs, revision="fixture-v1")
    for index, (context, name) in enumerate(zip(result["contexts"], row["doc"])):
        expected = cases[name]["content"] + ("" if matching else cases[name]["result"])
        assert context["text"] == f"《判决文书{index+1}》\n"+expected+"\n\n"
        assert "?case_sha256=" in context["source"]
    if matching:
        assert "候选结果A/B" in result["input"][0]["text"]
        assert all("裁决" not in c["text"] for c in result["contexts"])


def test_legal_record_name_cannot_leak_case_taxonomy_through_provenance(tmp_path):
    name = "PRIVATE_ADMINISTRATIVE_REGISTRATION_CATEGORY"
    path = archive(tmp_path, {"legal/legal.json": json.dumps({name: {"content": "Only original facts.", "result": "Original ruling."}})})
    row = example(type="legal", language="zh", doc=[name])
    with loong.Documents(path) as docs:
        result = loong.task(row, docs, revision="fixture-v1")
    turn = online_turn(result, "P1")
    assert name not in json.dumps(asdict(turn))
    blocks = tuple(to_evidence(c, "fixture-v1", corpus_id="loong") for c in turn.contexts)
    candidate, = build_candidates(SessionSnapshot(turn.conversation_id), turn.query, blocks)
    rendered = "".join(m.content for m in candidate.messages)
    assert name not in rendered
    assert "Only original facts." in rendered and "Original ruling." in rendered
    assert result["contexts"][0]["metadata"]["original_document"] == name


def test_financial_resolution_never_guesses_first_ambiguous_file(tmp_path):
    path = archive(tmp_path, {"financial/2024-Acme-j.txt": "Quarterly figure 12",
                              "financial/2024-Acme-n.txt": "Annual figure 42",
                              "financial/report-2024-甲公司-季度报告.txt": "甲公司的原始财务内容。"})
    with loong.Documents(path) as docs:
        with pytest.raises(loong.SourceResolutionError) as error:
            loong.task(example(type="financial", doc=["Acme"]), docs, revision="fixture-v1")
        assert len(error.value.candidates) == 2
        result = loong.task(example(type="financial", doc=["甲公司"]), docs, revision="fixture-v1")
        assert result["contexts"][0]["text"] == "《季度报告》\n甲公司的原始财务内容。\n\n"


def test_published_archive_style_utf8_names_without_encoding_flag(tmp_path):
    path = archive(tmp_path, {"financial/report-2024-甲公司-季度报告.txt": "完整原文。"})
    data = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        position = data.index(signature)+flag_offset
        flags, = struct.unpack_from("<H", data, position)
        assert flags & 0x800
        struct.pack_into("<H", data, position, flags & ~0x800)
    path.write_bytes(data)
    with zipfile.ZipFile(path) as conventional:
        assert "甲公司" not in conventional.namelist()[0]
    with loong.Documents(path) as docs:
        result = loong.task(example(type="financial", doc=["甲公司"]), docs, revision="fixture-v1")
        assert "完整原文。" in result["contexts"][0]["text"]


def test_missing_document_rejects_whole_task_instead_of_dropping_a_source(tmp_path):
    path = archive(tmp_path, {"paper/first.md": "# First\nOriginal"})
    with loong.Documents(path) as docs, pytest.raises(loong.SourceResolutionError):
        loong.task(example(), docs, revision="fixture-v1")


def test_cli_records_unresolved_tasks_and_declared_sources(tmp_path):
    path = archive(tmp_path, {"paper/first.md": "# First\nOriginal"})
    rows = [example(id="complete", doc=["first.md"], answer=["First"]),
            example(id="missing", doc=["missing.md"]), example(id="unselected", set=2)]
    questions = tmp_path/"questions.jsonl"
    questions.write_text("".join(json.dumps(r)+"\n" for r in rows))
    output = tmp_path/"output"
    args = ["--questions", str(questions), "--documents", str(path), "--output", str(output), "--length-set", "1"]
    manifest = loong.main(args)
    assert manifest["requested_tasks"] == 2 and manifest["exported_tasks"] == manifest["unresolved_tasks"] == 1
    assert manifest["status"] == "partial_source_resolution" and manifest["model_calls"] == 0
    result, = load_rows(output/"tasks.jsonl")
    assert result["task_id"] == "complete" and json.loads(result["targets"][0]["text"]) == ["First"]
    issues, = json.loads((output/"source_issues.json").read_text())
    assert issues["task_id"] == "missing" and issues["candidates"] == []
    assert json.loads((output/"split_manifest.json").read_text())["test"] == ["complete"]
    questions.write_text(questions.read_text()+"\n")
    assert loong.main(args)["exported_tasks"] == 1
