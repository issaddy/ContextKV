from hashlib import sha256
import json

import pytest

from adapters.api_evidence import build_bundle, extract_sdk_records, main
from adapters.corpus import CorpusIndex
from controller.evidence import EvidenceError


def spec(symbols, **overrides):
    return {"repository": "author/sdk", "commit": "a" * 40, "tag": "v1.2",
            "package": "sdk-python", "package_version": "1.2", "path": "sdk/client.py",
            "symbols": symbols, **overrides}


def extract(tmp_path, text, source_spec):
    path = tmp_path / "client.py"
    path.write_bytes(text.encode("utf-8"))
    return extract_sdk_records(path, source_spec)


def test_decorators_async_unicode_crlf_exact_spans_no_execution(tmp_path):
    text = ('# Copyright Sample\r\nraise RuntimeError("NEVER EXECUTE")\r\n'
            '@register("日本語")\r\nasync def café(名前: str, *, count: int = 2) -> str:\r\n'
            '    """原文の説明。"""\r\n    return 名前\r\n')
    records, manifest = extract(tmp_path, text, spec(["café"]))
    row = records[0]
    assert row["text"] == text[text.index("@register"):]
    assert text[slice(*row["span"])] == row["text"]
    details = row["api_source"]
    assert details["signature"]["text"] == 'async def café(名前: str, *, count: int = 2) -> str:'
    assert details["docstring"]["text"] == '"""原文の説明。"""'
    for fragment in (details["signature"], details["docstring"]):
        assert text[slice(*fragment["span"])] == fragment["text"]
    assert details["required_declared_parameters"] == ["名前"]
    assert manifest["original_source_header"] == "# Copyright Sample\r\n"
    assert manifest["code_executed"] is False
    assert manifest["repository_revision_verified_by_adapter"] is False


def test_signature_lambda_colons_and_missing_docstring(tmp_path):
    text = ('class Response(\n    ExternalBase,\n):\n    pass\n\n'
            'def request(value=lambda x: {"a": x}):\n    return value\n')
    records, _ = extract(tmp_path, text, spec(["Response", "request"]))
    cls, function = records
    assert cls["text"] == 'class Response(\n    ExternalBase,\n):'
    assert cls["api_source"]["line_end"] == 3
    assert cls["api_source"]["inheritance_status"] == "not_resolved"
    assert cls["api_source"]["docstring_status"] == "absent"
    assert function["api_source"]["signature"]["text"] == 'def request(value=lambda x: {"a": x}):'


def test_annotations_explicit_dependencies_and_corpus_closure(tmp_path):
    text = ('class Result:\n    """Result fields."""\n\n'
            'class Metadata:\n    """Not a type dependency."""\n\n'
            'def encode():\n    return 1\n\n'
            'def send(value: Union["Result", bytes], flag: Literal["Metadata"])'
            ' -> Annotated[Result, "Metadata"]:\n    return value\n')
    rows, manifest = extract(tmp_path, text, spec(["send"], dependencies={"Result": ["encode"]}))
    assert [r["api_source"]["symbol"] for r in rows] == ["send", "Result", "encode"]
    assert manifest["declarations"]["send"]["annotation_dependency_symbols"] == ["Result"]
    index = CorpusIndex.from_records(rows, corpus_id="sdk", snapshot_id="frozen")
    blocks = index.resolve((rows[0]["passage_id"],))
    assert len(blocks) == 3 and all(b.key.version == "1.2@" + "a" * 40 for b in blocks)
    assert all(b.text == text[slice(*b.span)] for b in blocks)


def test_literal_http_and_unknown_dynamic_or_reassigned_path(tmp_path):
    text = ('class Client:\n'
            '    def literal(self):\n'
            '        url = "/v1/items"\n'
            '        return self.prepare_request(method="GET", url=url)\n'
            '    def dynamic(self, item):\n'
            '        url = "/v1/items/{id}".format(id=item)\n'
            '        return self.prepare_request(method="GET", url=url)\n'
            '    def reassigned(self, flag):\n'
            '        url = "/v1/items"\n'
            '        if flag:\n'
            '            url = other_path()\n'
            '        return self.prepare_request(method="GET", url=url)\n'
            '    def nested(self):\n'
            '        def inner():\n'
            '            return self.prepare_request(method="GET", url="/not_outer")\n'
            '        return inner\n'
            '    @setup.prepare_request(method="POST", url="/decorator")\n'
            '    def declared(self, value=setup.prepare_request(method="GET", url="/default")):\n'
            '        return value\n')
    _, manifest = extract(tmp_path, text, spec(["Client." + s for s in
                                               ("literal", "dynamic", "reassigned", "nested", "declared")]))
    methods = manifest["declarations"]
    literal = methods["Client.literal"]["http_calls"][0]
    assert literal["method"]["value"] == "GET" and literal["path"]["value"] == "/v1/items"
    for name in ("dynamic", "reassigned"):
        item = methods["Client." + name]["http_calls"][0]
        assert item["path"]["status"] == "unknown" and item["path"]["value"] is None
    assert methods["Client.nested"]["http_status"] == "unknown_no_recognized_request_builder"
    assert methods["Client.declared"]["http_status"] == "unknown_no_recognized_request_builder"


def test_inheritance_and_overloads_not_guessed(tmp_path):
    text = ('class Parent:\n    def inherited(self):\n        pass\n'
            'class Child(Parent):\n    pass\n'
            '@overload\ndef choose(value: int):\n    ...\n'
            'def choose(value):\n    return value\n')
    rows, manifest = extract(tmp_path, text, spec(["Child.inherited", "choose", "Parent.inherited"]))
    assert len(rows) == 1 and rows[0]["api_source"]["symbol"] == "Parent.inherited"
    assert manifest["status"] == "incomplete_unknown_symbols"
    assert [r["reason"] for r in manifest["unknown_symbols"]] == [
        "not_declared_in_file_inheritance_and_imports_not_resolved", "ambiguous_multiple_declarations"]


def test_http_scope_excludes_decorator_defaults_annotations_and_nested_scopes(tmp_path):
    text = ('@wrap(client.prepare_request(method="GET", url="/decorator"))\n'
            'def operation(\n'
            '    value: client.prepare_request(method="GET", url="/parameter_annotation")\n'
            '    = client.prepare_request(method="GET", url="/default"),\n'
            ') -> client.prepare_request(method="GET", url="/return_annotation"):\n'
            '    class Nested:\n'
            '        request = client.prepare_request(method="GET", url="/nested_class")\n'
            '    deferred = lambda: client.prepare_request(method="GET", url="/lambda")\n'
            '    def inner():\n'
            '        return client.prepare_request(method="GET", url="/nested_function")\n'
            '    return client.prepare_request(method="POST", url="/actual_body")\n')
    rows, manifest = extract(tmp_path, text, spec(["operation"]))
    calls = rows[0]["api_source"]["http_calls"]
    assert len(calls) == 1
    assert calls[0]["method"]["value"] == "POST"
    assert calls[0]["path"]["value"] == "/actual_body"
    assert calls[0]["call"]["text"] == 'client.prepare_request(method="POST", url="/actual_body")'
    assert rows[0]["text"] == text and manifest["code_executed"] is False


def test_versions_never_merge_even_with_identical_source(tmp_path):
    text = 'class Result:\n    """value"""\ndef send() -> Result:\n    pass\n'
    old, _ = extract(tmp_path, text, spec(["send"]))
    new, _ = extract(tmp_path, text, spec(["send"], repository="fork/sdk", commit="b" * 40,
                                        tag="v2", package_version="2"))
    index = CorpusIndex.from_records(old + new, corpus_id="sdk", snapshot_id="both")
    with pytest.raises(EvidenceError, match="ambiguous"):
        index.resolve((old[0]["passage_id"],))
    old_key = next(b.key for b in index.blocks if b.key.version.startswith("1.2@")
                   and b.key.passage_id == old[0]["passage_id"])
    assert len(index.resolve((old_key,))) == 2
    assert all(b.key.version == old_key.version for b in index.resolve((old_key,)))
    assert all(b.key.document_id.startswith("author/sdk/") for b in index.resolve((old_key,)))


def test_same_version_symbol_in_different_repository_cannot_cross_bind(tmp_path):
    text = 'class Result:\n    """value"""\ndef send() -> Result:\n    pass\n'
    original, _ = extract(tmp_path, text, spec(["send"]))
    fork, _ = extract(tmp_path, text, spec(["send"], repository="fork/sdk"))
    with pytest.raises(EvidenceError, match="missing or ambiguous dependency"):
        CorpusIndex.from_records(original + fork, corpus_id="sdk", snapshot_id="both")


def test_unselected_dependency_owner_preserved_not_claimed_validated(tmp_path):
    rows, manifest = extract(tmp_path, 'def send():\n    pass\n',
                             spec(["send"], dependencies={"Unused": ["Missing"]}))
    assert len(rows) == 1 and manifest["status"] == "complete"
    assert manifest["declared_dependencies"] == {"Unused": ["Missing"]}
    assert manifest["unapplied_dependency_owners"] == ["Unused"]
    assert "targets_not_validated" in manifest["dependency_spec_scope"]


@pytest.mark.parametrize("override", [
    {"commit": "main"}, {"package_version": ""}, {"symbols": ["_private"]},
    {"path": "../other.py"}, {"symbols": ["send", "send"]},
])
def test_explicit_identity_and_public_symbols_required(tmp_path, override):
    with pytest.raises(ValueError):
        extract(tmp_path, "def send():\n    pass\n", {**spec(["send"]), **override})


def test_expected_source_bytes_checked_only_at_offline_boundary(tmp_path):
    text = "def send():\n    pass\n"
    correct = sha256(text.encode()).hexdigest()
    _, manifest = extract(tmp_path, text, spec(["send"], expected_source_sha256=correct))
    assert manifest["identity_status"] == "matches_supplied_expected_source_sha256"
    with pytest.raises(ValueError, match="expected_source_sha256"):
        extract(tmp_path, text, spec(["send"], expected_source_sha256="0" * 64))


def bundle_spec(tmp_path, *, symbols=None, dependencies=None):
    (tmp_path / "client.py").write_bytes(b'# Source copyright\ndef send():\n    """Original."""\n    pass\n')
    (tmp_path / "LICENSE").write_bytes(b"Original license\r\n")
    source = spec(symbols or ["send"], source_file="client.py", license_file="LICENSE",
                  license_id="test-license", dependencies=dependencies or {})
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"corpus_id": "sdk", "snapshot_id": "explicit-snapshot", "sources": [source]}))
    return path


def test_cli_roundtrip_preserves_license_and_refuses_overwrite(tmp_path):
    source = bundle_spec(tmp_path)
    output = tmp_path / "out"
    assert main(["--spec", str(source), "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["record_count"] == 1
    assert (output / "licenses/1-LICENSE.txt").read_bytes() == b"Original license\r\n"
    assert manifest["sources"][0]["original_source_header"] == "# Source copyright\n"
    index = CorpusIndex.from_jsonl(output / "records.jsonl", corpus_id="sdk", snapshot_id="explicit-snapshot")
    assert index.blocks[0].text == 'def send():\n    """Original."""\n    pass\n'
    with pytest.raises(FileExistsError):
        build_bundle(source, output)


def test_unknown_requested_symbol_kept_in_manifest_with_nonzero_status(tmp_path):
    source = bundle_spec(tmp_path, symbols=["send", "Client.inherited"])
    output = tmp_path / "out"
    assert main(["--spec", str(source), "--output", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["record_count"] == 1
    assert manifest["sources"][0]["unknown_symbols"][0]["symbol"] == "Client.inherited"


def test_missing_explicit_dependency_cannot_silently_publish_complete_bundle(tmp_path):
    source = bundle_spec(tmp_path, dependencies={"send": ["Missing"]})
    with pytest.raises(EvidenceError, match="missing or ambiguous dependency"):
        build_bundle(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
