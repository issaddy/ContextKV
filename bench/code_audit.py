"""Read-only syntax diagnostics for JSON examples and curl JSON request bodies.

Commands are never executed. Parsing does not validate an API, authentication,
field names, shell expansion, factual correctness or an unfenced code example.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex


FENCE = re.compile(r"^[ \t]*```([^\n]*)\n(.*?)^[ \t]*```[ \t]*(?:\n|$)", re.MULTILINE | re.DOTALL)
DATA_FLAGS = {"-d", "--data", "--data-raw", "--data-binary", "--json"}


def shell_commands(text):
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|")
    lexer.whitespace = " \t\r"
    current, commands = [], []
    for token in lexer:
        if token == "\n" or all(char in ";&|" for char in token):
            if current:
                commands.append(current)
                current = []
        else:
            current.append(token)
    if current:
        commands.append(current)
    return commands


def json_diagnostic(payload):
    try:
        json.loads(payload)
    except json.JSONDecodeError as error:
        hint = ("object_member_fragment" if re.match(r'^\s*"[^"\n]+"\s*:', payload)
                else "contains_comment_syntax" if re.search(r'(?m)^\s*//|"\s*//', payload)
                else "complete_payload_candidate")
        return {"status": "invalid_json", "message": error.msg,
                "line": error.lineno, "column": error.colno, "position": error.pos,
                "presentation_hint": hint}
    return {"status": "valid_json_syntax"}


def inspect_answer(answer):
    records = []
    for index, match in enumerate(FENCE.finditer(answer)):
        language, content = match.group(1).strip().lower(), match.group(2)
        record = {"block_index": index, "language": language, "start": match.start(2),
                  "end": match.end(2), "content": content, "checks": []}
        if language in {"json", "application/json"}:
            record["checks"].append({"kind": "json_block", "payload": content,
                                     **json_diagnostic(content)})
        elif language in {"bash", "sh", "shell", "console", "curl", "text", ""} and re.search(r"\bcurl\b", content):
            command = re.sub(r"\\\r?\n", " ", content)
            command = re.sub(r"(?m)^\s*\$\s+", "", command)
            try:
                commands = shell_commands(command)
            except ValueError as error:
                record["checks"].append({"kind": "curl_shell_lexing", "status": "unparsed_shell",
                                         "message": str(error)})
            else:
                curl_commands = [tokens for tokens in commands if tokens[0].split("/")[-1] == "curl"]
                for tokens in curl_commands:
                    for position, token in enumerate(tokens):
                        flag, equals, inline = token.partition("=")
                        if flag not in DATA_FLAGS:
                            continue
                        payload = inline if equals else (tokens[position+1] if position+1 < len(tokens) else None)
                        if payload is None:
                            check = {"status": "missing_data_argument"}
                        elif payload.lstrip().startswith("@"):
                            check = {"status": "external_payload_not_read"}
                        elif payload.lstrip().startswith(("{", "[")) or flag == "--json":
                            check = json_diagnostic(payload)
                        else:
                            check = {"status": "non_json_payload_not_assessed"}
                        record["checks"].append({"kind": "curl_data_argument", "flag": flag,
                                                 "payload": payload, **check})
        record["coverage"] = "static_syntax_only" if record["checks"] else "not_assessed"
        records.append(record)
    return records


def audit_record(record):
    blocks = inspect_answer(record["answer"])
    return {"task_id": record["task_id"], "conversation_id": record.get("conversation_id"),
            "answer_truncated": record.get("truncated"), "blocks": blocks,
            "invalid_json_payloads": sum(check["status"] == "invalid_json" for block in blocks for check in block["checks"]),
            "scope": "closed triple-backtick JSON or curl blocks only; no code execution or API validation",
            "commands_executed": 0, "answer_modified": False, "semantic_quality_validated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.audit).open() as source, target.open("x") as output:
        for line in source:
            if line.strip():
                output.write(json.dumps(audit_record(json.loads(line)), ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
