"""Extract versioned SDK evidence from local Python source without executing it.

This is an offline corpus builder. Source/package versions are explicit input,
and extracted text remains a contiguous span of the decoded original file.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path, PurePosixPath
import re
import token
import tokenize
from typing import Any, Mapping

from adapters.corpus import CorpusIndex


def _public_symbol(value: str) -> bool:
    parts = value.split(".")
    # An explicitly requested public class initializer is part of its API use
    # contract. Other private/dunder declarations remain outside this builder.
    if len(parts) > 1 and parts[-1] == "__init__":
        parts = parts[:-1]
    return bool(value) and all(part.isidentifier() and not part.startswith("_")
                               for part in parts)


def _validate_spec(spec: Mapping[str, Any]) -> None:
    for name in ("repository", "commit", "tag", "package", "package_version", "path"):
        if not isinstance(spec.get(name), str) or not spec[name].strip():
            raise ValueError(f"source_spec requires explicit {name}")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", spec["repository"]):
        raise ValueError("repository must be owner/repository")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", spec["commit"]):
        raise ValueError("commit must be a fixed full Git commit ID")
    path = PurePosixPath(spec["path"])
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ValueError("path must be a repository-relative Python source path")
    symbols = spec.get("symbols")
    if not isinstance(symbols, (list, tuple)) or not symbols or any(
            not isinstance(s, str) or not _public_symbol(s) for s in symbols):
        raise ValueError("symbols must name public declarations")
    if len(symbols) != len(set(symbols)):
        raise ValueError("symbols must not contain duplicates")
    dependencies = spec.get("dependencies", {})
    if not isinstance(dependencies, Mapping):
        raise ValueError("dependencies must map public symbols to explicit local symbols")
    for owner, targets in dependencies.items():
        if not _public_symbol(owner) or not isinstance(targets, (list, tuple)) or any(
                not isinstance(s, str) or not _public_symbol(s) for s in targets):
            raise ValueError("dependencies must map public symbols to explicit local symbols")


class _Source:
    def __init__(self, data: bytes):
        self.encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        self.text = data.decode(self.encoding)
        self.lines = self.text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))
        self.tokens = tuple(tokenize.generate_tokens(io.StringIO(self.text).readline))

    def offset(self, line: int, column: int, *, ast_bytes: bool = False) -> int:
        if ast_bytes:
            column = len(self.lines[line - 1].encode("utf-8")[:column].decode("utf-8"))
        return self.starts[line - 1] + column

    def fragment(self, node: ast.AST) -> dict[str, Any]:
        start = self.offset(node.lineno, node.col_offset, ast_bytes=True)
        end = self.offset(node.end_lineno, node.end_col_offset, ast_bytes=True)
        return {"span": [start, end], "text": self.text[start:end]}

    def signature(self, node: ast.AST) -> dict[str, Any]:
        start = self.offset(node.lineno, node.col_offset, ast_bytes=True)
        depth = 0
        for item in self.tokens:
            if self.offset(*item.start) < start:
                continue
            if item.type == token.OP:
                if item.string == ":" and depth == 0:
                    end = self.offset(*item.end)
                    return {"span": [start, end], "text": self.text[start:end]}
                depth += (item.string in "([{") - (item.string in ")]}")
        raise ValueError("declaration header has no suite colon")


def _declarations(tree: ast.Module) -> dict[str, list[ast.AST]]:
    result: dict[str, list[ast.AST]] = {}

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                result.setdefault(name, []).append(node)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + ".")

    visit(tree.body)
    return result


def _annotation_names(node: ast.AST) -> set[str]:
    """Only type-expression references, excluding Literal/Annotated metadata."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _annotation_names(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {ast.unparse(node)}
    if isinstance(node, ast.Subscript):
        base = ast.unparse(node.value).split(".")[-1]
        if base == "Literal":
            return set()
        value = node.slice
        if base == "Annotated" and isinstance(value, ast.Tuple):
            value = value.elts[0]
        return _annotation_names(value)
    result = set()
    for child in ast.iter_child_nodes(node):
        result.update(_annotation_names(child))
    return result


def _annotations(node: ast.AST) -> list[ast.AST]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = node.args
    params = args.posonlyargs + args.args + args.kwonlyargs
    params += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    return [arg.annotation for arg in params if arg.annotation is not None] + (
        [node.returns] if node.returns is not None else [])


def _scope_nodes(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from _scope_nodes(child)


def _http_calls(node: ast.AST, source: _Source) -> list[dict[str, Any]]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    scoped = tuple(child for statement in node.body
                   if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   for child in (statement, *_scope_nodes(statement)))
    assignments = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = statement

    def literal(expr, call):
        original = source.fragment(expr) if expr is not None else None
        if isinstance(expr, ast.Name):
            writes = [n for n in scoped if isinstance(n, ast.Name)
                      and isinstance(n.ctx, ast.Store) and n.id == expr.id]
            assignment = assignments.get(expr.id)
            if len(writes) == 1 and assignment is not None and assignment.lineno < call.lineno:
                expr = assignment.value
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {"status": "literal", "value": expr.value, "expression": original,
                    "literal_source": source.fragment(expr)}
        return {"status": "unknown", "value": None, "expression": original,
                "reason": "not_a_direct_literal_or_single_unconditional_literal_assignment"}

    result = []
    for call in scoped:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "prepare_request":
            continue
        kwargs = {item.arg: item.value for item in call.keywords if item.arg is not None}
        result.append({"call": source.fragment(call),
                       "method": literal(kwargs.get("method"), call),
                       "path": literal(kwargs.get("url"), call),
                       "scope": "syntactic_prepare_request_call_not_transport_execution"})
    return result


def extract_sdk_records(source_path: str | Path, source_spec: Mapping[str, Any]
                        ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Return ordinary CorpusIndex records and a local-source provenance manifest.

    Dependencies are same-file exact declarations, from explicit source_spec or
    original annotations. Imports, inheritance, decorators and SDK code are not run.
    """
    _validate_spec(source_spec)
    data = Path(source_path).read_bytes()
    source = _Source(data)
    tree = ast.parse(source.text, filename=str(source_path))
    declarations = _declarations(tree)
    explicit = source_spec.get("dependencies", {})
    records, symbols, unknown = [], {}, []
    pending = list(source_spec["symbols"])
    visited = set()
    prefix = f"{source_spec['package']}:{source_spec['path']}#"
    source_uri = (f"https://github.com/{source_spec['repository']}/blob/"
                  f"{source_spec['commit']}/{source_spec['path']}")
    version = f"{source_spec['package_version']}@{source_spec['commit']}"
    while pending:
        symbol = pending.pop(0)
        if symbol in visited:
            continue
        visited.add(symbol)
        matches = declarations.get(symbol, ())
        if len(matches) != 1:
            unknown.append({"symbol": symbol, "status": "unknown", "reason":
                "ambiguous_multiple_declarations" if matches else
                "not_declared_in_file_inheritance_and_imports_not_resolved"})
            continue
        node = matches[0]
        if symbol.endswith(".__init__"):
            owners = declarations.get(symbol.rpartition(".")[0], ())
            if (len(owners) != 1 or not isinstance(owners[0], ast.ClassDef)
                    or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                unknown.append({"symbol": symbol, "status": "unknown",
                                "reason": "initializer_not_unique_class_method"})
                continue
        signature = source.signature(node)
        first = node.body[0]
        has_doc = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                   and isinstance(first.value.value, str))
        doc = source.fragment(first) if has_doc else None
        start_line = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end_line = node.end_lineno
        end = source.starts[end_line]
        if isinstance(node, ast.ClassDef):
            end_line = first.end_lineno if has_doc else source.text[:signature["span"][1]].count("\n") + 1
            end = source.starts[end_line] if has_doc else signature["span"][1]
        start = source.starts[start_line - 1]
        annotations = _annotations(node)
        names = set().union(*(_annotation_names(a) for a in annotations)) if annotations else set()
        automatic = sorted(name for name in names if name in declarations and _public_symbol(name)
                           and name != symbol)
        dependencies = list(dict.fromkeys([*explicit.get(symbol, ()), *automatic]))
        pending.extend(dependencies)
        http = _http_calls(node, source)
        details = {"symbol": symbol, "kind": type(node).__name__, "line_start": start_line,
            "line_end": end_line, "signature": signature, "docstring": doc,
            "docstring_status": "present" if has_doc else "absent",
            "selection": "requested" if symbol in source_spec["symbols"] else "dependency",
            "explicit_dependency_symbols": list(explicit.get(symbol, ())),
            "annotation_dependency_symbols": automatic,
            "unresolved_annotation_symbols": sorted(names - set(automatic) - {symbol}),
            "http_calls": http,
            "http_status": "recognized_syntax" if http else "unknown_no_recognized_request_builder",
            "runtime_behavior": "unknown_not_executed"}
        if isinstance(node, ast.ClassDef):
            details["bases"] = [source.fragment(base) for base in node.bases]
            details["inheritance_status"] = "not_resolved"
        else:
            if symbol.endswith(".__init__"):
                details["initializer_of"] = symbol.rpartition(".")[0]
            positional = node.args.posonlyargs + node.args.args
            required = positional[:len(positional) - len(node.args.defaults)]
            required += [arg for arg, default in zip(node.args.kwonlyargs,
                                                     node.args.kw_defaults, strict=True)
                         if default is None]
            details["required_declared_parameters"] = [arg.arg for arg in required]
            details["extra_keyword_parameter"] = node.args.kwarg.arg if node.args.kwarg else None
        records.append({"passage_id": prefix + symbol,
            "document_id": f"{source_spec['repository']}/{source_spec['path']}",
            "version": version, "text": source.text[start:end], "span": [start, end],
            "source_uri": f"{source_uri}#L{start_line}",
            "title": f"{source_spec['package']} {source_spec['package_version']} — {symbol}",
            "parent_section": symbol.rpartition(".")[0] or None,
            "anchors": [symbol, source_spec["package"]],
            "provenance": [source_uri, f"sdk_package:{source_spec['package']}",
                f"sdk_version:{source_spec['package_version']}", f"git_commit:{source_spec['commit']}",
                f"git_tag_declared:{source_spec['tag']}", "source_expansion:not_original_mtrag_material"],
            "dependencies": [prefix + dep for dep in dependencies], "api_source": details})
        symbols[symbol] = details
    manifest = {key: source_spec[key] for key in
                ("repository", "commit", "tag", "package", "package_version", "path")}
    manifest.update(local_source_path=str(Path(source_path).resolve()), encoding=source.encoding,
        span_unit="decoded_source_unicode_characters_half_open", requested_symbols=list(source_spec["symbols"]),
        declarations=symbols, unknown_symbols=unknown,
        declared_dependencies={owner: list(targets) for owner, targets in explicit.items()},
        unapplied_dependency_owners=sorted(set(explicit) - set(symbols)),
        dependency_spec_scope="only_reached_owners_applied_unselected_owner_targets_not_validated",
        original_source_header=source.text[:source.starts[tree.body[0].lineno - 1]] if tree.body else source.text,
        status="complete" if not unknown else "incomplete_unknown_symbols")
    return tuple(records), manifest


def build_bundle(spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build a new local evidence directory, leaving all existing corpora unchanged."""
    spec_path, output_dir = Path(spec_path), Path(output_dir)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not spec.get("corpus_id") or not spec.get("snapshot_id") or not spec.get("sources"):
        raise ValueError("bundle requires corpus_id, snapshot_id and sources")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    records, sources, licenses = [], [], []
    for source_spec in spec["sources"]:
        source_path = spec_path.parent / source_spec["source_file"]
        license_path = spec_path.parent / source_spec["license_file"]
        if not source_spec.get("license_id"):
            raise ValueError("source requires an explicit license_id and local license_file")
        extracted, provenance = extract_sdk_records(source_path, source_spec)
        license_bytes = license_path.read_bytes()
        license_name = f"licenses/{len(sources) + 1}-LICENSE.txt"
        provenance["license"] = {"declared_id": source_spec["license_id"],
            "source_file": str(license_path.resolve()), "artifact_file": license_name,
            "classification": "source_spec_declaration"}
        records.extend(extracted)
        sources.append(provenance)
        licenses.append((license_name, license_bytes))
    # Existing version-specific dependency validation and closure, no new graph store.
    index = CorpusIndex.from_records(records, corpus_id=spec["corpus_id"], snapshot_id=spec["snapshot_id"])
    for block in index.blocks:
        index.resolve((block.key,))
    manifest = {"format": "contextkv_api_evidence_v1", "corpus_id": spec["corpus_id"],
        "snapshot_id": spec["snapshot_id"], "record_count": len(records), "sources": sources,
        "source_spec": spec, "status": "complete" if all(s["status"] == "complete" for s in sources)
            else "incomplete_unknown_symbols",
        "scope": "additional_versioned_evidence_not_replacement_of_original_corpus"}
    output_dir.mkdir(parents=True)
    (output_dir / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                    for row in records), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    for name, data in licenses:
        target = output_dir / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Local JSON bundle/source specification")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args(argv)
    manifest = build_bundle(args.spec, args.output)
    print(json.dumps({"status": manifest["status"], "records": manifest["record_count"],
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
