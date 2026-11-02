"""Freeze the actual tokenizer's required-claim-wire capacity bound."""
import argparse
import hashlib
from importlib.metadata import version
import json

from controller.claim_capacity import build_claim_capacity, claim_count_ceiling
from controller.config import create_model_template, load_config
from controller.context_schema import EvidenceKey, EvidenceUnit, ReadTask, SourceSpan
from controller.engine_client import SGLANG_COMMIT
from controller.reading import output_schema
from profiling.retention_trace import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    template = create_model_template(load_config(args.config).model)
    # This one-character source qualifies the literal-key grammar only.  It is
    # never submitted to the reader or included in QA/training observations.
    span = SourceSpan("wire-span", EvidenceKey("grammar", "wire", "wire", "qualification", "x"), 0, 1)
    schema = output_schema(ReadTask("wire", "READ", ("need",)),
                           (EvidenceUnit("wire-unit", (span,), "x"),))
    required = schema["properties"]["claims"]["items"]["required"]
    bound = build_claim_capacity(template.tokenizer, required, tokenizer_hash=template.fingerprint.tokenizer_hash)
    import xgrammar
    grammar = str(xgrammar.Grammar.from_json_schema(json.dumps(schema)))
    fixed = [json.dumps(key, ensure_ascii=True) for key in required]
    if not all(literal in grammar.replace('\\"', '"') for literal in fixed):
        raise ValueError("installed native JSON grammar did not expose the required fixed key literals")
    bound["grammar_contract"] = dict(backend="xgrammar", package_version=version("xgrammar"),
        engine_commit=SGLANG_COMMIT, protocol_format_version=2,
        required_schema=schema, compiled_ebnf=grammar,
        source_scope="one-character grammar qualification placeholder; no factual/QA observation",
        compiled_ebnf_sha256=hashlib.sha256(grammar.encode()).hexdigest(),
        compiler_call="Grammar.from_json_schema(json.dumps(schema)) with required fields/default fixed-order JSON mode")
    write_json(args.output, bound)
    print(json.dumps({"literal_characters": bound["required_literal_characters_per_claim"],
        "max_covered_per_token": bound["max_token_literal_coverage"],
        "claims_for_24x4096": claim_count_ceiling((4096,)*24, bound), "output": args.output}))
    return bound


if __name__ == "__main__":
    main()
