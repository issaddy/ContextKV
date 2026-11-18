"""Change only which equivalent source is resident, then measure membership and actual prefill."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from adapters.shared_encoding import SharedEvidenceEncoder, SharedQueryEncoder
from bench.cache_reuse import receipt
from controller.config import apply_study_config, create_model_template, load_config, load_kv_policy
from controller.cooperative_request import ANSWER_SCOPE, EvidenceCosts, compile_request, source_address
from controller.engine_client import SGLangEngineClient
from controller.evidence import make_evidence
from controller.sampling import SamplingRecipe
from controller.schema import EvidenceUnit, SourceSpan


async def compare(config):
    study = yaml.safe_load(Path(config).read_text())
    run = apply_study_config(load_config(study["generation"]["run_config"]), study)
    template = create_model_template(run.model)
    policy = load_kv_policy(run.cooperative)
    question = "How many days does a member have to request a replacement card? Give the number of days."
    body = ("Members have 19 days to request a replacement card.\n"
            + "Keep the original membership number and contact the service office for a replacement.\n" * 25)
    # Both manuals publish the same section. Titles distinguish their compiled Native prefixes.
    blocks = [make_evidence(body, key, title=title) for key, title in
              (("a", "Membership manual"), ("b", "Office manual"))]
    size = len(template.tokenizer.encode(body, add_special_tokens=False))
    units = tuple(EvidenceUnit(block.key.document_id,
        (SourceSpan(source_address(block), block.key, 0, len(body)),), body, "Replacement cards",
        body_tokens=size) for block in blocks)
    query_encoder = SharedQueryEncoder(**study["shared"]["query"])
    encoder = SharedEvidenceEncoder(**study["shared"]["evidence"])
    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=SamplingRecipe())
    results = []
    try:
        query = query_encoder.encode((), question, "replacement", 1)
        # Separate expansions preserve identical model inputs and avoid list-position effects.
        evidence = encoder.encode(query, units[:1]).merge(encoder.encode(query, units[1:]))
        fixed = compile_request(template, question, (), (), instruction=ANSWER_SCOPE)
        for block in blocks:
            for visible in (False, True):
                await engine.profile_prepare()
                args = dict(session_id="replacement", cache_scope="replacement", cache_namespace="replacement")
                warm = compile_request(template, "Read the manual.", (), (block,))
                seed = await engine.compute(warm, request_id="replacement:seed", max_new_tokens=1,
                                            mode="exact", **args, **policy)
                snapshot = await engine.evidence_state([source_address(b) for b in blocks],
                    cache_scope=args["cache_scope"], cache_namespace=args["cache_namespace"])
                if not visible:
                    snapshot = {**snapshot, "sources": {}, "prefixes": []}
                prices = EvidenceCosts(units, blocks, template.tokenizer, snapshot,
                    prompt_tokens=len(fixed.input_ids), output_tokens=run.context.max_new_tokens, policy=policy)
                chosen = evidence.select(units, body_tokens=size, cost_model=prices,
                    seconds_price=run.cooperative["prices"]["seconds"],
                    need_ids=tuple(n.need_id for n in query.task.needs))
                selected = tuple(b for b in blocks if b.key.document_id in chosen)
                request = compile_request(template, question, (), selected, instruction=ANSWER_SCOPE)
                answer = await engine.compute(request, request_id="replacement:answer",
                    max_new_tokens=run.context.max_new_tokens, mode="auto", **args, **policy)
                results.append(dict(warm=block.key.document_id, visible=visible, selected=chosen,
                    new_frontier=prices.changed_members, seed=receipt(seed), answer=answer["text"],
                    execution=receipt(answer)))
    finally:
        await engine.aclose()
        query_encoder.close()
        encoder.close()
    return dict(question=question, reference="19 days", one_section_budget=size,
                relevance=evidence.relevance.tolist(), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(asyncio.run(compare(args.config)), indent=2)+"\n")
