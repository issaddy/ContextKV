"""Small real task/support/search/READ integration over a controlled corpus."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from adapters.hybrid_retrieval import LocalBGE
from adapters.retrieval import BM25SRetriever, load_beir_corpus
from adapters.study_retrieval import SupportCoverageUtility, create_study_retriever
from adapters.teacher_support import TeacherSupportBackend
from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from controller.config import create_controller, load_config
from controller.serialization import to_jsonable


class SupportTeacherCoverage(TeacherSupportBackend):
    """Actual single/group support teacher and the published B3 coverage utility."""
    def predict_set(self, state, units, supports, *, context=None):
        return SupportCoverageUtility().predict_set(state, units, supports)


async def run(config_path, output_dir, *, reader_url, auxiliary_url,
              history_model, reader_gpu=0, auxiliary_gpu=0):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    config = replace(config, engine=replace(config.engine, base_url=reader_url),
                     state_dir=str(output/"state"))
    service = create_controller(config)
    native = NativePoolSampler(reader_url, gpu_capacity=config.engine.gpu_capacity_tokens,
        host_enabled=config.engine.host_capacity_tokens > 0, model=config.model.name, interval=.2)
    await native.start()
    ports = tuple({int(url.rsplit(":", 1)[1]) for url in (reader_url, auxiliary_url)})
    physical = StudyResourceSampler(tuple({reader_gpu, auxiliary_gpu}), listening_processes(ports))
    await physical.start()
    start = time.monotonic()
    corpus = output/"corpus.jsonl"
    documents = [
        ("windows-eligibility", "Windows eligibility: permanent residents aged 18 or older may request Lumen assistance."),
        ("windows-procedure", "Windows application procedure: applicants must meet the Lumen eligibility requirements. Open the Lumen desktop app, choose Assistance, and submit Form W-7."),
        ("linux-eligibility", "Linux eligibility: permanent residents aged 18 or older may request Lumen assistance."),
        ("linux-procedure", "Linux application procedure: applicants must meet the Lumen eligibility requirements. Run lumen-assist --apply, enter the residence identifier, and submit Form L-9."),
        ("unrelated", "Lumen library opening hours are Monday to Friday, 09:00 to 17:00."),
    ]
    corpus.write_text("".join(json.dumps({"_id": name, "text": text, "version": "v1"})+"\n"
                              for name, text in documents))
    study = {"study": {"strategy": "B6", "output_dir": str(output)},
        "generation": {"run_config": config_path, "teacher_url": auxiliary_url},
        "retrieval": {"corpus_path": str(corpus), "corpus_id": "provider-mechanisms",
            "current_top_k": 5, "per_need_top_k": 5, "archive_top_k": 5, "primary_pool_size": 16},
        "task_model": {"mode": "teacher", "base_path": history_model, "device": "cpu",
            "teacher_record_path": str(output/"task-teacher-calls.jsonl")},
        "support_model": {"mode": "teacher", "single_candidates_per_need": 8,
            "bridge_candidates_per_need": 4, "triple_seed_pairs": 1, "triple_successors_per_pair": 2},
        "selection": {"body_tokens": 4096, "max_actions": 6, "beam_width": 4,
            "retained_sets": 3, "initial_gap_search_rounds": 0, "layouts": ["existing", "need_order"]},
        "reading": {"enabled": True, "input_target_tokens": 12288,
            "output_tokens": 4096, "output_tokens_per_fact": 64, "mode": "typed_reduce"},
        "execution": {"window_tokens": 32768, "history_tokens": 4096, "output_tokens": 1024}}
    rows = []
    provider = None
    active = None
    try:
        retriever = await asyncio.to_thread(BM25SRetriever, load_beir_corpus(corpus),
                                            corpus_snapshot_id="provider-mechanisms-v1")
        frozen = await asyncio.to_thread(LocalBGE, history_model, device="cpu", dtype="bfloat16")
        support = SupportTeacherCoverage(auxiliary_url, service.template,
            record_path=output/"support-teacher-calls.jsonl")
        support.manifest["set_utility"] = "B3_support_coverage"
        provider = create_study_retriever(study, service.template, retriever=retriever,
            support_backend=support, frozen_history=frozen)
        for case, question in (
            ("windows", "What is the complete procedure, including the applicable eligibility rule, for applying for Lumen assistance on Windows?"),
            ("linux-correction", "Change the platform to Linux. Keep the eligibility requirement and explain the application steps."),
        ):
            request_start = time.monotonic()
            before = service.store.read_snapshot("provider-session", provider.corpus_snapshot_id)
            active = {"case": case, "question": question,
                "actual_history": to_jsonable(before.raw_history), "start": request_start}
            response = await service.serve_turn("provider-session", question, request_id=case,
                corpus_snapshot_id=provider.corpus_snapshot_id,
                evidence_provider=lambda snapshot: provider.retrieve(snapshot, question, case),
                open_need_provider=lambda plan, needs, claims: provider.extend_for_plan(before, plan, needs, claims))
            after = service.store.get_snapshot("provider-session")
            trace = provider.traces["provider-session", case]
            attempt = service.store.get_attempt(response.attempt_id)
            row = {"case": case, "status": "completed", "question": question, "actual_history": to_jsonable(before.raw_history),
                "trace": trace, "plan": to_jsonable(attempt.context_plan),
                "outcome": to_jsonable(attempt.reading_outcome), "response": to_jsonable(response),
                "qa_increment": len(after.raw_history)-len(before.raw_history),
                "version_increment": after.version-before.version,
                "resource_window": {"start": request_start, "end": time.monotonic()}}
            rows.append(row)
            active = None
            (output/f"{case}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
            print(json.dumps({"case": case, "answer": response.assistant_text,
                "read_count": response.read_count, "input_tokens": response.total_input_tokens,
                "output_tokens": response.total_output_tokens}, ensure_ascii=False), flush=True)
    finally:
        failure = sys.exc_info()[1]
        end = time.monotonic()
        if active is not None:
            attempt = service.store.find_request("provider-session", active["case"])
            after = service.store.get_snapshot("provider-session")
            rows.append({**active, "status": "failed", "error": str(failure),
                "attempt": to_jsonable(attempt),
                "steps": to_jsonable(service.store.execution_steps(attempt.attempt_id)) if attempt else [],
                "qa_increment": len(after.raw_history)-len(before.raw_history),
                "version_increment": after.version-before.version,
                "resource_window": {"start": active["start"], "end": end}})
        if provider is not None:
            provider.close()
        await service.aclose()
        service.store.close()
        await native.stop()
        await physical.stop()
        for row in rows:
            window = row["resource_window"]
            series, coverage = pool_series(native.observation(), window["start"], window["end"])
            row["native_cost"] = {"integrals": occupancy_integral(series), "coverage": coverage}
            row["physical_cost"] = integrate_physical(physical.samples, window["start"], window["end"])
            (output/f"{row['case']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
        series, coverage = pool_series(native.observation(), start, end)
        manifest = {"status": "completed" if failure is None else "failed", "error": str(failure) if failure else None,
            "input_origin": "controlled_sources_with_real_teacher_task_and_support_predictions",
            "background_offline_training": True, "configuration": study,
            "set_utility": "B3_support_coverage", "reader": "fixed_Qwen3_32B_nonthinking",
            "completed_cases": sum(row["status"] == "completed" for row in rows),
            "resource_window": {"start": start, "end": end},
            "physical_scope": "unique declared reader/auxiliary GPUs, their process groups and controller PSS",
            "endpoint_roles": {"READ_ANSWER": reader_url, "Task_Support": auxiliary_url},
            "physical_cost": integrate_physical(physical.samples, start, end),
            "native_cost": {"integrals": occupancy_integral(series), "coverage": coverage},
            "provider": provider.manifest if provider is not None else None,
            "auxiliary_model_usage": {role: model.usage for role, model in provider._usage_sources()}
                if provider is not None else {}}
        (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        (output/"native-observation.json").write_text(json.dumps(native.observation(), indent=2)+"\n")
        (output/"physical-observation.json").write_text(json.dumps(physical.observation(), indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reader-url", default="http://127.0.0.1:30000")
    parser.add_argument("--auxiliary-url", help="Defaults to the same reader endpoint")
    parser.add_argument("--history-model", default="models/bge-m3")
    parser.add_argument("--reader-gpu", type=int, default=0)
    parser.add_argument("--auxiliary-gpu", type=int, help="Defaults to the reader GPU")
    args = parser.parse_args()
    asyncio.run(run(args.config, args.output_dir, reader_url=args.reader_url,
        auxiliary_url=args.auxiliary_url or args.reader_url, history_model=args.history_model,
        reader_gpu=args.reader_gpu,
        auxiliary_gpu=args.auxiliary_gpu if args.auxiliary_gpu is not None else args.reader_gpu))


if __name__ == "__main__":
    main()
