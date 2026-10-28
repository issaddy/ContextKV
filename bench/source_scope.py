"""Active execution dependencies versus metric calculation and presentation."""
from dataclasses import asdict
import ast
from hashlib import sha256

import yaml
from pathlib import Path


METRIC_SOURCES = {"bench/report.py", "bench/metrics.py", "bench/quality.py",
                  "bench/quality_acceptance.py", "bench/judge.py", "bench/official.py"}


def retrieval_units(source):
    """Keep optional search algorithms independently attributable in a shared module."""
    module = ast.parse(source)
    units, common = {}, []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            units[node.name] = ast.dump(node, include_attributes=False)
        else:
            common.append(ast.dump(node, include_attributes=False))
    units["module"] = "\n".join(common)
    return {f"adapters/hybrid_retrieval.py::{name}": sha256(value.encode()).hexdigest()
            for name, value in units.items()}


def source_dependencies(spec, config, files):
    # Keep the complete file manifest separately. Only modules on this declared
    # execution path invalidate answers; disabled experiments are not inputs.
    generation = {path for path in files if path.startswith("controller/")
                  and path not in {"controller/api.py", "controller/__main__.py"}}
    generation.update({"adapters/mtrag.py", "adapters/retrieval.py", "adapters/corpus.py",
        "adapters/retriever_factory.py", "adapters/source_structure.py",
        "bench/experiments.py", "bench/run_sessions.py", "bench/native_metrics.py",
        "engine_patches/contextkv_model_scope.py"})
    if any(method.get("kind") == "native" for method in spec["methods"]):
        generation.add("bench/native_baseline.py")
    if spec.get("server_profiles") or spec.get("lifecycle", {}).get("before_job"):
        generation.update({"bench/b6_joint_sweep.py", "bench/native_policy_sweep.py",
            "bench/controller_capacity_sweep.py", "scripts/prepare_cold_job.py",
            "scripts/launch_sglang.sh", "profiling/common.py"})
    if "P3" in spec["protocols"]:
        for method in spec["methods"]:
            path = method.get("retriever_config") or spec.get("retriever_config")
            if not path:
                continue
            recipe = yaml.safe_load(Path(path).read_text())
            generation.update({"adapters/query_views.py",
                               "adapters/query_resolution.py", "adapters/table_relations.py"})
            names = {"module", "query_text", "rerank_query_text", "reciprocal_rank_fusion",
                     "select_ranked_evidence", "LocalBGE", "DenseIndex", "HybridRetriever",
                     "validate_retrieval_options"}
            fusion = recipe.get("query_fusion", "single")
            semantic = recipe.get("evidence_selection") == "semantic"
            if semantic:
                names.update({"joint_recall_queries", "search_current_topic"})
            elif fusion == "dialogue_rrf":
                names.update({"search_dialogue_views", "_recall_query_views"})
            else:
                names.add("search_current_topic" if fusion == "current_topic_rrf" else "search_variants")
                if recipe.get("query_mode") == "conversational":
                    names.update({"search_requirements", "select_requirement_evidence"})
            if "search_current_topic" in names:
                names.update({"search_variants", "_recall_query_views"})
            generation.update(f"adapters/hybrid_retrieval.py::{name}" for name in names)
            if recipe.get("query_mode") == "conversational":
                generation.add("adapters/conversation.py")
            if recipe.get("query_mode") in {"conversational", "source_topic", "assistant_topic_unverified"}:
                generation.add("adapters/contextpilot.py")  # Shared current-source navigation.
            if recipe.get("selector"):
                generation.add("adapters/query_selector.py")
            if recipe.get("evidence_selection") == "semantic":
                generation.add("adapters/evidence_selection.py")
            if recipe.get("source_window") and recipe.get("source_documents"):
                generation.add("adapters/source_documents.py")
    for method in spec["methods"]:
        for candidate in method.get("candidates", [{}]):
            context = {**asdict(config.context), **method.get("overrides", {}).get("context", {}),
                       **candidate.get("context", {})}
            planner = {**asdict(config.planner), **method.get("overrides", {}).get("planner", {}),
                       **candidate.get("planner", {})}
            if context["layout"] == "contextpilot" or planner["strategy"] == "contextpilot":
                generation.add("adapters/contextpilot.py")
            if context["table_relations"]:
                generation.update({"adapters/table_relations.py", "adapters/table_structure.py"})
                if context["table_annotation_style"] == "focus_refs":
                    generation.add("adapters/table_focus.py")
    return {"generation": sorted(generation & files.keys()),
            "metrics": sorted(METRIC_SOURCES & files.keys()),
            "presentation": ["bench/report_rendering.py"]}


def source_changes(recorded, current):
    """Classify changes without discarding the original source/provenance record."""
    if not recorded or not recorded.get("implementation") or not recorded.get("dependencies"):
        return {"generation": ["unbound_implementation"], "metrics": [], "presentation": []}
    old = {**recorded["implementation"]["files"], **recorded["implementation"].get("units", {})}
    new = {**current["implementation"]["files"], **current["implementation"].get("units", {})}
    return {kind: sorted(path for path in set(recorded["dependencies"][kind]) | set(paths)
                        if old.get(path) != new.get(path))
            for kind, paths in current["dependencies"].items()}
