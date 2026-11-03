"""Concrete mechanism and matched-budget profiles for the study matrix."""
from __future__ import annotations

from copy import deepcopy


def development_variants(matrix):
    methods = deepcopy(matrix["methods"])
    by_id = {method["id"]: method for method in methods}
    ablations = matrix["ablations"]

    def add(parent, suffix, override, purpose):
        methods.append({"id": parent+"-"+suffix, "strategy": by_id[parent]["strategy"],
            "parent": parent, "comparison": purpose, "overrides": override})

    for budget in ablations["body_tokens"]:
        for parent in ("Btoken", "B3", "B4"):
            if parent in by_id:
                add(parent, f"body{budget}", {"selection": {"body_tokens": budget,
                    "layouts": ["existing"]}}, "same_generation_body_budget")
    group_modes = {(1,): "direct", (1, 2): "single_pair", (1, 2, 3): "single_pair_triple"}
    for orders in ablations["support_orders"]:
        for parent in ("B3", "B4"):
            if parent in by_id:
                add(parent, "orders"+"_".join(map(str, orders)),
                    {"support_model": {"group_mode": group_modes[tuple(orders)]}}, "support_arity")
    for mode in ablations["reading"]:
        add("B6", mode, {"reading": {"mode": mode}}, "reading_state_and_operations")
    for layout in ablations["layouts"]:
        add("B7", layout, {"selection": {"layouts": [layout]}}, "fixed_semantics_layout")
    for budget in ablations["output_tokens"]:
        add("B6", f"output{budget}", {"execution": {"output_tokens": budget}}, "fixed_output_budget")
    for weight in matrix["selection"].get("lambda_time_grid", ()):
        add("B7", "lambda"+str(weight), {"execution": {"lambda_time": weight}}, "quality_cost_curve")
    return methods
