"""Human-readable plan diagnostics without starting a second execution path."""

from __future__ import annotations

from typing import Any

from ray_data_gpu_fusion._compat import explain_plan, optimized_physical_plan
from ray_data_gpu_fusion.config import settings
from ray_data_gpu_fusion.operators import ExecutableGPUOperator


def explain(dataset: Any) -> str:
    """Return Ray's logical/physical explanation plus plugin enablement state.

    Ray constructs and optimizes the physical plan for this report, so plugin
    operator names in the result are evidence of integration with Ray's own
    planner rather than a separately interpreted Dataset graph.
    """

    logical_plan = getattr(dataset, "_logical_plan", None)
    if logical_plan is None:
        raise TypeError("explain() requires a ray.data.Dataset")
    current = settings(getattr(logical_plan, "context", None))
    header = (
        "Ray Data GPU Fusion: " f"enabled={current.enabled}, fusion={current.fusion}\n"
    )
    ray_plan = explain_plan(logical_plan)
    if not current.enabled:
        decisions = "\n-------- GPU Fusion Decisions --------\nplugin disabled\n"
        return header + ray_plan + decisions

    physical = optimized_physical_plan(logical_plan)
    operators = []
    visited = set()
    stack = [physical.dag]
    while stack:
        operator = stack.pop()
        if operator in visited:
            continue
        visited.add(operator)
        operators.append(operator)
        stack.extend(reversed(operator.input_dependencies))

    lines = ["", "-------- GPU Fusion Decisions --------"]
    for operator in operators:
        if isinstance(operator, ExecutableGPUOperator):
            transforms = tuple(
                transform.kind for transform in operator.gpu_fusion_spec.transforms
            )
            shape = "fused" if len(transforms) > 1 else "standalone"
            lines.append(
                f"GPU {shape}: {operator.name}; transforms={transforms!r}; "
                "external_boundary=ray_arrow_blocks"
            )
            continue
        reason = getattr(operator, "_ray_data_gpu_fusion_decline_reason", None)
        if reason:
            lines.append(f"stock: {operator.name}; reason={reason}")
    if len(lines) == 2:
        lines.append("no supported or declined Phase-0 operators")
    return header + ray_plan + "\n".join(lines) + "\n"


__all__ = ["explain"]
