"""Plan-local enablement for the Ray Data GPU fusion rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ray_data_gpu_fusion._compat import (
    DataContext,
    get_rule_classes,
    set_rule_classes,
    verify_compatibility,
)

CONFIG_KEY = "ray_data_gpu_fusion.phase0"


@dataclass(frozen=True)
class FusionSettings:
    enabled: bool = True
    fusion: bool = True


def _context(context: Optional[DataContext]) -> DataContext:
    return DataContext.get_current() if context is None else context


def settings(context: Optional[DataContext] = None) -> FusionSettings:
    value = _context(context).get_config(CONFIG_KEY)
    return value if isinstance(value, FusionSettings) else FusionSettings(False, False)


def enable(
    *,
    context: Optional[DataContext] = None,
    fusion: bool = True,
) -> DataContext:
    """Enable GPU lowering and optional fusion on a Ray ``DataContext``.

    Call this before constructing a Dataset.  The two importable rule classes
    are copied with the Dataset context into remote planning locations; no
    process-global optimizer registry is modified.
    """

    selected = _context(context)
    verify_compatibility(selected)

    from ray_data_gpu_fusion.rules import (
        FuseClosedGPUOperators,
        LowerClosedGPUOperators,
    )

    classes = get_rule_classes(selected)
    for rule in (LowerClosedGPUOperators, FuseClosedGPUOperators):
        if rule not in classes:
            classes.append(rule)
    set_rule_classes(selected, classes)
    selected.set_config(CONFIG_KEY, FusionSettings(True, bool(fusion)))
    return selected


def disable(*, context: Optional[DataContext] = None) -> DataContext:
    """Remove this plugin's rules and settings from one context."""

    selected = _context(context)
    from ray_data_gpu_fusion.rules import (
        FuseClosedGPUOperators,
        LowerClosedGPUOperators,
    )

    if hasattr(selected, "custom_physical_optimizer_rule_classes"):
        ours = {LowerClosedGPUOperators, FuseClosedGPUOperators}
        set_rule_classes(
            selected, [rule for rule in get_rule_classes(selected) if rule not in ours]
        )
    selected.remove_config(CONFIG_KEY)
    return selected


def is_enabled(*, context: Optional[DataContext] = None) -> bool:
    return settings(context).enabled


__all__ = [
    "CONFIG_KEY",
    "FusionSettings",
    "disable",
    "enable",
    "is_enabled",
    "settings",
]
