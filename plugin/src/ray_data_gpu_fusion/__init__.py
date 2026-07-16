"""Opt-in GPU physical operators that execute inside Ray Data.

The package does not own a scheduler or consume a Dataset.  :func:`enable`
adds importable physical optimizer rule classes to a Ray ``DataContext``.  Ray
then plans, schedules, retries, meters, and shuts down the resulting operators.
"""

from ray_data_gpu_fusion.config import disable, enable, is_enabled
from ray_data_gpu_fusion.diagnostics import explain
from ray_data_gpu_fusion._compat import CompatibilityInfo, compatibility

__all__ = [
    "CompatibilityInfo",
    "compatibility",
    "disable",
    "enable",
    "explain",
    "is_enabled",
]
__version__ = "0.1.0.dev0"
