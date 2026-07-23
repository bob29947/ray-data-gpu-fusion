#!/usr/bin/env python3
"""Measure the Ray candidate's production Python NCLOC delta.

Blank and comment-only lines are excluded.  The comparison includes uncommitted
worktree changes so it can enforce the maintenance budget before finalizing the
candidate patch series.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


BASE = "2741c6461d2bd3e5ff114af67be7a1190453dadd"
PRODUCTION_GROUPS = {
    "contract_controller_allocator": (
        "python/ray/data/_internal/execution/interfaces/physical_operator.py",
        "python/ray/data/_internal/execution/resource_admission.py",
        "python/ray/data/_internal/execution/resource_manager.py",
    ),
    "startup_backpressure": (
        "python/ray/data/_internal/cluster_autoscaler/"
        "default_autoscaling_coordinator.py",
        "python/ray/data/_internal/execution/streaming_executor.py",
        "python/ray/data/_internal/execution/streaming_executor_state.py",
    ),
    "actor_admission": (
        "python/ray/data/_internal/actor_autoscaler/default_actor_autoscaler.py",
        "python/ray/data/_internal/execution/operators/actor_pool_map_operator.py",
    ),
    "gpu_shuffle_lifecycle": (
        "python/ray/data/_internal/gpu_shuffle/hash_aggregate.py",
        "python/ray/data/_internal/gpu_shuffle/hash_shuffle.py",
    ),
    "context_flag": ("python/ray/data/context.py",),
}


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ray-worktree", type=Path, default=root / ".worktrees/ray-pr-minimal"
    )
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--max-net", type=int, default=650)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _is_ncloc(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _measure(worktree: Path, base: str) -> dict[str, dict[str, int]]:
    paths = tuple(path for group in PRODUCTION_GROUPS.values() for path in group)
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "diff",
            "--no-ext-diff",
            "--unified=0",
            base,
            "--",
            *paths,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    counts: dict[str, dict[str, int]] = {
        path: {"added": 0, "deleted": 0, "net": 0} for path in paths
    }
    current: str | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("+++ b/"):
            current = line.removeprefix("+++ b/")
            continue
        if current not in counts:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            counts[current]["added"] += int(_is_ncloc(line[1:]))
        elif line.startswith("-") and not line.startswith("---"):
            counts[current]["deleted"] += int(_is_ncloc(line[1:]))
    for values in counts.values():
        values["net"] = values["added"] - values["deleted"]
    return counts


def main() -> int:
    args = _parse_args()
    worktree = args.ray_worktree.resolve()
    per_file = _measure(worktree, args.base)
    categories: dict[str, dict[str, int]] = {}
    for name, paths in PRODUCTION_GROUPS.items():
        totals: defaultdict[str, int] = defaultdict(int)
        for path in paths:
            for metric, value in per_file[path].items():
                totals[metric] += value
        categories[name] = dict(totals)

    total: defaultdict[str, int] = defaultdict(int)
    for values in categories.values():
        for metric, value in values.items():
            total[metric] += value

    document = {
        "base": args.base,
        "maintenance_budget_net_ncloc": args.max_net,
        "within_budget": total["net"] <= args.max_net,
        "total": dict(total),
        "categories": categories,
        "files": per_file,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0 if document["within_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
