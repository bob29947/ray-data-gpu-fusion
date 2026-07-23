#!/usr/bin/env python3
"""Run the focused Ray admission suite against both derived wheel layers.

The candidate tests live in the clean linked Ray worktree, but production Ray
is always imported from the wheel currently installed in ``.venv``.  The final
``finally`` block restores and validates the hooked wheel even if candidate or
hooked tests fail.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = (ROOT / ".venv").resolve()
CANDIDATE_WORKTREE = ROOT / ".worktrees" / "ray-pr-minimal"
CANDIDATE_BRANCH = "codex/gpu-admission-minimal"
TEST_RAY_ROOT = CANDIDATE_WORKTREE / "python" / "ray"
BOOTSTRAP = ROOT / "scripts" / "ray_acceptance_bootstrap"

SPILL_TEST = (
    "test_actor_pool_map_operator.py::"
    "test_resource_admission_handoff_spills_arrow_blocks"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-spill",
        action="store_true",
        help="skip only the 205 MB low-object-store spill E2E",
    )
    return parser.parse_args()


def _require_venv_interpreter() -> None:
    executable = Path(sys.executable).resolve()
    if VENV not in executable.parents:
        raise RuntimeError(
            f"run this script with {VENV / 'bin' / 'python'}, got {executable}"
        )


def _require_test_prerequisites() -> None:
    missing = [
        name
        for name in ("freezegun", "pyarrow", "pytest")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Ray acceptance prerequisites are missing from .venv: "
            f"{', '.join(missing)}; bootstrap the pinned environment first"
        )


def _wheel_name() -> str:
    pin = json.loads((ROOT / "pins" / "stock-ray.json").read_text())
    name = pin.get("wheel")
    if not isinstance(name, str) or not name.endswith(".whl"):
        raise RuntimeError("pins/stock-ray.json has no valid wheel filename")
    return name


def _verify_wheel_pin(wheel: Path, manifest_name: str, expected_layer: str) -> None:
    if not wheel.is_file():
        raise RuntimeError(f"required Ray wheel is missing: {wheel}")
    manifest = json.loads((ROOT / "pins" / manifest_name).read_text())
    if manifest.get("layer") != expected_layer:
        raise RuntimeError(f"pins/{manifest_name} has the wrong layer")
    if manifest.get("wheel") != wheel.name:
        raise RuntimeError(f"pins/{manifest_name} names a different wheel")
    expected_size = manifest.get("wheel_size_bytes")
    if expected_size != wheel.stat().st_size:
        raise RuntimeError(f"{wheel} size does not match pins/{manifest_name}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != manifest.get("wheel_sha256"):
        raise RuntimeError(f"{wheel} hash does not match pins/{manifest_name}")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(CANDIDATE_WORKTREE), *args], text=True
    ).strip()


def _verify_candidate_test_worktree() -> None:
    """Bind acceptance test sources to the finalized candidate manifest."""
    manifest = json.loads((ROOT / "pins" / "pr-candidate.json").read_text())
    if not isinstance(manifest, dict):
        raise RuntimeError("candidate manifest must contain a JSON object")
    if manifest.get("schema_version") != 1 or manifest.get("layer") != "pr-candidate":
        raise RuntimeError("candidate manifest has the wrong schema or layer")
    expected_head = manifest.get("local_commit")
    expected_tree = manifest.get("source_tree")
    if (
        not isinstance(expected_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected_head) is None
    ):
        raise RuntimeError("candidate manifest has no valid local commit")
    if (
        not isinstance(expected_tree, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected_tree) is None
    ):
        raise RuntimeError("candidate manifest has no valid source tree")
    if not CANDIDATE_WORKTREE.is_dir():
        raise RuntimeError(
            f"candidate Ray test worktree is missing: {CANDIDATE_WORKTREE}"
        )
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(f"candidate Ray test worktree must be clean:\n{status}")
    branch = _git("symbolic-ref", "--short", "HEAD")
    if branch != CANDIDATE_BRANCH:
        raise RuntimeError(
            f"candidate test branch {branch!r} != required {CANDIDATE_BRANCH!r}"
        )
    head = _git("rev-parse", "HEAD")
    if head != expected_head:
        raise RuntimeError(
            f"candidate test HEAD {head} != manifest commit {expected_head}"
        )
    tree = _git("rev-parse", "HEAD^{tree}")
    if tree != expected_tree:
        raise RuntimeError(
            f"candidate test tree {tree} != manifest tree {expected_tree}"
        )


def _install(wheel: Path) -> None:
    if not wheel.is_file():
        raise RuntimeError(f"required Ray wheel is missing: {wheel}")
    print(f"\n==> Installing {wheel.relative_to(ROOT)}", flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheel),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )


def _test_environment(layer: str) -> dict[str, str]:
    data_tests = TEST_RAY_ROOT / "data" / "tests"
    ray_tmpdir = Path(
        os.environ.get("RAY_GPU_ACCEPTANCE_TMPDIR", "/dev/shm/ray-admission")
    )
    ray_tmpdir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Do not retain an ambient source checkout on PYTHONPATH. The bootstrap
    # exposes exactly the test packages plus the top-level test module path used
    # when callable classes are imported in Ray workers.
    environment["PYTHONPATH"] = os.pathsep.join((str(BOOTSTRAP), str(data_tests)))
    environment.update(
        {
            "RAY_GPU_ACCEPTANCE_TEST_ROOT": str(TEST_RAY_ROOT),
            "RAY_GPU_ACCEPTANCE_VENV": str(VENV),
            "RAY_GPU_ACCEPTANCE_LAYER": layer,
            "RAY_TMPDIR": str(ray_tmpdir),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _assert_layer(layer: str) -> None:
    subprocess.run(
        [sys.executable, "-c", "import sitecustomize"],
        check=True,
        cwd=ROOT,
        env=_test_environment(layer),
    )


def _candidate_test_files() -> list[Path]:
    manifest = json.loads((ROOT / "pins" / "pr-candidate.json").read_text())
    prefix = "python/ray/"
    relative_files = [
        path.removeprefix(prefix)
        for path in manifest.get("changed_files", [])
        if isinstance(path, str)
        and path.startswith(f"{prefix}data/tests/")
        and path.endswith(".py")
    ]
    files = [TEST_RAY_ROOT / relative for relative in relative_files]
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError(
            "candidate manifest does not resolve to its changed Ray test modules"
        )
    return files


def _pytest(layer: str, label: str, targets: list[Path], *extra: str) -> None:
    print(f"\n==> Ray {layer}: {label}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(ROOT),
            "--basetemp",
            str(ROOT / ".worktrees" / "ray-acceptance-tmp"),
            *extra,
            *(str(path) for path in targets),
        ],
        check=True,
        cwd=ROOT,
        env=_test_environment(layer),
    )


def _run_tests(layer: str, *, skip_spill: bool) -> None:
    data_tests = TEST_RAY_ROOT / "data" / "tests"
    # The manifest supplies changed test modules so newly added admission tests
    # are automatically covered after a candidate amend. Keep the large spill
    # case separate so developers can explicitly omit only that one test.
    # Ray's module-scoped cluster fixtures are not reliable when these otherwise
    # independent suites share one pytest process, so isolate each changed module.
    for test_file in _candidate_test_files():
        _pytest(
            layer,
            test_file.name,
            [test_file],
            "-m",
            "not gpu",
            "-k",
            "not resource_admission_handoff_spills_arrow_blocks",
        )
    if not skip_spill:
        _pytest(layer, "low-object-store spill E2E", [data_tests / SPILL_TEST])


def _interrupt(_signal_number: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    args = _parse_args()
    _require_venv_interpreter()
    wheel_name = _wheel_name()
    candidate = ROOT / "wheels" / "pr-candidate" / wheel_name
    hooked = ROOT / "wheels" / "hooked" / wheel_name
    if not hooked.is_file():
        raise RuntimeError(
            f"final hooked wheel is unavailable for guaranteed restore: {hooked}"
        )
    _verify_wheel_pin(hooked, "hooked-ray.json", "hooked-ray")
    signal.signal(signal.SIGTERM, _interrupt)

    try:
        _verify_wheel_pin(candidate, "pr-candidate.json", "pr-candidate")
        _require_test_prerequisites()
        _verify_candidate_test_worktree()
        _install(candidate)
        _assert_layer("candidate")
        _run_tests("candidate", skip_spill=args.skip_spill)

        _install(hooked)
        _assert_layer("hooked")
        _run_tests("hooked", skip_spill=args.skip_spill)
    finally:
        # This is intentionally unconditional: a failed candidate test must not
        # leave the developer environment on the intermediate wheel.
        _install(hooked)
        _assert_layer("hooked")

    print("\nRay wheel-layer acceptance passed; hooked Ray is installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
