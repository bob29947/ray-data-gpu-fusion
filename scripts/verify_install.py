#!/usr/bin/env python3
"""Verify the pinned Ray layers and the installed hooked runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_WORKTREE = ROOT / ".worktrees" / "ray-pr-candidate"
CANDIDATE_BRANCH = "ray-data-gpu-actor-admission"
CANDIDATE_SUBJECT = "[Data] Add resource-aware admission for GPU actor pools"
CANDIDATE_PATCH = Path(
    "ray-pr-candidate/0001-ray-data-resource-aware-gpu-actor-admission.patch"
)
HOOK_PATCHES = (
    Path("ray-hooks/0001-ray-data-support-plan-local-physical-optimizer-rules.patch"),
    Path(
        "ray-hooks/"
        "0002-ray-data-expose-backend-neutral-parquet-scan-descriptor.patch"
    ),
)
CANDIDATE_MANIFEST = Path("pins/pr-candidate.json")
HOOKED_MANIFEST = Path("pins/hooked-ray.json")
STOCK_INPUTS = (
    Path("environment/rapids-25.12-linux-64.explicit.txt"),
    Path("environment/ray-runtime-linux-py311.lock"),
    Path("pins/stock-ray.json"),
    Path("pins/source-snapshot.json"),
)


def _load_pin(name: str) -> dict:
    path = ROOT / "pins" / name
    if not path.is_file():
        raise RuntimeError(
            f"missing finalized layer manifest: {path}; build C and finalize pins first"
        )
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"layer manifest is not a JSON object: {path}")
    return value


STOCK_PIN = _load_pin("stock-ray.json")
CANDIDATE_PIN = _load_pin("pr-candidate.json")
HOOKED_PIN = _load_pin("hooked-ray.json")


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _series_sha256(files: tuple[Path, ...]) -> str:
    """Hash an ordered patch series using the wheel builder's encoding."""

    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _verify_file(path: Path, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )


def _verify_file_size(path: Path, expected_size: object, label: str) -> None:
    if type(expected_size) is not int or expected_size < 0:
        raise RuntimeError(f"{label} manifest size must be a nonnegative integer")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )


def _patch_from_commit(path: Path) -> str:
    first_line = path.open(encoding="utf-8").readline().rstrip("\n")
    match = re.fullmatch(r"From ([0-9a-f]{40}) Mon Sep 17 00:00:00 2001", first_line)
    if match is None:
        raise RuntimeError(f"file is not an exact format-patch: {path}")
    return match.group(1)


def _verify_local_candidate_commit(candidate_patch: Path) -> None:
    worktree = CANDIDATE_WORKTREE
    if not worktree.is_dir() or not (worktree / ".git").is_file():
        raise RuntimeError(f"missing required linked PR-candidate worktree: {worktree}")
    candidate_common_dir = Path(
        _run("git", "-C", str(worktree), "rev-parse", "--git-common-dir")
    ).resolve()
    stock_common_dir = Path(
        _run("git", "-C", str(ROOT / "ray-stock"), "rev-parse", "--git-common-dir")
    ).resolve()
    if candidate_common_dir != stock_common_dir:
        raise RuntimeError("candidate worktree is not linked to the pinned ray-stock")
    status = _run("git", "-C", str(worktree), "status", "--porcelain")
    if status:
        raise RuntimeError(f"local PR-candidate worktree must be clean:\n{status}")
    branch = _run("git", "-C", str(worktree), "symbolic-ref", "--short", "HEAD")
    if branch != CANDIDATE_BRANCH:
        raise RuntimeError(
            f"candidate branch {branch!r} != required {CANDIDATE_BRANCH!r}"
        )
    subject = _run("git", "-C", str(worktree), "show", "-s", "--format=%s", "HEAD")
    if subject != CANDIDATE_SUBJECT:
        raise RuntimeError(
            f"candidate subject {subject!r} != required {CANDIDATE_SUBJECT!r}"
        )
    head = _run("git", "-C", str(worktree), "rev-parse", "HEAD")
    local_commit = CANDIDATE_PIN.get("local_commit")
    if head != local_commit or head != _patch_from_commit(candidate_patch):
        raise RuntimeError("candidate worktree, manifest, and patch commit disagree")
    parent = _run("git", "-C", str(worktree), "rev-parse", "HEAD^")
    if parent != STOCK_PIN.get("commit"):
        raise RuntimeError("local candidate parent is not the pinned stock commit")
    count = _run(
        "git",
        "-C",
        str(worktree),
        "rev-list",
        "--count",
        f"{STOCK_PIN['commit']}..HEAD",
    )
    if count != "1":
        raise RuntimeError(
            f"local candidate must be one commit above stock, got {count}"
        )
    tree = _run("git", "-C", str(worktree), "rev-parse", "HEAD^{tree}")
    if tree != CANDIDATE_PIN.get("source_tree"):
        raise RuntimeError("local candidate tree does not match candidate manifest")
    changed = sorted(
        _run(
            "git",
            "-C",
            str(worktree),
            "diff",
            "--name-only",
            f"{STOCK_PIN['commit']}..HEAD",
        ).splitlines()
    )
    if changed != CANDIDATE_PIN.get("changed_files"):
        raise RuntimeError("local candidate changed files do not match manifest")
    insertions = 0
    deletions = 0
    numstat = _run(
        "git",
        "-C",
        str(worktree),
        "diff",
        "--numstat",
        f"{STOCK_PIN['commit']}..HEAD",
    )
    for line in numstat.splitlines():
        added, removed, _path = line.split("\t", 2)
        if added == "-" or removed == "-":
            raise RuntimeError("local candidate contains an unrecorded binary delta")
        insertions += int(added)
        deletions += int(removed)
    if CANDIDATE_PIN.get("loc") != {
        "insertions": insertions,
        "deletions": deletions,
    }:
        raise RuntimeError("local candidate LOC does not match manifest")


def _wheel_python_files(wheel: Path) -> dict[str, str]:
    """Return hashes for every production Python source in a Ray wheel."""

    with zipfile.ZipFile(wheel) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("ray/") and name.endswith(".py")
        ]
        if len(names) != len(set(names)):
            raise RuntimeError(f"wheel contains duplicate Python members: {wheel}")
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(names)
        }


def _production_delta(base_wheel: Path, derived_wheel: Path) -> dict[str, object]:
    base = _wheel_python_files(base_wheel)
    derived = _wheel_python_files(derived_wheel)
    return {
        "files": {
            name: digest for name, digest in derived.items() if base.get(name) != digest
        },
        "removed": sorted(set(base) - set(derived)),
    }


def _validated_manifest_delta(manifest: dict, label: str) -> dict[str, object]:
    delta = manifest.get("production_delta")
    if not isinstance(delta, dict) or set(delta) != {"files", "removed"}:
        raise RuntimeError(f"{label} production_delta has the wrong schema")
    files = delta.get("files")
    removed = delta.get("removed")
    if not isinstance(files, dict) or not isinstance(removed, list):
        raise RuntimeError(f"{label} production_delta has the wrong schema")
    for relative, digest in files.items():
        if (
            not isinstance(relative, str)
            or not relative.startswith("ray/")
            or not relative.endswith(".py")
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError(f"{label} production file hash entry is invalid")
    if any(
        not isinstance(relative, str)
        or not relative.startswith("ray/")
        or not relative.endswith(".py")
        for relative in removed
    ):
        raise RuntimeError(f"{label} removed-file list is invalid")
    if removed != sorted(set(removed)):
        raise RuntimeError(f"{label} removed-file list must be sorted and unique")
    if set(files).intersection(removed):
        raise RuntimeError(f"{label} production delta both changes and removes a file")
    return delta


def _verify_wheel_production_delta(
    stock_wheel: Path, derived_wheel: Path, manifest: dict, label: str
) -> None:
    expected = _validated_manifest_delta(manifest, label)
    actual = _production_delta(stock_wheel, derived_wheel)
    if actual != expected:
        raise RuntimeError(
            f"{label} Python production delta does not match its manifest: "
            f"expected={expected}, actual={actual}"
        )


def _verify_checksums(wheel_name: str) -> None:
    checksum_path = ROOT / "pins" / "SHA256SUMS"
    if not checksum_path.is_file():
        raise RuntimeError(f"missing checksum manifest: {checksum_path}")
    entries: dict[Path, str] = {}
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\s].*)", line)
        if match is None:
            raise RuntimeError(f"invalid pins/SHA256SUMS line {line_number}: {line!r}")
        relative = Path(match.group(2))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"unsafe pins/SHA256SUMS path at line {line_number}: {relative}"
            )
        if relative in entries:
            raise RuntimeError(f"duplicate pins/SHA256SUMS path: {relative}")
        entries[relative] = match.group(1)

    expected_paths = {
        *STOCK_INPUTS,
        CANDIDATE_PATCH,
        *HOOK_PATCHES,
        CANDIDATE_MANIFEST,
        HOOKED_MANIFEST,
        Path("wheels/stock") / wheel_name,
        Path("wheels/pr-candidate") / wheel_name,
        Path("wheels/hooked") / wheel_name,
    }
    if set(entries) != expected_paths:
        missing = sorted(path.as_posix() for path in expected_paths - set(entries))
        unexpected = sorted(path.as_posix() for path in set(entries) - expected_paths)
        raise RuntimeError(
            "pins/SHA256SUMS path set mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, expected_hash in entries.items():
        _verify_file(ROOT / relative, expected_hash, f"checksum entry {relative}")


def _verify_manifest_links() -> tuple[Path, Path, Path]:
    if (
        CANDIDATE_PIN.get("schema_version") != 1
        or CANDIDATE_PIN.get("layer") != "pr-candidate"
    ):
        raise RuntimeError("pins/pr-candidate.json has the wrong schema or layer")
    if HOOKED_PIN.get("schema_version") != 1 or HOOKED_PIN.get("layer") != "hooked-ray":
        raise RuntimeError("pins/hooked-ray.json has the wrong schema or layer")
    if CANDIDATE_PIN.get("base_commit") != STOCK_PIN.get("commit"):
        raise RuntimeError("candidate manifest does not derive from pinned stock Ray")
    if CANDIDATE_PIN.get("base_version") != STOCK_PIN.get("version"):
        raise RuntimeError("candidate manifest version does not match stock Ray")
    if CANDIDATE_PIN.get("stock_wheel_sha256") != STOCK_PIN.get("wheel_sha256"):
        raise RuntimeError("candidate manifest has the wrong stock wheel hash")
    if HOOKED_PIN.get("base_manifest") != "pins/pr-candidate.json":
        raise RuntimeError("hooked manifest does not name the candidate manifest")
    if HOOKED_PIN.get("base_source_tree") != CANDIDATE_PIN.get("source_tree"):
        raise RuntimeError("hooked manifest has the wrong candidate source tree")
    if HOOKED_PIN.get("base_wheel_sha256") != CANDIDATE_PIN.get("wheel_sha256"):
        raise RuntimeError("hooked manifest has the wrong candidate wheel hash")

    candidate_patch = CANDIDATE_PIN.get("patch")
    if not isinstance(candidate_patch, dict):
        raise RuntimeError("candidate manifest has no patch object")
    expected_candidate = CANDIDATE_PATCH.as_posix()
    if candidate_patch.get("file") != expected_candidate:
        raise RuntimeError("candidate manifest does not name the exact C patch")
    _verify_file(
        ROOT / expected_candidate,
        str(candidate_patch.get("sha256")),
        "candidate patch C",
    )
    candidate_path = ROOT / expected_candidate
    candidate_from_commit = _patch_from_commit(candidate_path)
    if candidate_patch.get("from_commit") != candidate_from_commit:
        raise RuntimeError("candidate patch from_commit does not match patch header")
    if CANDIDATE_PIN.get("local_commit") != candidate_from_commit:
        raise RuntimeError("candidate local_commit does not match patch header")
    if CANDIDATE_PIN.get("patch_series_sha256") != _series_sha256((CANDIDATE_PATCH,)):
        raise RuntimeError("candidate patch-series hash does not match C")
    if not isinstance(CANDIDATE_PIN.get("changed_files"), list):
        raise RuntimeError("candidate manifest has no changed_files list")
    loc = CANDIDATE_PIN.get("loc")
    if not isinstance(loc, dict) or set(loc) != {"insertions", "deletions"}:
        raise RuntimeError("candidate manifest has no exact LOC counts")
    if any(not isinstance(value, int) or value < 0 for value in loc.values()):
        raise RuntimeError("candidate manifest LOC counts must be nonnegative integers")
    _verify_local_candidate_commit(candidate_path)

    hooks = HOOKED_PIN.get("hooks")
    expected_hooks = [
        (f"H{index}", relative.as_posix())
        for index, relative in enumerate(HOOK_PATCHES, 1)
    ]
    if not isinstance(hooks, list) or len(hooks) != len(expected_hooks):
        raise RuntimeError("hooked manifest must contain exactly H1 and H2")
    for hook, (expected_id, expected_file) in zip(hooks, expected_hooks, strict=True):
        if not isinstance(hook, dict):
            raise RuntimeError("hook manifest entry must be an object")
        if hook.get("id") != expected_id or hook.get("file") != expected_file:
            raise RuntimeError(
                f"hook manifest entry must be {expected_id} at {expected_file}"
            )
        _verify_file(
            ROOT / expected_file,
            str(hook.get("sha256")),
            f"hook {expected_id}",
        )
        hook_from_commit = _patch_from_commit(ROOT / expected_file)
        if hook.get("from_commit") != hook_from_commit:
            raise RuntimeError(
                f"hook {expected_id} from_commit does not match patch header"
            )

    if HOOKED_PIN.get("hook_series_sha256") != _series_sha256(HOOK_PATCHES):
        raise RuntimeError("hook patch-series hash does not match H1+H2")
    if HOOKED_PIN.get("aggregate_series_sha256") != _series_sha256(
        (CANDIDATE_PATCH, *HOOK_PATCHES)
    ):
        raise RuntimeError("aggregate patch-series hash does not match C+H1+H2")

    wheel_name = STOCK_PIN.get("wheel")
    if (
        not isinstance(wheel_name, str)
        or CANDIDATE_PIN.get("wheel") != wheel_name
        or HOOKED_PIN.get("wheel") != wheel_name
    ):
        raise RuntimeError("layer manifests disagree on the wheel filename")
    stock_wheel = ROOT / "wheels" / "stock" / wheel_name
    candidate_wheel = ROOT / "wheels" / "pr-candidate" / wheel_name
    hooked_wheel = ROOT / "wheels" / "hooked" / wheel_name
    _verify_file(stock_wheel, str(STOCK_PIN.get("wheel_sha256")), "stock wheel")
    _verify_file_size(stock_wheel, STOCK_PIN.get("wheel_size_bytes"), "stock wheel")
    _verify_file(
        candidate_wheel,
        str(CANDIDATE_PIN.get("wheel_sha256")),
        "PR-candidate wheel",
    )
    _verify_file_size(
        candidate_wheel,
        CANDIDATE_PIN.get("wheel_size_bytes"),
        "PR-candidate wheel",
    )
    _verify_file(hooked_wheel, str(HOOKED_PIN.get("wheel_sha256")), "hooked wheel")
    _verify_file_size(hooked_wheel, HOOKED_PIN.get("wheel_size_bytes"), "hooked wheel")
    if (
        len(
            {
                STOCK_PIN.get("wheel_sha256"),
                CANDIDATE_PIN.get("wheel_sha256"),
                HOOKED_PIN.get("wheel_sha256"),
            }
        )
        != 3
    ):
        raise RuntimeError("stock, candidate, and hooked wheel hashes must be distinct")
    _verify_wheel_production_delta(
        stock_wheel,
        candidate_wheel,
        CANDIDATE_PIN,
        "PR-candidate wheel",
    )
    _verify_wheel_production_delta(
        stock_wheel,
        hooked_wheel,
        HOOKED_PIN,
        "hooked wheel",
    )
    return stock_wheel, candidate_wheel, hooked_wheel


def _verify_installed_delta(site_packages: Path) -> None:
    delta = _validated_manifest_delta(HOOKED_PIN, "hooked manifest")
    files = delta["files"]
    removed = delta["removed"]
    assert isinstance(files, dict)
    assert isinstance(removed, list)
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise RuntimeError("hooked production file hash entry is invalid")
        _verify_file(site_packages / relative, expected_hash, f"installed {relative}")
    for relative in removed:
        if not isinstance(relative, str):
            raise RuntimeError("hooked removed-file entry is invalid")
        if (site_packages / relative).exists():
            raise RuntimeError(f"installed hooked layer retained {relative}")


def main() -> int:
    stock = ROOT / "ray-stock"
    head = _run("git", "-C", str(stock), "rev-parse", "HEAD")
    if head != STOCK_PIN["commit"]:
        raise RuntimeError(f"ray-stock HEAD {head} != {STOCK_PIN['commit']}")
    if _run("git", "-C", str(stock), "status", "--porcelain"):
        raise RuntimeError("ray-stock must remain byte-for-byte clean")

    stock_wheel, candidate_wheel, hooked_wheel = _verify_manifest_links()
    _verify_checksums(stock_wheel.name)

    import ray

    if ray.__version__ != STOCK_PIN["version"] or ray.__commit__ != STOCK_PIN["commit"]:
        raise RuntimeError(
            f"installed Ray identity mismatch: {ray.__version__} {ray.__commit__}"
        )
    installed_ray = Path(ray.__file__).resolve()
    if stock in installed_ray.parents:
        raise RuntimeError("Ray is imported from ray-stock instead of the wheel")
    expected_prefix = (ROOT / ".venv").resolve()
    if expected_prefix not in Path(sys.executable).resolve().parents:
        raise RuntimeError(
            f"interpreter is outside {expected_prefix}: {sys.executable}"
        )
    _verify_installed_delta(installed_ray.parent.parent)

    from ray.data.context import DataContext
    from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
    from ray.data._internal.execution.resource_manager import (
        GPU_ACTOR_ADMISSION_CONTROL_VERSION,
    )

    if not hasattr(DataContext(), "custom_physical_optimizer_rule_classes"):
        raise RuntimeError("plan-local physical optimizer hook H1 is absent")
    if not hasattr(ParquetDatasource, "get_external_scan_descriptor"):
        raise RuntimeError("Parquet external scan descriptor hook H2 is absent")
    context = DataContext()
    if GPU_ACTOR_ADMISSION_CONTROL_VERSION != 1:
        raise RuntimeError("candidate GPU actor admission capability has wrong version")
    if not isinstance(
        getattr(context, "_enable_gpu_actor_admission_control", None), bool
    ):
        raise RuntimeError("candidate GPU actor admission rollback field is absent")

    import ray_data_gpu_fusion as rgf

    compatibility = rgf.compatibility()
    if compatibility.ray_commit != STOCK_PIN["commit"] or not compatibility.supported:
        raise RuntimeError(f"plugin compatibility check failed: {compatibility}")

    if os.environ.get("PYTHONHOME"):
        raise RuntimeError("PYTHONHOME may redirect the pinned interpreter")
    python_path = os.environ.get("PYTHONPATH", "")
    if str(stock) in python_path:
        raise RuntimeError("PYTHONPATH redirects imports into ray-stock")

    # RAPIDS is installed from the explicit conda lock. ``pip check`` cannot
    # validate that environment because native dependencies are conda-owned.
    import cudf
    import cupy
    import pyarrow
    import rmm

    runtime_versions = {
        "cudf": cudf.__version__,
        "cupy": cupy.__version__,
        "pyarrow": pyarrow.__version__,
        "rmm": rmm.__version__,
    }
    expected_versions = {
        "cudf": "25.12.00",
        "cupy": "13.6.0",
        "pyarrow": "19.0.1",
        "rmm": "25.12.00",
    }
    if runtime_versions != expected_versions:
        raise RuntimeError(
            "RAPIDS runtime version mismatch: "
            f"installed={runtime_versions}, expected={expected_versions}"
        )
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "ray": ray.__version__,
                "ray_commit": ray.__commit__,
                "ray_path": str(installed_ray),
                "stock_wheel": str(stock_wheel),
                "candidate_wheel": str(candidate_wheel),
                "candidate_wheel_sha256": CANDIDATE_PIN["wheel_sha256"],
                "hooked_wheel": str(hooked_wheel),
                "hooked_wheel_sha256": HOOKED_PIN["wheel_sha256"],
                "plugin": rgf.__version__,
                "adapter": compatibility.adapter,
                "gpu_actor_admission_control": GPU_ACTOR_ADMISSION_CONTROL_VERSION,
                "runtime": runtime_versions,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
