#!/usr/bin/env python3
"""Build and verify the PR-candidate and locally hooked Ray wheel layers.

The pinned ``ray-stock`` checkout is never modified.  Two independent temporary
worktrees prove the layer boundaries:

* ``wheels/pr-candidate`` is the stock wheel plus candidate patch C.
* ``wheels/hooked`` is the candidate wheel plus local hooks H1 and H2.

The hooked wheel is also derived directly from stock+C+H1+H2 and must be
byte-for-byte equal to the layered result.  Missing candidate inputs are
reported as finalization work; this script never substitutes placeholder
hashes.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

CANDIDATE_PATCH = Path(
    "ray-pr-candidate/0001-ray-data-generic-resource-admission.patch"
)
HOOK_PATCHES = (
    Path("ray-hooks/0001-ray-data-support-plan-local-physical-optimizer-rules.patch"),
    Path(
        "ray-hooks/0002-ray-data-expose-backend-neutral-parquet-scan-descriptor.patch"
    ),
)
CANDIDATE_MANIFEST = Path("pins/pr-candidate.json")
HOOKED_MANIFEST = Path("pins/hooked-ray.json")
BUILD_SCRIPT = "scripts/build_ray_wheels.py"
CANDIDATE_BRANCH = "codex/gpu-admission-minimal"
CANDIDATE_WORKTREE = Path(".worktrees/ray-pr-minimal")
CANDIDATE_SUBJECT = "[Data] Add generic resource admission for GPU operators"
LEGACY_LIFECYCLE_SYMBOLS = (
    "defer_actor_start",
    "wait_for_upstream_deferred_operators",
    "release_idle_actors_on_completion",
    "actor_pool_start_deferred",
    "configure_demand_driven_start",
)
LEGACY_ARTIFACTS = (
    Path("pins/patched-ray.json"),
    Path("ray-patches"),
    Path("wheels/patched"),
    Path("scripts/build_patched_ray_wheel.py"),
)
EXPECTED_HOOK_PATHS = {
    HOOK_PATCHES[0]: {
        "python/ray/data/_internal/logical/optimizers.py",
        "python/ray/data/context.py",
    },
    HOOK_PATCHES[1]: {
        "python/ray/data/_internal/datasource/parquet_datasource.py",
    },
}
CANDIDATE_PRODUCTION_PATHS = {
    "python/ray/data/context.py",
    "python/ray/data/_internal/actor_autoscaler/default_actor_autoscaler.py",
    "python/ray/data/_internal/cluster_autoscaler/default_autoscaling_coordinator.py",
    "python/ray/data/_internal/execution/interfaces/physical_operator.py",
    "python/ray/data/_internal/execution/operators/actor_pool_map_operator.py",
    "python/ray/data/_internal/execution/resource_admission.py",
    "python/ray/data/_internal/execution/resource_manager.py",
    "python/ray/data/_internal/execution/streaming_executor.py",
    "python/ray/data/_internal/execution/streaming_executor_state.py",
    "python/ray/data/_internal/gpu_shuffle/hash_aggregate.py",
    "python/ray/data/_internal/gpu_shuffle/hash_shuffle.py",
}
CANDIDATE_FORBIDDEN_TEXT = (
    "custom_physical_optimizer_rule_classes",
    "get_external_scan_descriptor",
    "parquetexternalscan",
    "ray_data_gpu_fusion",
    "ray-data-gpu-fusion",
    "plugin",
    "fusion",
)


def _run(*args: str | Path, capture: bool = False) -> str:
    result = subprocess.run(
        [str(arg) for arg in args],
        check=True,
        stdout=subprocess.PIPE if capture else None,
        text=True,
    )
    return result.stdout if capture else ""


def _git(repo: Path, *args: str, capture: bool = True) -> str:
    return _run("git", "-C", repo, *args, capture=capture)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _series_sha256(project_root: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update((project_root / relative).read_bytes())
    return digest.hexdigest()


def _record_digest(data: bytes) -> tuple[str, str]:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return f"sha256={encoded.rstrip(b'=').decode()}", str(len(data))


def _zip_info(name: str, epoch: int) -> zipfile.ZipInfo:
    # ZIP timestamps cannot represent dates before 1980.
    info = zipfile.ZipInfo(name, time.gmtime(max(epoch, 315532800))[:6])
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _load_json(path: Path, *, required: bool) -> dict | None:
    if not path.is_file():
        if required:
            raise RuntimeError(f"missing required manifest: {path}")
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"manifest must contain a JSON object: {path}")
    return value


def _find_stock_wheel(
    project_root: Path, stock_pin: dict, requested: Path | None
) -> Path:
    if requested is not None:
        wheel = requested.resolve()
    else:
        wheel_name = stock_pin.get("wheel")
        if not isinstance(wheel_name, str) or not wheel_name:
            raise RuntimeError("pins/stock-ray.json has no wheel filename")
        wheel = (project_root / "wheels" / "stock" / wheel_name).resolve()
    if not wheel.is_file():
        raise RuntimeError(f"stock wheel does not exist: {wheel}")
    return wheel


def _validate_exact_inputs(project_root: Path) -> tuple[Path, tuple[Path, ...]]:
    candidate = project_root / CANDIDATE_PATCH
    if not candidate.is_file():
        raise RuntimeError(
            "PR candidate layer C is not finalized: expected "
            f"{CANDIDATE_PATCH}. Export the reviewed candidate patch, then rerun "
            "this builder to obtain the real tree and wheel hashes."
        )

    hook_dir = project_root / "ray-hooks"
    expected_hook_names = {path.name for path in HOOK_PATCHES}
    actual_hook_entries = (
        {path.relative_to(hook_dir).as_posix() for path in hook_dir.rglob("*")}
        if hook_dir.is_dir()
        else set()
    )
    if actual_hook_entries != expected_hook_names:
        raise RuntimeError(
            "ray-hooks must contain exactly H1 and H2: "
            f"expected={sorted(expected_hook_names)}, "
            f"actual={sorted(actual_hook_entries)}"
        )
    hooks = tuple(project_root / relative for relative in HOOK_PATCHES)
    for hook in hooks:
        hook_text = hook.read_text().lower()
        forbidden = [
            symbol for symbol in LEGACY_LIFECYCLE_SYMBOLS if symbol.lower() in hook_text
        ]
        if "admission" in hook_text:
            forbidden.append("alternate admission implementation")
        if forbidden:
            raise RuntimeError(
                f"local hook {hook} contains forbidden lifecycle/admission code: "
                f"{forbidden}"
            )
    present_legacy_artifacts = [
        relative.as_posix()
        for relative in LEGACY_ARTIFACTS
        if (project_root / relative).exists()
    ]
    if present_legacy_artifacts:
        raise RuntimeError(
            "legacy patched-Ray artifacts must be removed before building: "
            f"{present_legacy_artifacts}"
        )
    return candidate, hooks


def _patch_from_commit(path: Path) -> str:
    first_line = path.open(encoding="utf-8").readline().rstrip("\n")
    match = re.fullmatch(r"From ([0-9a-f]{40}) Mon Sep 17 00:00:00 2001", first_line)
    if match is None:
        raise RuntimeError(
            f"patch must be a format-patch with an exact source commit: {path}"
        )
    return match.group(1)


def _validate_stock(ray_stock: Path, stock_wheel: Path, stock_pin: dict) -> None:
    expected_commit = stock_pin.get("commit")
    commit = _git(ray_stock, "rev-parse", "HEAD").strip()
    if commit != expected_commit:
        raise RuntimeError(
            f"ray-stock is at {commit}, expected pinned commit {expected_commit}"
        )
    status = _git(ray_stock, "status", "--porcelain").strip()
    if status:
        raise RuntimeError(f"ray-stock must be pristine before building:\n{status}")

    actual_hash = _sha256(stock_wheel)
    expected_hash = stock_pin.get("wheel_sha256")
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"stock wheel SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    with zipfile.ZipFile(stock_wheel) as archive:
        try:
            version_source = archive.read("ray/_version.py").decode()
        except KeyError as error:
            raise RuntimeError("stock wheel has no ray/_version.py") from error
    match = re.search(
        r'^commit = "([0-9a-f]{40})"$', version_source, flags=re.MULTILINE
    )
    if match is None or match.group(1) != expected_commit:
        reported = match.group(1) if match else "<missing>"
        raise RuntimeError(
            f"stock wheel reports commit {reported}, expected {expected_commit}"
        )


def _validate_local_candidate_commit(
    project_root: Path,
    stock_commit: str,
    candidate_commit: str,
    *,
    expected_tree: str | None = None,
) -> None:
    candidate_worktree = project_root / CANDIDATE_WORKTREE
    if not candidate_worktree.exists():
        raise RuntimeError(
            f"missing required local PR-candidate worktree: {candidate_worktree}"
        )
    status = _git(candidate_worktree, "status", "--porcelain").strip()
    if status:
        raise RuntimeError(
            f"local PR-candidate worktree must be clean before finalization:\n{status}"
        )
    head = _git(candidate_worktree, "rev-parse", "HEAD").strip()
    if head != candidate_commit:
        raise RuntimeError(
            f"candidate patch commit {candidate_commit} != local HEAD {head}"
        )
    branch = _git(candidate_worktree, "symbolic-ref", "--short", "HEAD").strip()
    if branch != CANDIDATE_BRANCH:
        raise RuntimeError(
            f"candidate branch {branch!r} != required {CANDIDATE_BRANCH!r}"
        )
    subject = _git(candidate_worktree, "show", "-s", "--format=%s", "HEAD").strip()
    if subject != CANDIDATE_SUBJECT:
        raise RuntimeError(
            f"candidate subject {subject!r} != required {CANDIDATE_SUBJECT!r}"
        )
    parent = _git(candidate_worktree, "rev-parse", "HEAD^").strip()
    if parent != stock_commit:
        raise RuntimeError(f"candidate parent {parent} != stock base {stock_commit}")
    count = _git(
        candidate_worktree, "rev-list", "--count", f"{stock_commit}..HEAD"
    ).strip()
    if count != "1":
        raise RuntimeError(
            f"candidate must be exactly one commit above stock, got {count} commits"
        )
    if expected_tree is not None:
        local_tree = _git(candidate_worktree, "rev-parse", "HEAD^{tree}").strip()
        if local_tree != expected_tree:
            raise RuntimeError(
                "applied candidate tree does not equal local candidate commit tree: "
                f"{expected_tree} != {local_tree}"
            )


@contextmanager
def _temporary_worktree(
    ray_stock: Path, stock_commit: str, prefix: str
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        worktree = Path(temporary) / "ray"
        _run(
            "git",
            "-C",
            ray_stock,
            "worktree",
            "add",
            "--detach",
            worktree,
            stock_commit,
        )
        try:
            if _git(worktree, "status", "--porcelain").strip():
                raise RuntimeError(f"new worktree is unexpectedly dirty: {worktree}")
            yield worktree
        finally:
            if worktree.exists():
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(ray_stock),
                        "worktree",
                        "remove",
                        "--force",
                        str(worktree),
                    ],
                    check=False,
                )
            subprocess.run(
                ["git", "-C", str(ray_stock), "worktree", "prune"], check=False
            )


def _apply_patches(worktree: Path, patches: tuple[Path, ...]) -> None:
    for patch in patches:
        _git(worktree, "apply", "--check", str(patch), capture=False)
        _git(worktree, "apply", str(patch), capture=False)
    _git(worktree, "diff", "--check", capture=False)


def _changed_paths(worktree: Path, base: str | None = "HEAD") -> list[str]:
    command = ["git", "-C", str(worktree), "diff", "--name-only", "-z"]
    if base is not None:
        command.append(base)
    raw = subprocess.check_output(command).split(b"\0")
    return sorted(item.decode() for item in raw if item)


def _line_counts(worktree: Path, base: str | None = "HEAD") -> dict[str, int]:
    args = ["diff", "--numstat"]
    if base is not None:
        args.append(base)
    output = _git(worktree, *args)
    insertions = 0
    deletions = 0
    for line in output.splitlines():
        added, removed, _path = line.split("\t", 2)
        if added == "-" or removed == "-":
            raise RuntimeError("candidate patch must not contain binary changes")
        insertions += int(added)
        deletions += int(removed)
    return {"insertions": insertions, "deletions": deletions}


def _patch_stats(patch: Path) -> tuple[list[str], dict[str, int]]:
    output = _run("git", "apply", "--numstat", patch, capture=True)
    files: list[str] = []
    insertions = 0
    deletions = 0
    for line in output.splitlines():
        added, removed, path = line.split("\t", 2)
        if added == "-" or removed == "-":
            raise RuntimeError(f"patch must not contain binary changes: {patch}")
        files.append(path)
        insertions += int(added)
        deletions += int(removed)
    return sorted(files), {"insertions": insertions, "deletions": deletions}


def _validate_candidate_scope(
    worktree: Path, candidate_patch: Path
) -> tuple[list[str], dict[str, int]]:
    changed_paths = _changed_paths(worktree)
    if not changed_paths:
        raise RuntimeError("candidate patch C has no changed files")
    for path in changed_paths:
        if path in CANDIDATE_PRODUCTION_PATHS:
            continue
        if path.startswith("python/ray/data/tests/") and path.endswith(".py"):
            continue
        if path.startswith("doc/source/data/release-notes/") and path.endswith(
            (".md", ".rst")
        ):
            continue
        raise RuntimeError(f"candidate has an out-of-scope changed path: {path}")

    candidate_text = candidate_patch.read_text().lower()
    forbidden = [token for token in CANDIDATE_FORBIDDEN_TEXT if token in candidate_text]
    forbidden.extend(
        symbol
        for symbol in LEGACY_LIFECYCLE_SYMBOLS
        if symbol.lower() in candidate_text
    )
    if forbidden:
        raise RuntimeError(
            "candidate C must contain only generic resource admission control; "
            f"found forbidden hook/plugin terms: {forbidden}"
        )
    return changed_paths, _line_counts(worktree)


def _validate_hook_scope(
    worktree: Path, hook_patches: tuple[Path, ...]
) -> tuple[list[str], dict[str, int]]:
    changed_paths = _changed_paths(worktree, base=None)
    expected_paths = sorted(
        path for relative in HOOK_PATCHES for path in EXPECTED_HOOK_PATHS[relative]
    )
    if changed_paths != expected_paths:
        raise RuntimeError(
            "hook source scope mismatch: "
            f"expected={expected_paths}, actual={changed_paths}"
        )

    for relative, patch in zip(HOOK_PATCHES, hook_patches, strict=True):
        patch_paths, _ = _patch_stats(patch)
        expected_patch_paths = sorted(EXPECTED_HOOK_PATHS[relative])
        if patch_paths != expected_patch_paths:
            raise RuntimeError(
                f"hook {relative} scope mismatch: "
                f"expected={expected_patch_paths}, actual={patch_paths}"
            )

    hook_text = "\n".join(path.read_text().lower() for path in hook_patches)
    forbidden = [
        symbol for symbol in LEGACY_LIFECYCLE_SYMBOLS if symbol.lower() in hook_text
    ]
    forbidden.extend(
        token
        for token in ("ray_data_gpu_fusion", "ray-data-gpu-fusion", "cudf", "cuda")
        if token in hook_text
    )
    if "gpu_actor_admission" in hook_text:
        forbidden.append("alternate GPU actor admission implementation")
    if forbidden:
        raise RuntimeError(f"hook patches contain out-of-scope code: {forbidden}")
    return changed_paths, _line_counts(worktree, base=None)


def _validate_no_legacy_lifecycle_sources(worktree: Path, layer: str) -> None:
    source_root = worktree / "python" / "ray" / "data"
    findings: list[str] = []
    for source in source_root.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        for symbol in LEGACY_LIFECYCLE_SYMBOLS:
            if symbol in text:
                findings.append(f"{source.relative_to(worktree)}:{symbol}")
    if findings:
        raise RuntimeError(
            f"{layer} contains the removed actor-lifecycle seam: {findings}"
        )


def _validate_wheel_has_no_legacy_lifecycle(output: Path) -> None:
    findings: list[str] = []
    with zipfile.ZipFile(output) as archive:
        for name in archive.namelist():
            if not (name.startswith("ray/data/") and name.endswith(".py")):
                continue
            text = archive.read(name).decode("utf-8")
            for symbol in LEGACY_LIFECYCLE_SYMBOLS:
                if symbol in text:
                    findings.append(f"{name}:{symbol}")
    if findings:
        raise RuntimeError(
            f"derived wheel contains the removed actor-lifecycle seam: {findings}"
        )


def _production_overlay(
    worktree: Path, *, base: str | None
) -> tuple[dict[str, bytes], set[str]]:
    command = [
        "git",
        "-C",
        str(worktree),
        "diff",
        "--name-status",
        "--no-renames",
        "-z",
    ]
    if base is not None:
        command.append(base)
    raw = subprocess.check_output(command).split(b"\0")
    if raw and raw[-1] == b"":
        raw.pop()
    if len(raw) % 2:
        raise RuntimeError("unexpected git diff --name-status output")

    overlay: dict[str, bytes] = {}
    removals: set[str] = set()
    for index in range(0, len(raw), 2):
        status = raw[index].decode()
        source_name = raw[index + 1].decode()
        if "/tests/" in source_name or source_name.endswith("/BUILD.bazel"):
            continue
        # Release notes and other explicitly scope-checked non-wheel files are
        # provenance, not wheel members.
        if not source_name.startswith("python/ray/") or not source_name.endswith(".py"):
            continue
        wheel_name = source_name.removeprefix("python/")
        status_kind = status[0]
        if status_kind == "D":
            removals.add(wheel_name)
        elif status_kind in {"A", "M"}:
            overlay[wheel_name] = (worktree / source_name).read_bytes()
        else:
            raise RuntimeError(f"unsupported patch status {status!r}: {source_name}")
    if not overlay and not removals:
        label = base if base is not None else "the candidate index"
        raise RuntimeError(f"layer relative to {label} has no production wheel delta")
    return overlay, removals


def _stage_and_write_tree(worktree: Path) -> str:
    _git(worktree, "add", "--all", capture=False)
    _git(worktree, "diff", "--cached", "--check", capture=False)
    return _git(worktree, "write-tree").strip()


def _write_wheel(
    base_wheel: Path,
    output: Path,
    overlay: dict[str, bytes],
    removals: set[str],
    epoch: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(base_wheel) as source:
            names = set(source.namelist())
            signed_records = {
                name
                for name in names
                if name.endswith((".dist-info/RECORD.jws", ".dist-info/RECORD.p7s"))
            }
            if signed_records:
                raise RuntimeError(
                    f"cannot derive a wheel with signed RECORD files: {signed_records}"
                )
            record_names = [
                name
                for name in names
                if name.endswith(".dist-info/RECORD")
                and "/" not in name.removesuffix("/RECORD")
            ]
            if len(record_names) != 1:
                raise RuntimeError("base wheel must contain exactly one RECORD")
            record_name = record_names[0]
            original_records = {
                row[0]: (row[1], row[2])
                for row in csv.reader(
                    io.StringIO(source.read(record_name).decode("utf-8"))
                )
            }
            records = {
                name: fields
                for name, fields in original_records.items()
                if name not in removals and name != record_name
            }
            for name, data in overlay.items():
                records[name] = _record_digest(data)

            record_stream = io.StringIO(newline="")
            writer = csv.writer(record_stream, lineterminator="\n")
            for name in sorted(records):
                writer.writerow((name, *records[name]))
            writer.writerow((record_name, "", ""))
            record_data = record_stream.getvalue().encode()

            with zipfile.ZipFile(
                temporary,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as target:
                written: set[str] = set()
                for info in source.infolist():
                    name = info.filename
                    if name == record_name or name in removals:
                        continue
                    if name in overlay:
                        target.writestr(info, overlay[name])
                    else:
                        with (
                            source.open(info) as source_file,
                            target.open(info, "w") as target_file,
                        ):
                            shutil.copyfileobj(source_file, target_file, length=1 << 20)
                    written.add(name)
                for name in sorted(set(overlay) - written):
                    target.writestr(_zip_info(name, epoch), overlay[name])
                target.writestr(_zip_info(record_name, epoch), record_data)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_output_wheel(
    output: Path,
    overlay: dict[str, bytes],
    removals: set[str],
) -> None:
    """Verify archive integrity, layered bytes, and every top-level RECORD row."""

    with zipfile.ZipFile(output) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"derived wheel has a corrupt member: {corrupt}")
        names = {name for name in archive.namelist() if not name.endswith("/")}
        record_names = [
            name
            for name in names
            if name.endswith(".dist-info/RECORD")
            and "/" not in name.removesuffix("/RECORD")
        ]
        if len(record_names) != 1:
            raise RuntimeError(
                "derived wheel must contain exactly one top-level RECORD"
            )
        record_name = record_names[0]
        rows = {
            row[0]: (row[1], row[2])
            for row in csv.reader(
                io.StringIO(archive.read(record_name).decode("utf-8"))
            )
        }
        if set(rows) != names:
            raise RuntimeError(
                "derived wheel RECORD membership mismatch: "
                f"unrecorded={sorted(names - set(rows))}, "
                f"missing={sorted(set(rows) - names)}"
            )
        for name, (digest, size) in rows.items():
            if name == record_name:
                if digest or size:
                    raise RuntimeError("RECORD must not hash itself")
                continue
            data = archive.read(name)
            if size and int(size) != len(data):
                raise RuntimeError(f"RECORD size mismatch for {name}")
            if digest and digest != _record_digest(data)[0]:
                raise RuntimeError(f"RECORD digest mismatch for {name}")
        for name, data in overlay.items():
            if archive.read(name) != data:
                raise RuntimeError(f"derived wheel has wrong layered bytes: {name}")
        unexpected = names.intersection(removals)
        if unexpected:
            raise RuntimeError(f"derived wheel retained removed files: {unexpected}")


def _delta_manifest(overlay: dict[str, bytes], removals: set[str]) -> dict[str, object]:
    return {
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(overlay.items())
        },
        "removed": sorted(removals),
    }


def _compare_manifest(
    actual: object, expected: object, path: str, missing: list[str]
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeError(f"manifest value {path} must be an object")
        extra = sorted(set(actual) - set(expected))
        if extra:
            raise RuntimeError(f"manifest value {path} has unexpected keys: {extra}")
        for key, expected_value in expected.items():
            child = f"{path}.{key}"
            if key not in actual:
                missing.append(child)
            else:
                _compare_manifest(actual[key], expected_value, child, missing)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(
                f"manifest value {path} must be a {len(expected)}-item list"
            )
        for index, expected_value in enumerate(expected):
            _compare_manifest(
                actual[index], expected_value, f"{path}[{index}]", missing
            )
        return
    if actual != expected:
        raise RuntimeError(
            f"manifest mismatch at {path}: expected {expected!r}, got {actual!r}"
        )


def _validate_or_report_manifest(
    manifest_path: Path,
    actual: dict | None,
    expected: dict,
    *,
    require_pins: bool,
) -> None:
    if actual is None:
        if require_pins:
            raise RuntimeError(
                f"missing finalized manifest {manifest_path}; run without "
                "--require-pins to print the real values after C exists"
            )
        print(f"FINALIZE {manifest_path} with:")
        print(json.dumps(expected, indent=2, sort_keys=False))
        return
    missing: list[str] = []
    _compare_manifest(actual, expected, manifest_path.as_posix(), missing)
    if missing:
        if require_pins:
            raise RuntimeError(
                f"manifest {manifest_path} is not finalized; missing {missing}"
            )
        print(f"FINALIZE {manifest_path}; missing fields: {missing}")
        print(json.dumps(expected, indent=2, sort_keys=False))


def _stage_manifest(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
        temporary.chmod(mode)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_manifests_transactionally(items: tuple[tuple[Path, dict], ...]) -> None:
    """Publish a manifest set, restoring every original on replacement failure."""
    paths = [path for path, _ in items]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest transaction contains duplicate paths")
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    rollback_paths: list[Path] = []
    preserve_backups = False
    try:
        for path, value in items:
            staged[path] = _stage_manifest(path, value)
        for path in paths:
            if path.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{path.name}.backup.", dir=path.parent
                )
                os.close(descriptor)
                backup = Path(backup_name)
                shutil.copy2(path, backup)
                backups[path] = backup
            else:
                backups[path] = None
        for path in paths:
            rollback_paths.append(path)
            os.replace(staged[path], path)
    except BaseException as error:
        rollback_errors = []
        for path in reversed(rollback_paths):
            try:
                backup = backups[path]
                if backup is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(backup, path)
            except BaseException as rollback_error:
                rollback_errors.append(f"{path}: {rollback_error}")
        if rollback_errors:
            preserve_backups = True
            recovery_paths = [
                str(backup)
                for backup in backups.values()
                if backup is not None and backup.exists()
            ]
            raise RuntimeError(
                "manifest publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
                + f"; preserved recovery files: {recovery_paths}"
            ) from error
        raise
    finally:
        cleanup = list(staged.values())
        if not preserve_backups:
            cleanup.extend(backup for backup in backups.values() if backup is not None)
        for temporary in cleanup:
            temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    project_root = args.project_root.resolve()
    if getattr(args, "write_pins", False) and (
        args.candidate_output_dir is not None or args.hooked_output_dir is not None
    ):
        raise RuntimeError("--write-pins requires the default wheel output directories")
    ray_stock = project_root / "ray-stock"
    stock_pin = _load_json(project_root / "pins" / "stock-ray.json", required=True)
    assert stock_pin is not None
    candidate_patch, hook_patches = _validate_exact_inputs(project_root)
    stock_wheel = _find_stock_wheel(project_root, stock_pin, args.stock_wheel)
    candidate_manifest = _load_json(project_root / CANDIDATE_MANIFEST, required=False)
    hooked_manifest = _load_json(project_root / HOOKED_MANIFEST, required=False)

    _validate_stock(ray_stock, stock_wheel, stock_pin)
    stock_commit = str(stock_pin["commit"])
    candidate_commit = _patch_from_commit(candidate_patch)
    _validate_local_candidate_commit(project_root, stock_commit, candidate_commit)
    epoch = int(_git(ray_stock, "show", "-s", "--format=%ct", stock_commit).strip())
    candidate_output_dir = (
        args.candidate_output_dir.resolve()
        if args.candidate_output_dir is not None
        else project_root / "wheels" / "pr-candidate"
    )
    hooked_output_dir = (
        args.hooked_output_dir.resolve()
        if args.hooked_output_dir is not None
        else project_root / "wheels" / "hooked"
    )
    candidate_output = candidate_output_dir / stock_wheel.name
    hooked_output = hooked_output_dir / stock_wheel.name

    with _temporary_worktree(
        ray_stock, stock_commit, "ray-pr-candidate-layer-"
    ) as worktree:
        _apply_patches(worktree, (candidate_patch,))
        # Stage before inspecting so newly added files (for example a release
        # note) are included in scope, LOC, and source-tree provenance.
        candidate_tree = _stage_and_write_tree(worktree)
        _validate_no_legacy_lifecycle_sources(worktree, "candidate layer C")
        candidate_changed_files, candidate_loc = _validate_candidate_scope(
            worktree, candidate_patch
        )
        candidate_overlay, candidate_removals = _production_overlay(
            worktree, base="HEAD"
        )
        _validate_local_candidate_commit(
            project_root,
            stock_commit,
            candidate_commit,
            expected_tree=candidate_tree,
        )
        _write_wheel(
            stock_wheel,
            candidate_output,
            candidate_overlay,
            candidate_removals,
            epoch,
        )
        _validate_output_wheel(candidate_output, candidate_overlay, candidate_removals)
        _validate_wheel_has_no_legacy_lifecycle(candidate_output)
    _validate_stock(ray_stock, stock_wheel, stock_pin)

    with _temporary_worktree(ray_stock, stock_commit, "ray-hooked-layer-") as worktree:
        # Apply C independently again rather than reusing the candidate worktree.
        _apply_patches(worktree, (candidate_patch,))
        independently_derived_candidate_tree = _stage_and_write_tree(worktree)
        if independently_derived_candidate_tree != candidate_tree:
            raise RuntimeError(
                "independent candidate applications produced different trees: "
                f"{candidate_tree} != {independently_derived_candidate_tree}"
            )

        _apply_patches(worktree, hook_patches)
        hook_changed_files, hook_loc = _validate_hook_scope(worktree, hook_patches)
        hook_overlay, hook_removals = _production_overlay(worktree, base=None)
        full_overlay, full_removals = _production_overlay(worktree, base="HEAD")
        hooked_tree = _stage_and_write_tree(worktree)
        _validate_no_legacy_lifecycle_sources(worktree, "hooked layer")

        # The published hooked wheel is layered on the candidate wheel.
        _write_wheel(
            candidate_output,
            hooked_output,
            hook_overlay,
            hook_removals,
            epoch,
        )
        _validate_output_wheel(hooked_output, hook_overlay, hook_removals)
        _validate_wheel_has_no_legacy_lifecycle(hooked_output)

    _validate_stock(ray_stock, stock_wheel, stock_pin)

    # Independently apply the full series from stock. This is deliberately a
    # third worktree: the direct proof must not reuse the tree from which the
    # layered wheel was assembled.
    with _temporary_worktree(
        ray_stock, stock_commit, "ray-direct-final-layer-"
    ) as direct_worktree:
        _apply_patches(direct_worktree, (candidate_patch, *hook_patches))
        direct_tree = _stage_and_write_tree(direct_worktree)
        direct_full_overlay, direct_full_removals = _production_overlay(
            direct_worktree, base="HEAD"
        )
        _validate_no_legacy_lifecycle_sources(
            direct_worktree, "independently derived final layer"
        )
        if direct_tree != hooked_tree:
            raise RuntimeError(
                "final source-tree equality failed: candidate+hooks differs from "
                f"independent stock+C+H1+H2 ({hooked_tree} != {direct_tree})"
            )

        # A direct stock+C+H1+H2 wheel derivation must be byte-identical.
        direct_output = Path(direct_worktree.parent) / stock_wheel.name
        _write_wheel(
            stock_wheel,
            direct_output,
            direct_full_overlay,
            direct_full_removals,
            epoch,
        )
        _validate_output_wheel(direct_output, direct_full_overlay, direct_full_removals)
        _validate_wheel_has_no_legacy_lifecycle(direct_output)
        layered_hash = _sha256(hooked_output)
        direct_hash = _sha256(direct_output)
        if layered_hash != direct_hash:
            raise RuntimeError(
                "layer equality failed: candidate+hooks wheel differs from "
                f"direct stock+C+H1+H2 ({layered_hash} != {direct_hash})"
            )
    _validate_stock(ray_stock, stock_wheel, stock_pin)

    candidate_files = (CANDIDATE_PATCH,)
    hook_files = HOOK_PATCHES
    all_files = candidate_files + hook_files
    candidate_expected = {
        "schema_version": 1,
        "layer": "pr-candidate",
        "base_commit": stock_commit,
        "base_version": stock_pin["version"],
        "stock_wheel_sha256": stock_pin["wheel_sha256"],
        "local_commit": candidate_commit,
        "patch": {
            "file": CANDIDATE_PATCH.as_posix(),
            "from_commit": _patch_from_commit(candidate_patch),
            "sha256": _sha256(candidate_patch),
        },
        "patch_series_sha256": _series_sha256(project_root, candidate_files),
        "source_tree": candidate_tree,
        "changed_files": candidate_changed_files,
        "loc": candidate_loc,
        "production_delta": _delta_manifest(candidate_overlay, candidate_removals),
        "wheel": stock_wheel.name,
        "wheel_sha256": _sha256(candidate_output),
        "wheel_size_bytes": candidate_output.stat().st_size,
        "build": BUILD_SCRIPT,
    }
    hooked_expected = {
        "schema_version": 1,
        "layer": "hooked-ray",
        "base_manifest": CANDIDATE_MANIFEST.as_posix(),
        "base_source_tree": candidate_tree,
        "base_wheel_sha256": candidate_expected["wheel_sha256"],
        "hooks": [
            {
                "id": f"H{index}",
                "file": relative.as_posix(),
                "from_commit": _patch_from_commit(project_root / relative),
                "sha256": _sha256(project_root / relative),
                "changed_files": _patch_stats(project_root / relative)[0],
                "loc": _patch_stats(project_root / relative)[1],
            }
            for index, relative in enumerate(hook_files, start=1)
        ],
        "hook_series_sha256": _series_sha256(project_root, hook_files),
        "aggregate_series_sha256": _series_sha256(project_root, all_files),
        "changed_files": hook_changed_files,
        "loc": hook_loc,
        "source_tree": hooked_tree,
        "production_delta": _delta_manifest(full_overlay, full_removals),
        "wheel": stock_wheel.name,
        "wheel_sha256": _sha256(hooked_output),
        "wheel_size_bytes": hooked_output.stat().st_size,
        "build": BUILD_SCRIPT,
    }
    if getattr(args, "write_pins", False):
        _write_manifests_transactionally(
            (
                (project_root / CANDIDATE_MANIFEST, candidate_expected),
                (project_root / HOOKED_MANIFEST, hooked_expected),
            )
        )
        candidate_manifest = candidate_expected
        hooked_manifest = hooked_expected
    _validate_or_report_manifest(
        CANDIDATE_MANIFEST,
        candidate_manifest,
        candidate_expected,
        require_pins=args.require_pins,
    )
    _validate_or_report_manifest(
        HOOKED_MANIFEST,
        hooked_manifest,
        hooked_expected,
        require_pins=args.require_pins,
    )

    print(f"ray_stock_commit={stock_commit}")
    print(f"stock_wheel_sha256={_sha256(stock_wheel)}")
    print(f"candidate_source_tree={candidate_tree}")
    print(f"candidate_wheel={candidate_output}")
    print(f"candidate_wheel_sha256={candidate_expected['wheel_sha256']}")
    print(f"hooked_source_tree={hooked_tree}")
    print(f"hooked_wheel={hooked_output}")
    print(f"hooked_wheel_sha256={hooked_expected['wheel_sha256']}")
    print("layer_equality=verified")
    return candidate_output, hooked_output


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--stock-wheel", type=Path)
    parser.add_argument("--candidate-output-dir", type=Path)
    parser.add_argument("--hooked-output-dir", type=Path)
    parser.add_argument(
        "--require-pins",
        action="store_true",
        help="require complete, matching candidate and hooked manifests",
    )
    parser.add_argument(
        "--write-pins",
        action="store_true",
        help="atomically refresh the derived candidate and hooked manifests",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
