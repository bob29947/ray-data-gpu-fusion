#!/usr/bin/env python3
"""Apply the Phase 0 Ray patches and create a reproducible derived wheel.

The pinned ``ray-stock`` checkout is never modified. Patches are applied in a
temporary Git worktree and their production Python delta is overlaid onto the
validated stock wheel. The wheel's RECORD is regenerated for every changed file.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


RAY_STOCK_COMMIT = "2741c6461d2bd3e5ff114af67be7a1190453dadd"
RAY_STOCK_WHEEL_SHA256 = (
    "fee51e2415cb4f4f46fb81d23ad0ff557600f8278c7aefc782cf2b7301668dea"
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


def _find_stock_wheel(project_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        wheel = requested.resolve()
        if not wheel.is_file():
            raise RuntimeError(f"stock wheel does not exist: {wheel}")
        return wheel
    wheels = sorted((project_root / "wheels" / "stock").glob("ray-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            "expected exactly one stock Ray wheel under wheels/stock; "
            "pass --stock-wheel to select one explicitly"
        )
    return wheels[0].resolve()


def _validate_inputs(
    ray_stock: Path,
    stock_wheel: Path,
    patch_dir: Path,
) -> list[Path]:
    commit = _git(ray_stock, "rev-parse", "HEAD").strip()
    if commit != RAY_STOCK_COMMIT:
        raise RuntimeError(
            f"ray-stock is at {commit}, expected pinned commit {RAY_STOCK_COMMIT}"
        )
    status = _git(ray_stock, "status", "--porcelain").strip()
    if status:
        raise RuntimeError(f"ray-stock must be pristine before building:\n{status}")

    actual_hash = _sha256(stock_wheel)
    if actual_hash != RAY_STOCK_WHEEL_SHA256:
        raise RuntimeError(
            "stock wheel SHA-256 mismatch: "
            f"expected {RAY_STOCK_WHEEL_SHA256}, got {actual_hash}"
        )
    with zipfile.ZipFile(stock_wheel) as archive:
        try:
            version_source = archive.read("ray/_version.py").decode()
        except KeyError as error:
            raise RuntimeError("stock wheel has no ray/_version.py") from error
    match = re.search(
        r'^commit = "([0-9a-f]{40})"$', version_source, flags=re.MULTILINE
    )
    if match is None or match.group(1) != RAY_STOCK_COMMIT:
        reported = match.group(1) if match else "<missing>"
        raise RuntimeError(
            f"stock wheel reports commit {reported}, expected {RAY_STOCK_COMMIT}"
        )

    patches = sorted(patch_dir.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    if not patches:
        raise RuntimeError(f"no numbered patches found in {patch_dir}")
    expected_numbers = [f"{number:04d}" for number in range(1, len(patches) + 1)]
    actual_numbers = [patch.name[:4] for patch in patches]
    if actual_numbers != expected_numbers:
        raise RuntimeError(
            f"patch series must be contiguous from 0001, got {actual_numbers}"
        )
    return patches


def _apply_patch_series(ray_stock: Path, worktree: Path, patches: list[Path]) -> None:
    _run(
        "git",
        "-C",
        ray_stock,
        "worktree",
        "add",
        "--detach",
        worktree,
        RAY_STOCK_COMMIT,
    )
    for patch in patches:
        _git(worktree, "apply", "--check", str(patch), capture=False)
        _git(worktree, "apply", str(patch), capture=False)
    _git(worktree, "diff", "--check", capture=False)


def _production_overlay(worktree: Path) -> tuple[dict[str, bytes], set[str]]:
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(worktree),
            "diff",
            "--name-status",
            "--no-renames",
            "-z",
            "HEAD",
        ]
    ).split(b"\0")
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
        if not source_name.startswith("python/ray/") or not source_name.endswith(".py"):
            raise RuntimeError(
                "patched production files must be Python modules under python/ray: "
                f"{source_name}"
            )
        wheel_name = source_name.removeprefix("python/")
        status_kind = status[0]
        if status_kind == "D":
            removals.add(wheel_name)
        elif status_kind in {"A", "M"}:
            overlay[wheel_name] = (worktree / source_name).read_bytes()
        else:
            raise RuntimeError(f"unsupported patch status {status!r}: {source_name}")
    if not overlay and not removals:
        raise RuntimeError("patch series has no production Python wheel delta")
    return overlay, removals


def _write_wheel(
    stock_wheel: Path,
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
        with zipfile.ZipFile(stock_wheel) as source:
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
                raise RuntimeError("stock wheel must contain exactly one RECORD")
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
    """Verify archive integrity, patched bytes, and every top-level RECORD row."""

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
                raise RuntimeError(
                    f"derived wheel did not install patched bytes: {name}"
                )
        unexpected = names.intersection(removals)
        if unexpected:
            raise RuntimeError(f"derived wheel retained removed files: {unexpected}")


def build(args: argparse.Namespace) -> Path:
    project_root = args.project_root.resolve()
    ray_stock = project_root / "ray-stock"
    patch_dir = project_root / "ray-patches"
    stock_wheel = _find_stock_wheel(project_root, args.stock_wheel)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "wheels" / "patched"
    )
    patches = _validate_inputs(ray_stock, stock_wheel, patch_dir)
    patch_digest = hashlib.sha256()
    for patch in patches:
        patch_digest.update(patch.name.encode())
        patch_digest.update(b"\0")
        patch_digest.update(patch.read_bytes())

    with tempfile.TemporaryDirectory(prefix="ray-phase0-patches-") as temporary:
        worktree = Path(temporary) / "ray"
        try:
            _apply_patch_series(ray_stock, worktree, patches)
            overlay, removals = _production_overlay(worktree)
            epoch = int(
                _git(ray_stock, "show", "-s", "--format=%ct", RAY_STOCK_COMMIT).strip()
            )
            output = output_dir / stock_wheel.name
            _write_wheel(stock_wheel, output, overlay, removals, epoch)
            _validate_output_wheel(output, overlay, removals)
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

    print(f"ray_stock_commit={RAY_STOCK_COMMIT}")
    print(f"stock_wheel_sha256={_sha256(stock_wheel)}")
    print(f"patch_series_sha256={patch_digest.hexdigest()}")
    print(f"patched_wheel={output}")
    print(f"patched_wheel_sha256={_sha256(output)}")
    return output


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--stock-wheel", type=Path)
    parser.add_argument("--output-dir", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
