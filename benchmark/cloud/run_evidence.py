#!/usr/bin/env python3
"""Render, and only with --execute run, the direct-AWS G6 evidence matrix."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.evidence_statistics import (  # noqa: E402
    compare_randomized_runs,
    regression_gate,
    speedup_gate,
    summarize_samples,
)


CLOUD_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = CLOUD_ROOT / "cluster-g6.yaml.tmpl"
DEFAULT_CONFIG = CLOUD_ROOT / "local-config.json"
REMOTE_ENV_PYTHON = "/home/ray/.envs/rapids-25.12/bin/python"
PROJECT_TAG_KEY = "ray-gpu-evidence-project"
PROJECT_TAG_VALUE = "ray-data-gpu-admission"
RUN_TAG_KEY = "ray-gpu-evidence-run-id"
ARM_TAG_KEY = "ray-gpu-evidence-arm"
CLUSTER_TAG_KEY = "ray-gpu-evidence-cluster"
RAY_CLUSTER_TAG_KEY = "ray-cluster-name"

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
CLUSTER_NAME_RE = re.compile(r"^ray-gpu-adm-[a-z0-9][a-z0-9-]{0,110}$")
INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
VOLUME_ID_RE = re.compile(r"^vol-[0-9a-f]{8,17}$")
TOKEN_RE = re.compile(r"__[A-Z0-9_]+__")
PLACEHOLDER_RE = re.compile(
    r"(?:REPLACE(?:_|\b)|CHANGEME|YOUR[_ -]|<[^>]+>|__[A-Z0-9_]+__)", re.I
)
ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
PEM_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


@dataclass(frozen=True)
class Topology:
    name: str
    head_node_type: str
    head_gpus: int
    gpu_worker_min: int
    gpu_worker_max: int
    min_gpus: int
    max_gpus: int


@dataclass(frozen=True)
class Arm:
    name: str
    wheel_layer: str
    manifest: str
    admission_enabled: bool
    description: str


@dataclass(frozen=True)
class EvidenceCase:
    case_id: str
    cluster_name: str
    arm: str
    topology: str
    workload: str
    shuffle_ranks: int | str
    repetition: int
    map_actors_min: int
    map_actors_max: int
    role: str = "full-matrix"
    correctness_group: str | None = None
    randomization_block: int | None = None
    randomization_order: int | None = None


@dataclass(frozen=True)
class ScaleCaseSpec:
    role: str
    arm: str
    topology: str
    workload: str
    shuffle_ranks: int | str
    map_actors_min: int
    map_actors_max: int
    correctness_group: str


TOPOLOGIES = {
    item.name: item
    for item in (
        Topology("fixed-1", "ray.head.gpu", 1, 0, 0, 1, 1),
        Topology("fixed-2", "ray.head.gpu", 1, 1, 1, 2, 2),
        Topology("fixed-4", "ray.head.gpu", 1, 3, 3, 4, 4),
        Topology("autoscale-1-4", "ray.head.gpu", 1, 0, 3, 1, 4),
        Topology("autoscale-1-7", "ray.head.gpu", 1, 0, 6, 1, 7),
        Topology("autoscale-0-4-cold", "ray.head.cpu", 0, 0, 4, 0, 4),
    )
}

FULL_TOPOLOGIES = (
    "fixed-1",
    "fixed-2",
    "fixed-4",
    "autoscale-1-4",
    "autoscale-0-4-cold",
)

SCALE_CASE_SPECS = (
    ScaleCaseSpec(
        "scale-stock-workaround",
        "stock",
        "fixed-4",
        "incident",
        1,
        1,
        1,
        "scale-incident",
    ),
    ScaleCaseSpec(
        "scale-candidate-full",
        "minimal",
        "fixed-4",
        "incident",
        4,
        4,
        4,
        "scale-incident",
    ),
    ScaleCaseSpec(
        "normal-stock-equal",
        "stock",
        "fixed-4",
        "actor-only",
        "default",
        4,
        4,
        "scale-actor-only",
    ),
    ScaleCaseSpec(
        "normal-candidate-equal",
        "minimal",
        "fixed-4",
        "actor-only",
        "default",
        4,
        4,
        "scale-actor-only",
    ),
    ScaleCaseSpec(
        "fixed-elastic-candidate",
        "minimal",
        "fixed-4",
        "incident",
        4,
        1,
        4,
        "scale-incident",
    ),
    ScaleCaseSpec(
        "autoscale-elastic-candidate",
        "minimal",
        "autoscale-1-4",
        "incident",
        4,
        1,
        4,
        "scale-incident",
    ),
    ScaleCaseSpec(
        "autoscale-fixed-candidate",
        "minimal",
        "autoscale-1-4",
        "incident",
        4,
        4,
        4,
        "scale-incident",
    ),
    ScaleCaseSpec(
        "autoscale-more-stock",
        "stock",
        "autoscale-1-7",
        "incident",
        4,
        1,
        4,
        "scale-incident",
    ),
    ScaleCaseSpec(
        "autoscale-default-rank-candidate",
        "minimal",
        "autoscale-1-4",
        "incident",
        "default",
        1,
        4,
        "scale-incident",
    ),
)

SCALE_DEFAULT_ROWS = 4_000_000_000
SCALE_DEFAULT_BLOCKS = 1_024

STAGED_HARNESS_FILES = (
    "benchmark/__init__.py",
    "benchmark/ray_core_state.py",
    "benchmark/cloud/preflight.py",
    "benchmark/cloud/incident_workload.py",
    "benchmark/cloud/aws-runtime.requirements.txt",
    "environment/rapids-25.12-linux-64.explicit.txt",
)
STAGED_HARNESS_MANIFEST = "MANIFEST.json"

ARMS = {
    item.name: item
    for item in (
        Arm(
            "stock",
            "stock",
            "pins/stock-ray.json",
            False,
            "unmodified Ray scheduling",
        ),
        Arm(
            "pg-only",
            "pr-candidate",
            "pins/pr-candidate.json",
            False,
            "final candidate wheel with DAG admission disabled",
        ),
        Arm(
            "minimal",
            "pr-candidate",
            "pins/pr-candidate.json",
            True,
            "final minimal candidate with DAG admission enabled",
        ),
        Arm(
            "prototype",
            "prototype",
            "pins/prototype.json",
            True,
            "preserved full 5b9ac4 prototype with admission enabled",
        ),
    )
}

WORKLOADS = (
    "incident",
    "actor-only",
    "map-heavy",
    "shuffle-heavy",
    "forced-spill",
    "fan-in",
    "failure-cleanup",
)

SHUFFLE_WORKLOADS = frozenset(
    {"incident", "shuffle-heavy", "forced-spill", "failure-cleanup"}
)
EXPECTED_FAILURE_WORKLOADS = frozenset({"failure-cleanup"})
NORMAL_OBJECT_STORE_BYTES = 8 * 1024**3
SPILL_OBJECT_STORE_BYTES = 128 * 1024**2

REQUIRED_CONFIG_KEYS = {
    "region",
    "availability_zone",
    "ami_id",
    "subnet_id",
    "security_group_ids",
    "iam_instance_profile_arn",
    "aws_profile",
    "ec2_key_name",
    "ssh_private_key_path",
    "ssh_user",
    "docker_image",
    "gpu_instance_type",
    "cpu_head_instance_type",
    "root_volume_gib",
}


def _utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    entropy = hashlib.sha256(os.urandom(16)).hexdigest()[:6]
    return f"{stamp}-{entropy}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_json_bytes(document: object) -> bytes:
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode()


def _verify_staged_harness(directory: Path, expected_sha256: str) -> dict[str, object]:
    manifest_path = directory / STAGED_HARNESS_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid staged harness manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "files",
        "content_sha256",
    }:
        raise ValueError("staged harness manifest has an invalid schema")
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict):
        raise ValueError("staged harness manifest has an invalid schema")
    core = {"schema_version": 1, "files": files}
    content_sha256 = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    if content_sha256 != expected_sha256 or manifest["content_sha256"] != expected_sha256:
        raise ValueError("staged harness content digest does not match its manifest")
    total_size = manifest_path.stat().st_size
    for relative_name, entry in files.items():
        relative = Path(relative_name)
        if (
            not isinstance(relative_name, str)
            or not relative_name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_name
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "size_bytes"}
        ):
            raise ValueError(f"unsafe staged harness entry: {relative_name!r}")
        staged_file = directory / relative
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not staged_file.is_file()
            or staged_file.stat().st_size != size
            or _sha256_file(staged_file) != digest
        ):
            raise ValueError(f"staged harness file failed verification: {relative_name}")
        total_size += size
    return {
        "directory": str(directory.resolve()),
        "content_sha256": content_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": files,
    }


def stage_harness(case_directory: Path) -> dict[str, object]:
    """Copy only the immutable inputs mounted into one cloud evidence case."""
    files: dict[str, dict[str, object]] = {}
    for relative_name in STAGED_HARNESS_FILES:
        source = ROOT / relative_name
        if not source.is_file():
            raise ValueError(f"staged harness source is missing: {source}")
        files[relative_name] = {
            "sha256": _sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
    core = {"schema_version": 1, "files": files}
    content_sha256 = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    directory = case_directory / f"harness-{content_sha256[:16]}"
    manifest_path = directory / STAGED_HARNESS_MANIFEST
    if directory.exists():
        return _verify_staged_harness(directory, content_sha256)

    directory.mkdir(parents=True)
    for relative_name in STAGED_HARNESS_FILES:
        destination = directory / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_name, destination)
        destination.chmod(0o444)
    _atomic_json(
        manifest_path,
        {**core, "content_sha256": content_sha256},
    )
    manifest_path.chmod(0o444)
    return _verify_staged_harness(directory, content_sha256)


def _reject_credentials(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower().replace("-", "_")
            path_only = normalized == "ssh_private_key_path"
            forbidden = ("secret", "password", "access_key", "session_token", "credential")
            if not path_only and any(piece in normalized for piece in forbidden):
                raise ValueError(
                    f"credential material is forbidden in local config: {'.'.join((*path, key))}"
                )
            _reject_credentials(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_credentials(child, (*path, str(index)))
    elif isinstance(value, str):
        if ACCESS_KEY_RE.search(value) or PEM_RE.search(value):
            raise ValueError(
                f"credential-like value is forbidden in local config: {'.'.join(path)}"
            )


def _reject_placeholders(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_placeholders(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, (*path, str(index)))
    elif isinstance(value, str) and PLACEHOLDER_RE.search(value):
        raise ValueError(f"unresolved placeholder at {'.'.join(path)}")


def load_local_config(path: Path, *, for_execute: bool = False) -> dict[str, object]:
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(
            f"local config is missing: {path}; copy local-config.example.json and fill it"
        ) from error
    if not isinstance(document, dict):
        raise ValueError("local config must be one JSON object")
    _reject_credentials(document)
    _reject_placeholders(document)
    unknown = sorted(set(document) - REQUIRED_CONFIG_KEYS)
    missing = sorted(REQUIRED_CONFIG_KEYS - set(document))
    if unknown or missing:
        raise ValueError(f"local config schema mismatch: missing={missing}, unknown={unknown}")
    def require_string(key: str, pattern: str | None = None) -> str:
        value = document[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        if pattern and re.fullmatch(pattern, value) is None:
            raise ValueError(f"{key} has an invalid format")
        return value

    region = require_string("region", r"[a-z]{2}(?:-gov)?-[a-z]+-\d")
    zone = require_string("availability_zone", rf"{re.escape(region)}[a-z]")
    if not zone.startswith(region):
        raise ValueError("availability_zone must belong to region")
    require_string("ami_id", r"ami-[0-9a-f]{8}(?:[0-9a-f]{9})?")
    require_string("subnet_id", r"subnet-[0-9a-f]{8}(?:[0-9a-f]{9})?")
    groups = document["security_group_ids"]
    if (
        not isinstance(groups, list)
        or not groups
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"sg-[0-9a-f]{8}(?:[0-9a-f]{9})?", item) is None
            for item in groups
        )
    ):
        raise ValueError("security_group_ids must contain EC2 security-group IDs")
    require_string(
        "iam_instance_profile_arn",
        r"arn:aws(?:-us-gov|-cn)?:iam::[0-9]{12}:instance-profile/[A-Za-z0-9+=,.@_/-]+",
    )
    require_string("ssh_user", r"[A-Za-z_][A-Za-z0-9_-]{0,31}")
    image = require_string("docker_image", r"[^\s]+")
    if "py311" not in image:
        raise ValueError("docker_image must identify a CPython 3.11 image")
    if require_string("gpu_instance_type") != "g6.4xlarge":
        raise ValueError("gpu_instance_type must be exactly g6.4xlarge")
    require_string("cpu_head_instance_type", r"[cmr][0-9a-z]*\.[a-z0-9]+")
    volume = document["root_volume_gib"]
    if not isinstance(volume, int) or isinstance(volume, bool) or not 50 <= volume <= 2_000:
        raise ValueError("root_volume_gib must be an integer from 50 through 2000")
    for key in ("aws_profile", "ec2_key_name", "ssh_private_key_path"):
        value = document[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be null or a non-empty string")
    private_key = document["ssh_private_key_path"]
    if private_key is not None:
        private_path = Path(private_key).expanduser()
        if not private_path.is_absolute():
            raise ValueError("ssh_private_key_path must be absolute")
        if for_execute and not private_path.is_file():
            raise ValueError("ssh_private_key_path does not exist")
        document["ssh_private_key_path"] = str(private_path)
    return document


def render_cluster_config(
    template: str,
    *,
    config: Mapping[str, object],
    topology: Topology,
    cluster_name: str,
    run_id: str,
    arm: Arm,
    harness_directory: Path,
    wheel_directory: Path,
    wheel_file: str,
    expected_wheel_sha256: str,
    expected_ray_commit: str,
    source_provenance: str,
    expected_harness_sha256: str,
    object_store_memory_bytes: int = NORMAL_OBJECT_STORE_BYTES,
) -> str:
    validate_teardown_identity(cluster_name, run_id, arm.name)
    if object_store_memory_bytes < 80 * 1024**2:
        raise ValueError("object store memory must exceed Ray's minimum")
    if re.fullmatch(r"[0-9a-f]{64}", expected_wheel_sha256) is None:
        raise ValueError("expected wheel SHA-256 is malformed")
    if re.fullmatch(r"[0-9a-f]{40}", expected_ray_commit) is None:
        raise ValueError("expected Ray commit is malformed")
    if re.fullmatch(r"[0-9a-f]{64}", expected_harness_sha256) is None:
        raise ValueError("expected staged harness SHA-256 is malformed")
    if not source_provenance:
        raise ValueError("source provenance must be non-empty")
    key_name = config["ec2_key_name"]
    private_key = config["ssh_private_key_path"]
    replacements = {
        "__CLUSTER_NAME__": cluster_name,
        "__TOTAL_MAX_WORKERS__": str(topology.gpu_worker_max),
        "__REGION_JSON__": json.dumps(config["region"]),
        "__REGION_RAW__": str(config["region"]),
        "__AVAILABILITY_ZONE_JSON__": json.dumps(config["availability_zone"]),
        "__SSH_USER_JSON__": json.dumps(config["ssh_user"]),
        "__SSH_PRIVATE_KEY_LINE__": (
            f"  ssh_private_key: {json.dumps(private_key)}"
            if private_key
            else "  # ssh_private_key intentionally omitted"
        ),
        "__DOCKER_IMAGE_JSON__": json.dumps(config["docker_image"]),
        "__HEAD_RUN_OPTIONS_JSON__": json.dumps(
            ["--gpus=all"] if topology.head_gpus else []
        ),
        "__GPU_INSTANCE_TYPE_JSON__": json.dumps(config["gpu_instance_type"]),
        "__CPU_HEAD_INSTANCE_TYPE_JSON__": json.dumps(config["cpu_head_instance_type"]),
        "__AMI_ID_JSON__": json.dumps(config["ami_id"]),
        "__SUBNET_ID_JSON__": json.dumps(config["subnet_id"]),
        "__SECURITY_GROUP_IDS_JSON__": json.dumps(config["security_group_ids"]),
        "__IAM_INSTANCE_PROFILE_ARN_JSON__": json.dumps(config["iam_instance_profile_arn"]),
        "__GPU_KEY_NAME_LINE__": (
            f"      KeyName: {json.dumps(key_name)}"
            if key_name
            else "      # EC2 key name intentionally omitted"
        ),
        "__CPU_KEY_NAME_LINE__": (
            f"      KeyName: {json.dumps(key_name)}"
            if key_name
            else "      # EC2 key name intentionally omitted"
        ),
        "__ROOT_VOLUME_GIB__": str(config["root_volume_gib"]),
        "__RUN_ID_JSON__": json.dumps(run_id),
        "__ARM_JSON__": json.dumps(arm.name),
        "__TOPOLOGY_JSON__": json.dumps(topology.name),
        "__HEAD_NODE_TYPE__": topology.head_node_type,
        "__GPU_WORKER_MIN__": str(topology.gpu_worker_min),
        "__GPU_WORKER_MAX__": str(topology.gpu_worker_max),
        "__HARNESS_DIRECTORY_JSON__": json.dumps(str(harness_directory.resolve())),
        "__WHEEL_DIRECTORY_JSON__": json.dumps(str(wheel_directory.resolve())),
        "__RAY_WHEEL_FILE__": wheel_file,
        "__OBJECT_STORE_MEMORY_BYTES__": str(object_store_memory_bytes),
        "__EXPECTED_WHEEL_SHA256__": expected_wheel_sha256,
        "__EXPECTED_RAY_COMMIT__": expected_ray_commit,
        "__EXPECTED_HARNESS_SHA256__": expected_harness_sha256,
        "__EXPECTED_SOURCE_PROVENANCE_SHELL__": shlex.quote(source_provenance),
    }
    rendered = template
    for token, value in replacements.items():
        if token not in rendered:
            raise ValueError(f"cluster template is missing token {token}")
        rendered = rendered.replace(token, value)
    leftovers = sorted(set(TOKEN_RE.findall(rendered)))
    if leftovers:
        raise ValueError(f"unrendered cluster-template tokens: {leftovers}")
    try:
        import yaml

        parsed = yaml.safe_load(rendered)
    except Exception as error:
        raise ValueError(f"rendered cluster YAML is invalid: {error}") from error
    if parsed.get("cluster_name") != cluster_name:
        raise ValueError("rendered cluster name changed during YAML parsing")
    if parsed.get("head_node_type") != topology.head_node_type:
        raise ValueError("rendered head node type does not match topology")
    return rendered


def validate_teardown_identity(cluster_name: str, run_id: str, arm: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe run ID: {run_id!r}")
    if arm not in ARMS:
        raise ValueError(f"unknown evidence arm: {arm!r}")
    if not CLUSTER_NAME_RE.fullmatch(cluster_name):
        raise ValueError(f"unsafe cluster name for teardown: {cluster_name!r}")
    if run_id not in cluster_name or f"-{arm}-" not in cluster_name:
        raise ValueError("cluster name does not bind the requested run and arm")


def _tags(resource: Mapping[str, object]) -> dict[str, str]:
    return {
        str(item["Key"]): str(item["Value"])
        for item in resource.get("Tags", ())
        if isinstance(item, Mapping) and "Key" in item and "Value" in item
    }


class InstanceTimeline:
    """Checkpoint exact-cluster EC2 state without affecting Ray autoscaling."""

    def __init__(
        self,
        *,
        client: object,
        cluster_name: str,
        run_id: str,
        arm: str,
        output: Path,
        poll_seconds: float = 5.0,
    ):
        validate_teardown_identity(cluster_name, run_id, arm)
        if poll_seconds <= 0:
            raise ValueError("EC2 timeline poll interval must be positive")
        self._client = client
        self._filters = _instance_filters(cluster_name, run_id, arm)
        self._cluster_name = cluster_name
        self._run_id = run_id
        self._arm = arm
        self._output = output
        self._poll_seconds = poll_seconds
        self._started = time.monotonic()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._stop = threading.Event()
        self._observations: list[dict[str, object]] = []
        self._errors: list[dict[str, str]] = []
        self._data_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread_started = False
        self._report: dict[str, object] | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._report is not None:
                raise RuntimeError("cannot restart a stopped EC2 timeline")
            if self._thread_started:
                return
            self._sample()
            self._thread.start()
            self._thread_started = True

    def stop(self) -> dict[str, object]:
        with self._lifecycle_lock:
            if self._report is not None:
                return self._report
            self._stop.set()
            if self._thread_started:
                self._thread.join(timeout=max(10.0, self._poll_seconds * 2))
            if self._thread.is_alive():
                self._append_error(
                    RuntimeError("EC2 timeline sampler did not stop")
                )
            else:
                # Capture the final cluster state only after the polling
                # thread has exited, so stop never races a second AWS call.
                self._sample()
            with self._data_lock:
                report = {
                    "cluster_name": self._cluster_name,
                    "run_id": self._run_id,
                    "arm": self._arm,
                    "poll_seconds": self._poll_seconds,
                    "started_at": self._started_at,
                    "observations": list(self._observations),
                    "errors": list(self._errors),
                }
            _atomic_json(self._output, report)
            self._report = report
            return report

    def _append_error(self, error: BaseException) -> None:
        with self._data_lock:
            self._errors.append(
                {"type": type(error).__name__, "message": str(error)}
            )

    def _sample(self) -> None:
        with self._sample_lock:
            try:
                response = self._client.describe_instances(Filters=self._filters)
                instances = [
                    instance
                    for reservation in response.get("Reservations", ())
                    for instance in reservation.get("Instances", ())
                ]
                records = []
                for instance in instances:
                    _validate_returned_target(
                        instance,
                        cluster_name=self._cluster_name,
                        run_id=self._run_id,
                        arm=self._arm,
                    )
                    launch_time = instance.get("LaunchTime")
                    records.append(
                        {
                            "instance_id": instance.get("InstanceId"),
                            "instance_type": instance.get("InstanceType"),
                            "state": dict(instance.get("State") or {}).get("Name"),
                            "launch_time": (
                                launch_time.isoformat()
                                if hasattr(launch_time, "isoformat")
                                else str(launch_time) if launch_time else None
                            ),
                            "availability_zone": dict(
                                instance.get("Placement") or {}
                            ).get("AvailabilityZone"),
                            "private_ip_address": instance.get("PrivateIpAddress"),
                            "ray_node_kind": _tags(instance).get("ray-node-kind"),
                        }
                    )
                observation = {
                    "elapsed_s": time.monotonic() - self._started,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "instances": sorted(
                        records, key=lambda item: str(item["instance_id"])
                    ),
                }
                with self._data_lock:
                    self._observations.append(observation)
            except Exception as error:
                self._append_error(error)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            self._sample()


def timeline_instance_ids(report: Mapping[str, object]) -> list[str]:
    """Return only validated instance IDs observed before ``ray down``."""
    observations = report.get("observations")
    if not isinstance(observations, list):
        raise ValueError("EC2 timeline observations must be a list")
    instance_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise ValueError("EC2 timeline observation must be an object")
        instances = observation.get("instances")
        if not isinstance(instances, list):
            raise ValueError("EC2 timeline instances must be a list")
        for instance in instances:
            if not isinstance(instance, Mapping):
                raise ValueError("EC2 timeline instance must be an object")
            instance_id = instance.get("instance_id")
            if not isinstance(instance_id, str) or not INSTANCE_ID_RE.fullmatch(
                instance_id
            ):
                raise ValueError(f"EC2 timeline contains malformed ID: {instance_id!r}")
            instance_ids.add(instance_id)
    return sorted(instance_ids)


def _validate_returned_target(
    resource: Mapping[str, object], *, cluster_name: str, run_id: str, arm: str
) -> None:
    tags = _tags(resource)
    expected = {
        PROJECT_TAG_KEY: PROJECT_TAG_VALUE,
        RUN_TAG_KEY: run_id,
        ARM_TAG_KEY: arm,
        CLUSTER_TAG_KEY: cluster_name,
    }
    mismatches = {
        key: {"expected": value, "actual": tags.get(key)}
        for key, value in expected.items()
        if tags.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"refusing teardown target with mismatched tags: {mismatches}")


def _instance_filters(cluster_name: str, run_id: str, arm: str) -> list[dict]:
    return [
        {"Name": f"tag:{RAY_CLUSTER_TAG_KEY}", "Values": [cluster_name]},
        {"Name": f"tag:{PROJECT_TAG_KEY}", "Values": [PROJECT_TAG_VALUE]},
        {"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]},
        {"Name": f"tag:{ARM_TAG_KEY}", "Values": [arm]},
        {"Name": f"tag:{CLUSTER_TAG_KEY}", "Values": [cluster_name]},
        {
            "Name": "instance-state-name",
            "Values": ["pending", "running", "stopping", "stopped", "shutting-down"],
        },
    ]


def terminate_cluster_backstop(
    cluster_name: str,
    *,
    run_id: str,
    arm: str,
    client: object,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 5.0,
    stable_empty_observations: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Terminate an exact-tag instance set until three polls are empty."""
    validate_teardown_identity(cluster_name, run_id, arm)
    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("teardown polling bounds must be positive")
    if stable_empty_observations < 3:
        raise ValueError("teardown requires at least three stable empty polls")
    filters = _instance_filters(cluster_name, run_id, arm)
    deadline = monotonic() + timeout_seconds
    requested: set[str] = set()
    empty = 0
    observations = 0
    current: list[str] = []
    while True:
        response = client.describe_instances(Filters=filters)
        observations += 1
        instances = [
            instance
            for reservation in response.get("Reservations", ())
            for instance in reservation.get("Instances", ())
        ]
        current = []
        for instance in instances:
            _validate_returned_target(
                instance, cluster_name=cluster_name, run_id=run_id, arm=arm
            )
            if _tags(instance).get(RAY_CLUSTER_TAG_KEY) != cluster_name:
                raise RuntimeError(
                    "refusing instance without the exact Ray cluster-name tag"
                )
            instance_id = str(instance.get("InstanceId", ""))
            if not INSTANCE_ID_RE.fullmatch(instance_id):
                raise RuntimeError(f"refusing malformed instance ID: {instance_id!r}")
            current.append(instance_id)
        current.sort()
        if current:
            empty = 0
            new_ids = sorted(set(current) - requested)
            if new_ids:
                client.terminate_instances(InstanceIds=new_ids)
                requested.update(new_ids)
        else:
            empty += 1
            if empty >= stable_empty_observations:
                return {
                    "cluster_name": cluster_name,
                    "terminated_instance_ids": sorted(requested),
                    "poll_observations": observations,
                    "stable_empty_observations": empty,
                }
        now = monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"instance teardown timed out; current={current}, requested={sorted(requested)}"
            )
        sleep(min(poll_interval_seconds, deadline - now))


def verify_no_tagged_volumes(
    cluster_name: str,
    *,
    run_id: str,
    arm: str,
    client: object,
    allowed_instance_ids: Iterable[str] = (),
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 5.0,
    stable_empty_observations: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    validate_teardown_identity(cluster_name, run_id, arm)
    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("volume polling bounds must be positive")
    if stable_empty_observations < 3:
        raise ValueError("volume verification requires three stable empty polls")
    filters = [
        {"Name": f"tag:{PROJECT_TAG_KEY}", "Values": [PROJECT_TAG_VALUE]},
        {"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]},
        {"Name": f"tag:{ARM_TAG_KEY}", "Values": [arm]},
        {"Name": f"tag:{CLUSTER_TAG_KEY}", "Values": [cluster_name]},
    ]
    deadline = monotonic() + timeout_seconds
    allowed_attachments = set(allowed_instance_ids)
    if any(not INSTANCE_ID_RE.fullmatch(item) for item in allowed_attachments):
        raise ValueError("allowed volume attachment IDs must be EC2 instance IDs")
    deletion_requested: set[str] = set()
    empty = 0
    observations = 0
    while True:
        response = client.describe_volumes(Filters=filters)
        observations += 1
        volumes = list(response.get("Volumes", ()))
        current = []
        for volume in volumes:
            _validate_returned_target(
                volume, cluster_name=cluster_name, run_id=run_id, arm=arm
            )
            volume_id = str(volume.get("VolumeId", ""))
            if not VOLUME_ID_RE.fullmatch(volume_id):
                raise RuntimeError(f"refusing malformed volume ID: {volume_id!r}")
            current.append(volume_id)
            attachments = list(volume.get("Attachments", ()))
            attachment_ids = {
                str(attachment.get("InstanceId", "")) for attachment in attachments
            }
            if any(not INSTANCE_ID_RE.fullmatch(item) for item in attachment_ids):
                raise RuntimeError(
                    f"refusing volume {volume_id} with malformed attachment IDs"
                )
            unexpected = attachment_ids - allowed_attachments
            if unexpected:
                raise RuntimeError(
                    f"refusing volume {volume_id} attached outside the exact cluster: "
                    f"{sorted(unexpected)}"
                )
            if not attachments and volume_id not in deletion_requested:
                client.delete_volume(VolumeId=volume_id)
                deletion_requested.add(volume_id)
        if current:
            empty = 0
        else:
            empty += 1
            if empty >= stable_empty_observations:
                return {
                    "cluster_name": cluster_name,
                    "poll_observations": observations,
                    "stable_empty_observations": empty,
                    "remaining_volume_ids": [],
                    "deleted_volume_ids": sorted(deletion_requested),
                }
        now = monotonic()
        if now >= deadline:
            raise RuntimeError(f"tagged EBS volumes still exist: {sorted(current)}")
        sleep(min(poll_interval_seconds, deadline - now))


def _csv_choices(value: str, choices: Iterable[str], label: str) -> tuple[str, ...]:
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(choices))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(f"invalid {label}: selected={selected}, unknown={unknown}")
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"duplicate {label} entries are not allowed")
    return selected


def _rank_csv(value: str) -> tuple[int | str, ...]:
    ranks: list[int | str] = []
    for part in (item.strip() for item in value.split(",")):
        if not part:
            continue
        if part == "default":
            rank: int | str = part
        else:
            try:
                rank = int(part)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "ranks must be positive integers or 'default'"
                ) from error
            if rank < 1:
                raise argparse.ArgumentTypeError("ranks must be positive")
        ranks.append(rank)
    if not ranks or len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("ranks must be unique and non-empty")
    return tuple(ranks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("full", "scale"),
        default="full",
        help=(
            "full preserves the exhaustive factor matrix; scale selects the "
            "paired fixed/autoscaling merge-evidence campaign"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cluster-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--run-id", default=_utc_run_id())
    parser.add_argument(
        "--arms",
        type=lambda value: _csv_choices(value, ARMS, "arms"),
        default=tuple(ARMS),
    )
    parser.add_argument(
        "--topologies",
        type=lambda value: _csv_choices(value, TOPOLOGIES, "topologies"),
        default=FULL_TOPOLOGIES,
    )
    parser.add_argument(
        "--workloads",
        type=lambda value: _csv_choices(value, WORKLOADS, "workloads"),
        default=WORKLOADS,
    )
    parser.add_argument("--ranks", type=_rank_csv, default=(1, 2, 3, 4, "default"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--rows", type=int, default=16_000_000)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--workload-timeout-seconds", type=int, default=1_800)
    parser.add_argument("--teardown-timeout-seconds", type=int, default=900)
    parser.add_argument("--teardown-poll-seconds", type=float, default=5.0)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--max-execute-cases",
        type=int,
        default=100,
        help="refuse a larger paid campaign unless --confirm-large-run matches --run-id",
    )
    parser.add_argument("--confirm-large-run")
    parser.add_argument("--ray-cli", type=Path, default=ROOT / ".venv" / "bin" / "ray")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch AWS clusters; without this flag only local plans are rendered",
    )
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if args.profile == "scale":
        owned_factors = ("--arms", "--topologies", "--workloads", "--ranks")
        overridden = [
            option
            for option in owned_factors
            if any(token == option or token.startswith(option + "=") for token in raw_argv)
        ]
        if overridden:
            parser.error(
                "--profile scale owns its comparison factors; remove "
                + ", ".join(overridden)
            )
        args.arms = ("stock", "minimal")
        args.topologies = ("fixed-4", "autoscale-1-4", "autoscale-1-7")
        args.workloads = ("incident", "actor-only")
        args.ranks = (1, 4, "default")
        if not any(
            token == "--rows" or token.startswith("--rows=") for token in raw_argv
        ):
            args.rows = SCALE_DEFAULT_ROWS
        if not any(
            token == "--blocks" or token.startswith("--blocks=")
            for token in raw_argv
        ):
            args.blocks = SCALE_DEFAULT_BLOCKS
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id must be 1-40 lowercase letters, digits, or hyphens")
    for name in (
        "repetitions",
        "rows",
        "blocks",
        "groups",
        "batch_size",
        "workload_timeout_seconds",
        "teardown_timeout_seconds",
        "max_execute_cases",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sample_interval_seconds <= 0 or args.teardown_poll_seconds <= 0:
        parser.error("sampling and teardown poll intervals must be positive")
    args.config = args.config.expanduser().resolve()
    args.cluster_template = args.cluster_template.expanduser().resolve()
    args.ray_cli = args.ray_cli.expanduser().resolve()
    if args.artifact_root is None:
        args.artifact_root = ROOT / "benchmark" / "results" / "cloud" / args.run_id
    else:
        args.artifact_root = args.artifact_root.expanduser().resolve()
    return args


def _wheel_admission_fields(wheel_path: Path) -> list[str] | None:
    """Read the private grant schema without importing or installing the wheel."""
    with zipfile.ZipFile(wheel_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith("ray/data/_internal/execution/resource_admission.py")
        ]
        if not names:
            return None
        if len(names) != 1:
            raise ValueError(f"wheel has ambiguous admission modules: {names}")
        module = ast.parse(archive.read(names[0]).decode())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "ResourceAdmissionGrant":
            return [
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            ]
    raise ValueError("admission module lacks ResourceAdmissionGrant")


def _artifact_expected_fields(arm_name: str) -> list[str] | None:
    if arm_name == "stock":
        return None
    if arm_name == "prototype":
        return ["max_units"]
    return ["max_units", "may_submit"]


def _arm_artifact(arm: Arm, *, require: bool) -> dict[str, object]:
    manifest_path = ROOT / arm.manifest
    stock_manifest = json.loads((ROOT / "pins" / "stock-ray.json").read_text())
    wheel_file = stock_manifest["wheel"]
    manifest = None
    issues: list[str] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path} must contain one JSON object")
        wheel_file = manifest.get("wheel", wheel_file)
        commit = manifest.get("local_commit") or manifest.get("commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            issues.append("manifest source commit is missing or malformed")
        expected_sha_value = manifest.get("wheel_sha256")
        if (
            not isinstance(expected_sha_value, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha_value) is None
        ):
            issues.append("manifest wheel SHA-256 is missing or malformed")
        size = manifest.get("wheel_size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            issues.append("manifest wheel size is missing or malformed")
        if arm.name != "stock" and manifest.get("base_commit") != stock_manifest.get("commit"):
            issues.append("manifest base commit does not match the stock control")
        if manifest.get("python", "3.11") != "3.11":
            issues.append("manifest is not a CPython 3.11 artifact")
        if manifest.get("platform", "linux_x86_64") != "linux_x86_64":
            issues.append("manifest is not a linux_x86_64 artifact")
    if not isinstance(wheel_file, str) or Path(wheel_file).name != wheel_file:
        raise ValueError(f"unsafe wheel filename in {manifest_path}: {wheel_file!r}")
    wheel_path = ROOT / "wheels" / arm.wheel_layer / str(wheel_file)
    available = manifest_path.is_file() and wheel_path.is_file()
    expected_sha = manifest.get("wheel_sha256") if manifest else None
    actual_sha = _sha256_file(wheel_path) if wheel_path.is_file() else None
    if available and expected_sha != actual_sha:
        issues.append("wheel digest does not match its manifest")
    if available and manifest.get("wheel_size_bytes") != wheel_path.stat().st_size:
        issues.append("wheel size does not match its manifest")
    grant_fields = _wheel_admission_fields(wheel_path) if wheel_path.is_file() else None
    expected_fields = _artifact_expected_fields(arm.name)
    if available and grant_fields != expected_fields:
        issues.append(
            f"grant schema is {grant_fields!r}, expected {expected_fields!r}"
        )
    verified = bool(available and not issues)
    if require and not available:
        raise ValueError(
            f"{arm.name} evidence artifact is missing: manifest={manifest_path}, wheel={wheel_path}"
        )
    if require and not verified:
        raise ValueError(f"{arm.name} artifact is not executable: {issues}")
    return {
        "manifest_path": str(manifest_path),
        "wheel_path": str(wheel_path),
        "wheel_directory": str(wheel_path.parent),
        "wheel_file": str(wheel_file),
        "available": available,
        "digest_verified": verified,
        "ready": verified,
        "validation_issues": issues,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "source_commit": (
            manifest.get("local_commit") or manifest.get("commit")
            if manifest
            else None
        ),
        "source_tree": manifest.get("source_tree") if manifest else None,
        "base_commit": manifest.get("base_commit") if manifest else None,
        "grant_fields": grant_fields,
    }


def _validate_artifact_separation(
    artifacts: Mapping[str, Mapping[str, object]], *, require: bool
) -> list[str]:
    issues: list[str] = []
    candidate = artifacts.get("minimal") or artifacts.get("pg-only")
    prototype = artifacts.get("prototype")
    stock = artifacts.get("stock")
    if candidate and prototype:
        for field in ("actual_sha256", "source_commit", "source_tree"):
            value = candidate.get(field)
            if value is not None and value == prototype.get(field):
                issues.append(f"final candidate and prototype share {field}: {value}")
    if candidate and stock and candidate.get("actual_sha256") == stock.get("actual_sha256"):
        issues.append("final candidate wheel is byte-identical to stock")
    if require and issues:
        raise ValueError("evidence artifacts are not independent: " + "; ".join(issues))
    return issues


def build_cases(
    *,
    run_id: str,
    arms: Sequence[str],
    topologies: Sequence[str],
    workloads: Sequence[str],
    ranks: Sequence[int | str],
    repetitions: int,
) -> list[EvidenceCase]:
    cases = []
    sequence = 0
    for topology_name in topologies:
        topology = TOPOLOGIES[topology_name]
        for workload in workloads:
            workload_ranks: Sequence[int | str] = (
                ranks if workload in SHUFFLE_WORKLOADS else ("default",)
            )
            for rank in workload_ranks:
                if isinstance(rank, int) and rank > topology.max_gpus:
                    continue
                for repetition in range(1, repetitions + 1):
                    for arm_name in arms:
                        sequence += 1
                        case_id = (
                            f"{sequence:03d}-{arm_name}-{topology_name}-{workload}"
                            f"-r{'d' if rank == 'default' else rank}-n{repetition}"
                        )
                        cluster_name = f"ray-gpu-adm-{run_id}-{case_id}"
                        validate_teardown_identity(cluster_name, run_id, arm_name)
                        cases.append(
                            EvidenceCase(
                                case_id=case_id,
                                cluster_name=cluster_name,
                                arm=arm_name,
                                topology=topology_name,
                                workload=workload,
                                shuffle_ranks=rank,
                                repetition=repetition,
                                map_actors_min=topology.max_gpus,
                                map_actors_max=topology.max_gpus,
                            )
                        )
    if not cases:
        raise ValueError("rank/topology selection produced no evidence cases")
    random.Random(
        int(hashlib.sha256(run_id.encode()).hexdigest()[:16], 16)
    ).shuffle(cases)
    return cases


def build_scale_cases(*, run_id: str, repetitions: int) -> list[EvidenceCase]:
    """Build deterministic repetition blocks for the focused paid campaign."""

    cases: list[EvidenceCase] = []
    sequence = 0
    seed = int(hashlib.sha256(run_id.encode()).hexdigest()[:16], 16)
    for repetition in range(1, repetitions + 1):
        block = list(SCALE_CASE_SPECS)
        random.Random(seed + repetition).shuffle(block)
        for order, spec in enumerate(block, start=1):
            sequence += 1
            rank_label = "d" if spec.shuffle_ranks == "default" else spec.shuffle_ranks
            case_id = (
                f"{sequence:03d}-{spec.arm}-{spec.topology}-{spec.workload}"
                f"-r{rank_label}-n{repetition}"
            )
            cluster_name = f"ray-gpu-adm-{run_id}-{case_id}"
            validate_teardown_identity(cluster_name, run_id, spec.arm)
            cases.append(
                EvidenceCase(
                    case_id=case_id,
                    cluster_name=cluster_name,
                    arm=spec.arm,
                    topology=spec.topology,
                    workload=spec.workload,
                    shuffle_ranks=spec.shuffle_ranks,
                    repetition=repetition,
                    map_actors_min=spec.map_actors_min,
                    map_actors_max=spec.map_actors_max,
                    role=spec.role,
                    correctness_group=spec.correctness_group,
                    randomization_block=repetition,
                    randomization_order=order,
                )
            )
    return cases


def _scale_campaign_budget(
    cases: Sequence[EvidenceCase], args: argparse.Namespace
) -> dict[str, object]:
    """Return price-neutral bounds that can be audited before paid execution."""

    workload_gpu_seconds_min = sum(
        TOPOLOGIES[case.topology].min_gpus * args.workload_timeout_seconds
        for case in cases
    )
    workload_gpu_seconds_max = sum(
        TOPOLOGIES[case.topology].max_gpus * args.workload_timeout_seconds
        for case in cases
    )
    rows_per_block = math.ceil(args.rows / args.blocks)
    maximum_samples = (
        math.ceil(args.workload_timeout_seconds / args.sample_interval_seconds) + 3
    )
    return {
        "cases": len(cases),
        "workload_timeout_scope": (
            "remote driver startup, materialization, correctness validation, "
            "and final evidence checkpoint"
        ),
        "workload_window_gpu_node_hours_at_minimum_capacity": (
            workload_gpu_seconds_min / 3_600
        ),
        "workload_window_gpu_node_hours_at_maximum_capacity": (
            workload_gpu_seconds_max / 3_600
        ),
        "capacity_bounds_exclude_cluster_setup_preflight_cleanup": True,
        "resource_sample_upper_bound_per_case": maximum_samples,
        "resource_samples_contain_bounded_core_state_records": True,
        "artifact_size_is_data_dependent": True,
        "logical_dataset_lower_bounds": {
            "rows": args.rows,
            "blocks": args.blocks,
            "maximum_rows_per_block": rows_per_block,
            "source_id_column_bytes": args.rows * 8,
            "keyed_id_and_key_columns_bytes": args.rows * 16,
            "python_cudf_and_shuffle_overhead_excluded": True,
        },
        "object_store_bytes_per_node": NORMAL_OBJECT_STORE_BYTES,
    }


def _git_provenance() -> dict[str, object]:
    def capture(*command: str) -> str:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    return {
        "commit": capture("git", "rev-parse", "HEAD"),
        "dirty": bool(capture("git", "status", "--porcelain")),
    }


def build_plan(
    args: argparse.Namespace,
    *,
    config: Mapping[str, object],
    template: str,
) -> dict[str, object]:
    if args.profile == "scale" and config["gpu_instance_type"] != "g6.4xlarge":
        raise ValueError(
            "--profile scale requires gpu_instance_type=g6.4xlarge so every "
            "comparison uses identical one-L4 nodes"
        )
    cases = (
        build_scale_cases(run_id=args.run_id, repetitions=args.repetitions)
        if args.profile == "scale"
        else build_cases(
            run_id=args.run_id,
            arms=args.arms,
            topologies=args.topologies,
            workloads=args.workloads,
            ranks=args.ranks,
            repetitions=args.repetitions,
        )
    )
    selected_arms = tuple(dict.fromkeys(case.arm for case in cases))
    artifacts = {
        arm_name: _arm_artifact(ARMS[arm_name], require=args.execute)
        for arm_name in selected_arms
    }
    validation_artifacts = dict(artifacts)
    if any(name in selected_arms for name in ("minimal", "pg-only")):
        if args.profile == "full":
            validation_artifacts.setdefault(
                "prototype", _arm_artifact(ARMS["prototype"], require=args.execute)
            )
        validation_artifacts.setdefault(
            "stock", _arm_artifact(ARMS["stock"], require=args.execute)
        )
    artifact_separation_issues = _validate_artifact_separation(
        validation_artifacts, require=args.execute
    )
    config_fingerprint = hashlib.sha256(
        json.dumps(config, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    rendered_cases = []
    for case in cases:
        case_directory = args.artifact_root / "cases" / case.case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        cluster_path = case_directory / "cluster.yaml"
        artifact = artifacts[case.arm]
        staged_harness = stage_harness(case_directory)
        rendered = render_cluster_config(
            template,
            config=config,
            topology=TOPOLOGIES[case.topology],
            cluster_name=case.cluster_name,
            run_id=args.run_id,
            arm=ARMS[case.arm],
            harness_directory=Path(str(staged_harness["directory"])),
            wheel_directory=Path(str(artifact["wheel_directory"])),
            wheel_file=str(artifact["wheel_file"]),
            expected_wheel_sha256=str(artifact["actual_sha256"]),
            expected_ray_commit=str(
                artifact["base_commit"] or artifact["source_commit"]
            ),
            source_provenance=str(
                artifact["source_tree"] or artifact["source_commit"]
            ),
            expected_harness_sha256=str(staged_harness["content_sha256"]),
            object_store_memory_bytes=(
                SPILL_OBJECT_STORE_BYTES
                if case.workload == "forced-spill"
                else NORMAL_OBJECT_STORE_BYTES
            ),
        )
        cluster_path.write_text(rendered)
        rendered_cases.append(
            {
                **asdict(case),
                "cluster_config": str(cluster_path),
                "cluster_config_sha256": _sha256_file(cluster_path),
                "map_actors_per_stage": (
                    case.map_actors_min
                    if case.map_actors_min == case.map_actors_max
                    else None
                ),
                "map_actor_pool": {
                    "min_size": case.map_actors_min,
                    "max_size": case.map_actors_max,
                    "constructor": (
                        "size"
                        if case.map_actors_min == case.map_actors_max
                        else "min_size,max_size"
                    ),
                },
                "object_store_memory_bytes": (
                    SPILL_OBJECT_STORE_BYTES
                    if case.workload == "forced-spill"
                    else NORMAL_OBJECT_STORE_BYTES
                ),
                "expected_wheel_sha256": artifact["actual_sha256"],
                "expected_ray_commit": artifact["base_commit"]
                or artifact["source_commit"],
                "source_provenance": artifact["source_tree"] or artifact["source_commit"],
                "staged_harness": staged_harness,
                "expected_harness_sha256": staged_harness["content_sha256"],
                "remote_wheel_path": f"/home/ray/wheels/{artifact['wheel_file']}",
            }
        )
    return {
        "schema_version": 1,
        "mode": "execute" if args.execute else "dry-run",
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_provenance(),
        "environment": {
            "rapids_explicit_lock": "environment/rapids-25.12-linux-64.explicit.txt",
            "rapids_explicit_sha256": _sha256_file(
                ROOT / "environment" / "rapids-25.12-linux-64.explicit.txt"
            ),
            "aws_runtime_requirements": "benchmark/cloud/aws-runtime.requirements.txt",
            "aws_runtime_requirements_sha256": _sha256_file(
                CLOUD_ROOT / "aws-runtime.requirements.txt"
            ),
            "python": "3.11",
        },
        "harness": {
            "cluster_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "runner_sha256": _sha256_file(Path(__file__)),
            "workload_sha256": _sha256_file(CLOUD_ROOT / "incident_workload.py"),
            "preflight_sha256": _sha256_file(CLOUD_ROOT / "preflight.py"),
            "core_gcs_state_reader_sha256": _sha256_file(
                ROOT / "benchmark" / "ray_core_state.py"
            ),
            "staged_files": list(STAGED_HARNESS_FILES),
            "live_repository_is_not_mounted": True,
        },
        "infrastructure": {
            "region": config["region"],
            "availability_zone": config["availability_zone"],
            "gpu_instance_type": config["gpu_instance_type"],
            "cpu_head_instance_type": config["cpu_head_instance_type"],
            "config_sha256": config_fingerprint,
            "credential_source": (
                "named AWS profile" if config["aws_profile"] else "ambient AWS provider chain"
            ),
        },
        "matrix": {
            "profile": args.profile,
            "arms": {name: asdict(ARMS[name]) for name in selected_arms},
            "topologies": {name: asdict(TOPOLOGIES[name]) for name in args.topologies},
            "workloads": list(args.workloads),
            "ranks": list(args.ranks),
            "repetitions": args.repetitions,
            "invalid_rank_topology_pairs_are_skipped": True,
            "no_shuffle_workloads_run_once_with_default_rank": True,
            "case_order_seed": hashlib.sha256(args.run_id.encode()).hexdigest()[:16],
            "repetition_blocked_randomization": args.profile == "scale",
        },
        "workload": {
            "rows": args.rows,
            "blocks": args.blocks,
            "groups": args.groups,
            "batch_size": args.batch_size,
            "timeout_seconds": args.workload_timeout_seconds,
            "sample_interval_seconds": args.sample_interval_seconds,
        },
        "artifacts": artifacts,
        "artifact_validation": {
            "comparison_artifacts": validation_artifacts,
            "separation_issues": artifact_separation_issues,
        },
        "cases": rendered_cases,
        "scale_profile": (
            {
                "fixed_four_speedup_gate": {
                    "baseline_role": "scale-stock-workaround",
                    "candidate_role": "scale-candidate-full",
                    "minimum_relative_time_reduction": 0.10,
                },
                "equal_shape_regression_gate": {
                    "baseline_role": "normal-stock-equal",
                    "candidate_role": "normal-candidate-equal",
                    "maximum_regression": 0.05,
                },
                "autoscale_more_control": {
                    "stock_topology": "autoscale-1-7",
                    "candidate_topology": "autoscale-1-4",
                    "maximum_gpu_capacity_ratio": 7 / 4,
                    "extra_gpu_node_ceiling": 3,
                    "reason": (
                        "four eager shuffle actors plus three elastic map-pool "
                        "minimums can require seven simultaneous GPUs"
                    ),
                },
                "campaign_budget": _scale_campaign_budget(cases, args),
            }
            if args.profile == "scale"
            else None
        ),
    }


def _aws_environment(config: Mapping[str, object]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["AWS_REGION"] = str(config["region"])
    environment["AWS_DEFAULT_REGION"] = str(config["region"])
    if config["aws_profile"]:
        environment["AWS_PROFILE"] = str(config["aws_profile"])
    return environment


def aws_preflight(
    config: Mapping[str, object], *, run_id: str
) -> tuple[object, dict[str, object]]:
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError(
            "--execute requires boto3 in the launcher environment (Ray AWS provider extra)"
        ) from error
    aws_versions = {
        name: metadata.version(name) for name in ("boto3", "botocore")
    }
    expected_versions = {"boto3": "1.42.90", "botocore": "1.42.90"}
    if aws_versions != expected_versions:
        raise RuntimeError(
            "launcher AWS provider versions do not match "
            f"aws-runtime.requirements.txt: {aws_versions}"
        )
    session = boto3.Session(
        profile_name=config["aws_profile"] or None,
        region_name=str(config["region"]),
    )
    if session.get_credentials() is None:
        raise RuntimeError("AWS provider chain returned no credentials")
    session.client("sts").get_caller_identity()
    ec2 = session.client("ec2")
    image = ec2.describe_images(ImageIds=[config["ami_id"]]).get("Images", [])
    if len(image) != 1 or image[0].get("State") != "available":
        raise RuntimeError("configured AMI is not uniquely available")
    if image[0].get("Architecture") != "x86_64":
        raise RuntimeError("configured AMI is not x86_64")
    subnet = ec2.describe_subnets(SubnetIds=[config["subnet_id"]]).get("Subnets", [])
    if len(subnet) != 1 or subnet[0].get("AvailabilityZone") != config["availability_zone"]:
        raise RuntimeError("configured subnet is not in the requested availability zone")
    groups = ec2.describe_security_groups(
        GroupIds=list(config["security_group_ids"])
    ).get("SecurityGroups", [])
    if len(groups) != len(config["security_group_ids"]):
        raise RuntimeError("one or more configured security groups are unavailable")
    if any(group.get("VpcId") != subnet[0].get("VpcId") for group in groups):
        raise RuntimeError("security groups and subnet are not in one VPC")
    if config["ec2_key_name"]:
        ec2.describe_key_pairs(KeyNames=[config["ec2_key_name"]])
    profile_name = str(config["iam_instance_profile_arn"]).rsplit("/", 1)[-1]
    session.client("iam").get_instance_profile(InstanceProfileName=profile_name)
    offerings = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[
            {"Name": "location", "Values": [config["availability_zone"]]},
            {
                "Name": "instance-type",
                "Values": [
                    config["gpu_instance_type"],
                    config["cpu_head_instance_type"],
                ],
            },
        ],
    ).get("InstanceTypeOfferings", [])
    offered_types = {item.get("InstanceType") for item in offerings}
    required_types = {config["gpu_instance_type"], config["cpu_head_instance_type"]}
    if offered_types != required_types:
        raise RuntimeError(
            f"instance types are not both offered in the selected AZ: {offered_types}"
        )
    stale_instances = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{PROJECT_TAG_KEY}", "Values": [PROJECT_TAG_VALUE]},
            {"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped", "shutting-down"],
            },
        ]
    )
    stale_instance_ids = [
        instance.get("InstanceId")
        for reservation in stale_instances.get("Reservations", ())
        for instance in reservation.get("Instances", ())
    ]
    stale_volumes = ec2.describe_volumes(
        Filters=[
            {"Name": f"tag:{PROJECT_TAG_KEY}", "Values": [PROJECT_TAG_VALUE]},
            {"Name": f"tag:{RUN_TAG_KEY}", "Values": [run_id]},
        ]
    ).get("Volumes", [])
    if stale_instance_ids or stale_volumes:
        raise RuntimeError(
            "run ID already has tagged cloud resources; refuse to mix evidence: "
            f"instances={stale_instance_ids}, "
            f"volumes={[item.get('VolumeId') for item in stale_volumes]}"
        )
    return session, {
        "authenticated": True,
        "region": config["region"],
        "availability_zone": config["availability_zone"],
        "ami_available": True,
        "subnet_and_security_groups_share_vpc": True,
        "instance_profile_available": True,
        "instance_types_offered_in_az": sorted(offered_types),
        "run_id_resource_audit_clean": True,
        "credential_source": (
            "named AWS profile" if config["aws_profile"] else "ambient AWS provider chain"
        ),
        "aws_provider_versions": aws_versions,
    }


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or b""
        output = (
            partial.decode(errors="replace")
            if isinstance(partial, bytes)
            else str(partial)
        )
        output += f"\nLOCAL COMMAND TIMEOUT after {timeout_seconds}s\n"
        completed = subprocess.CompletedProcess(command, 124, output, None)
    log_path.write_text(
        f"COMMAND={shlex.join(str(item) for item in command)}\n"
        f"ELAPSED_S={time.monotonic() - started:.3f}\n"
        f"RETURN_CODE={completed.returncode}\n{output}"
    )
    return completed


def _parse_prefixed_json(output: str, prefix: str) -> dict[str, object]:
    matches = [line[len(prefix) :] for line in output.splitlines() if line.startswith(prefix)]
    if not matches:
        raise ValueError(f"command output is missing {prefix.rstrip('=')}")
    value = json.loads(matches[-1])
    if not isinstance(value, dict):
        raise ValueError(f"{prefix.rstrip('=')} payload must be a JSON object")
    return value


def _remote_commands(
    *, case: Mapping[str, object], args: argparse.Namespace
) -> tuple[str, str, str]:
    topology = TOPOLOGIES[str(case["topology"])]
    remote_directory = f"/home/ray/evidence-results/{args.run_id}/{case['case_id']}"
    preflight = shlex.join(
        (
            REMOTE_ENV_PYTHON,
            "/home/ray/evidence/benchmark/cloud/preflight.py",
            "--arm",
            str(case["arm"]),
            "--expected-min-gpus",
            str(topology.min_gpus),
            "--expected-max-gpus",
            str(topology.max_gpus),
            "--expected-wheel-path",
            str(case["remote_wheel_path"]),
            "--expected-wheel-sha256",
            str(case["expected_wheel_sha256"]),
            "--expected-ray-commit",
            str(case["expected_ray_commit"]),
            "--expected-harness-sha256",
            str(case["expected_harness_sha256"]),
            "--expected-source-provenance",
            str(case["source_provenance"]),
        )
    )
    workload_argv = [
        REMOTE_ENV_PYTHON,
        "/home/ray/evidence/benchmark/cloud/incident_workload.py",
        "--arm",
        str(case["arm"]),
        "--workload",
        str(case["workload"]),
        "--topology",
        str(case["topology"]),
        "--shuffle-ranks",
        str(case["shuffle_ranks"]),
        "--map-actors-min",
        str(case["map_actors_min"]),
        "--map-actors-max",
        str(case["map_actors_max"]),
        "--max-gpus",
        str(topology.max_gpus),
        "--rows",
        str(args.rows),
        "--blocks",
        str(args.blocks),
        "--groups",
        str(args.groups),
        "--batch-size",
        str(args.batch_size),
        "--sample-interval-seconds",
        str(args.sample_interval_seconds),
        "--require-one-l4-per-gpu-node",
        "--result",
        f"{remote_directory}/result.json",
    ]
    admission = "1" if ARMS[str(case["arm"])].admission_enabled else "0"
    workload = (
        f"mkdir -p {shlex.quote(remote_directory)} && "
        f"RAY_DATA_ENABLE_RESOURCE_ADMISSION_CONTROL={admission} "
        f"timeout --signal=TERM --kill-after=30s {args.workload_timeout_seconds}s "
        f"{shlex.join(workload_argv)}"
    )
    return preflight, workload, remote_directory


def execute_case(
    *,
    case: Mapping[str, object],
    args: argparse.Namespace,
    config: Mapping[str, object],
    ec2_client: object,
) -> dict[str, object]:
    case_directory = Path(str(case["cluster_config"])).parent
    cluster_config = str(case["cluster_config"])
    ray_cli = str(args.ray_cli)
    environment = _aws_environment(config)
    result: dict[str, object] = {
        "case_id": case["case_id"],
        "cluster_name": case["cluster_name"],
        "status": "starting",
    }
    teardown: dict[str, object] = {}
    case_started = time.monotonic()
    timeline = InstanceTimeline(
        client=ec2_client,
        cluster_name=str(case["cluster_name"]),
        run_id=args.run_id,
        arm=str(case["arm"]),
        output=case_directory / "ec2-node-timeline.json",
        poll_seconds=min(5.0, args.teardown_poll_seconds),
    )
    timeline.start()
    try:
        up = _run_logged(
            (ray_cli, "up", "-y", cluster_config),
            log_path=case_directory / "ray-up.log",
            environment=environment,
            timeout_seconds=1_800,
        )
        if up.returncode:
            result.update(status="infrastructure-error", phase="ray-up", returncode=up.returncode)
            return result
        result["ray_up_completed_s"] = time.monotonic() - case_started
        preflight_command, workload_command, remote_directory = _remote_commands(
            case=case, args=args
        )
        preflight = _run_logged(
            (ray_cli, "exec", cluster_config, preflight_command),
            log_path=case_directory / "preflight.log",
            environment=environment,
            timeout_seconds=900,
        )
        if preflight.returncode:
            result.update(
                status="infrastructure-error",
                phase="cluster-preflight",
                returncode=preflight.returncode,
            )
            return result
        result["initial_cluster_preflight_completed_s"] = (
            time.monotonic() - case_started
        )
        try:
            preflight_payload = _parse_prefixed_json(preflight.stdout or "", "PREFLIGHT=")
        except (ValueError, json.JSONDecodeError) as error:
            result.update(
                status="infrastructure-error",
                phase="cluster-preflight-evidence",
                error={"type": type(error).__name__, "message": str(error)},
            )
            return result
        _atomic_json(case_directory / "preflight.json", preflight_payload)
        result["preflight"] = preflight_payload
        result["workload_command_requested_s"] = time.monotonic() - case_started
        result["workload_command_requested_at"] = datetime.now(timezone.utc).isoformat()
        workload = _run_logged(
            (ray_cli, "exec", cluster_config, workload_command),
            log_path=case_directory / "workload.log",
            environment=environment,
            timeout_seconds=args.workload_timeout_seconds + 180,
        )
        local_remote = case_directory / "remote"
        local_remote.mkdir(parents=True, exist_ok=True)
        sync = _run_logged(
            (ray_cli, "rsync-down", cluster_config, remote_directory + "/", str(local_remote)),
            log_path=case_directory / "rsync-down.log",
            environment=environment,
            timeout_seconds=300,
        )
        remote_result = local_remote / "result.json"
        if sync.returncode:
            result.update(
                status="infrastructure-error",
                phase="evidence-rsync",
                workload_returncode=workload.returncode,
                rsync_returncode=sync.returncode,
            )
            return result
        if not remote_result.is_file():
            result.update(
                status="infrastructure-error",
                phase="evidence-missing",
                workload_returncode=workload.returncode,
                rsync_returncode=sync.returncode,
            )
            return result
        try:
            payload = json.loads(remote_result.read_text())
        except (OSError, json.JSONDecodeError) as error:
            result.update(
                status="infrastructure-error",
                phase="evidence-invalid",
                workload_returncode=workload.returncode,
                rsync_returncode=sync.returncode,
                error={"type": type(error).__name__, "message": str(error)},
            )
            return result
        if not isinstance(payload, dict):
            result.update(status="infrastructure-error", phase="evidence-not-object")
            return result
        ray_job_id = payload.get("ray_job_id")
        if not isinstance(ray_job_id, str) or not ray_job_id:
            result.update(status="infrastructure-error", phase="evidence-missing-job-id")
            return result
        leak_command = shlex.join(
            (
                REMOTE_ENV_PYTHON,
                "/home/ray/evidence/benchmark/cloud/preflight.py",
                "--check-job-id",
                ray_job_id,
                "--leak-timeout-seconds",
                "120",
            )
        )
        leak_audit = _run_logged(
            (ray_cli, "exec", cluster_config, leak_command),
            log_path=case_directory / "post-workload-leak-audit.log",
            environment=environment,
            timeout_seconds=180,
        )
        if leak_audit.returncode:
            result.update(
                status="workload-error",
                phase="ray-resource-leak",
                workload_returncode=workload.returncode,
                rsync_returncode=sync.returncode,
                workload_result=payload,
            )
            return result
        try:
            leak_payload = _parse_prefixed_json(
                leak_audit.stdout or "", "PREFLIGHT="
            )
        except (ValueError, json.JSONDecodeError) as error:
            result.update(
                status="infrastructure-error",
                phase="leak-audit-evidence",
                error={"type": type(error).__name__, "message": str(error)},
            )
            return result
        _atomic_json(case_directory / "post-workload-leak-audit.json", leak_payload)
        result["post_workload_leak_audit"] = leak_payload
        payload_status = payload.get("status")
        expected_failure = str(case["workload"]) in EXPECTED_FAILURE_WORKLOADS
        if workload.returncode == 0 and payload_status == "success" and not expected_failure:
            workload_status = "completed"
        elif expected_failure and payload_status == "expected-failure":
            workload_status = "expected-failure"
        elif (
            workload.returncode == 124
            and payload_status in {"terminated", "timeout"}
            and payload.get("resource_samples")
        ):
            workload_status = "workload-timeout"
        elif workload.returncode == 0:
            workload_status = "infrastructure-error"
        else:
            workload_status = "workload-error"
        result.update(
            status=workload_status,
            workload_returncode=workload.returncode,
            rsync_returncode=sync.returncode,
            workload_result=payload,
        )
        if workload_status == "infrastructure-error":
            result["phase"] = "evidence-status-mismatch"
        return result
    finally:
        cleanup_errors: list[dict[str, str]] = []
        pre_down_instance_ids: list[str] = []
        try:
            timeline_report = timeline.stop()
            result["ec2_node_timeline"] = timeline_report
            pre_down_instance_ids = timeline_instance_ids(timeline_report)
            teardown["instances_discovered_before_ray_down"] = pre_down_instance_ids
            if timeline_report.get("errors"):
                cleanup_errors.append(
                    {
                        "phase": "ec2-node-timeline-observation",
                        "type": "RuntimeError",
                        "message": "EC2 timeline contains sampling errors",
                    }
                )
        except Exception as error:
            cleanup_errors.append(
                {
                    "phase": "ec2-node-timeline",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        try:
            down = _run_logged(
                (ray_cli, "down", "-y", cluster_config),
                log_path=case_directory / "ray-down.log",
                environment=environment,
                timeout_seconds=300,
            )
            teardown["ray_down_returncode"] = down.returncode
        except Exception as error:
            teardown["ray_down_returncode"] = None
            cleanup_errors.append(
                {"phase": "ray-down", "type": type(error).__name__, "message": str(error)}
            )
        terminated_ids: list[str] = []
        try:
            teardown["instances"] = terminate_cluster_backstop(
                str(case["cluster_name"]),
                run_id=args.run_id,
                arm=str(case["arm"]),
                client=ec2_client,
                timeout_seconds=args.teardown_timeout_seconds,
                poll_interval_seconds=args.teardown_poll_seconds,
            )
            terminated_ids = list(teardown["instances"]["terminated_instance_ids"])
        except Exception as error:
            cleanup_errors.append(
                {
                    "phase": "instance-backstop",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        allowed_instance_ids = sorted(set(pre_down_instance_ids) | set(terminated_ids))
        teardown["allowed_volume_attachment_instance_ids"] = allowed_instance_ids
        try:
            teardown["volumes"] = verify_no_tagged_volumes(
                str(case["cluster_name"]),
                run_id=args.run_id,
                arm=str(case["arm"]),
                client=ec2_client,
                allowed_instance_ids=allowed_instance_ids,
                timeout_seconds=args.teardown_timeout_seconds,
                poll_interval_seconds=args.teardown_poll_seconds,
            )
        except Exception as error:
            cleanup_errors.append(
                {
                    "phase": "volume-backstop",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        teardown["verified"] = not cleanup_errors
        teardown["errors"] = cleanup_errors
        _atomic_json(case_directory / "teardown.json", teardown)
        result["teardown"] = teardown
        if cleanup_errors:
            result["pre_cleanup_status"] = result.get("status")
            result["status"] = "cleanup-error"


def _structural_closed_wait(result: Mapping[str, object]) -> bool:
    payload = result.get("workload_result")
    if not isinstance(payload, Mapping):
        return False
    metrics = payload.get("resource_metrics")
    return bool(
        isinstance(metrics, Mapping) and metrics.get("structural_closed_wait") is True
    )


def _ec2_timeline_metrics(timeline: Mapping[str, object]) -> dict[str, float]:
    """Derive observed node launch/readiness times from an EC2 timeline."""

    observations = timeline.get("observations")
    if not isinstance(observations, list):
        return {}
    running_counts: list[tuple[float, int]] = []
    pending_counts: list[tuple[float, int]] = []
    running_worker_counts: list[tuple[float, int]] = []
    pending_worker_counts: list[tuple[float, int]] = []
    first_seen: dict[str, float] = {}
    first_pending: dict[str, float] = {}
    first_running: dict[str, float] = {}
    first_worker_seen: dict[str, float] = {}
    first_worker_pending: dict[str, float] = {}
    first_worker_running: dict[str, float] = {}
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        elapsed = observation.get("elapsed_s")
        instances = observation.get("instances")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not isinstance(instances, list)
        ):
            continue
        elapsed_s = float(elapsed)
        running = 0
        pending = 0
        running_workers = 0
        pending_workers = 0
        for instance in instances:
            if not isinstance(instance, Mapping):
                continue
            instance_id = instance.get("instance_id")
            if not isinstance(instance_id, str) or not INSTANCE_ID_RE.fullmatch(
                instance_id
            ):
                continue
            first_seen[instance_id] = min(
                elapsed_s, first_seen.get(instance_id, elapsed_s)
            )
            is_worker = instance.get("ray_node_kind") == "worker"
            if is_worker:
                first_worker_seen[instance_id] = min(
                    elapsed_s, first_worker_seen.get(instance_id, elapsed_s)
                )
            state = instance.get("state")
            if state == "pending":
                pending += 1
                first_pending[instance_id] = min(
                    elapsed_s, first_pending.get(instance_id, elapsed_s)
                )
                if is_worker:
                    pending_workers += 1
                    first_worker_pending[instance_id] = min(
                        elapsed_s,
                        first_worker_pending.get(instance_id, elapsed_s),
                    )
            elif state == "running":
                running += 1
                first_running[instance_id] = min(
                    elapsed_s, first_running.get(instance_id, elapsed_s)
                )
                if is_worker:
                    running_workers += 1
                    first_worker_running[instance_id] = min(
                        elapsed_s,
                        first_worker_running.get(instance_id, elapsed_s),
                    )
        running_counts.append((elapsed_s, running))
        pending_counts.append((elapsed_s, pending))
        running_worker_counts.append((elapsed_s, running_workers))
        pending_worker_counts.append((elapsed_s, pending_workers))
    if not running_counts:
        return {}
    metrics: dict[str, float] = {}
    metrics["ec2_timeline_duration_s"] = max(elapsed for elapsed, _ in running_counts)
    if first_seen:
        metrics["ec2_first_instance_seen_s"] = min(first_seen.values())
    if first_pending:
        metrics["ec2_first_pending_s"] = min(first_pending.values())
    if first_running:
        metrics["ec2_first_running_s"] = min(first_running.values())
    peak = max(count for _, count in running_counts)
    if peak > 0:
        metrics["ec2_peak_running_instances"] = float(peak)
        metrics["ec2_observed_peak_running_s"] = min(
            elapsed_s for elapsed_s, count in running_counts if count == peak
        )
    if first_seen and first_seen.keys() <= first_running.keys():
        metrics["ec2_all_observed_instances_running_s"] = max(
            first_running[instance_id] for instance_id in first_seen
        )
    if first_worker_seen:
        metrics["ec2_first_worker_seen_s"] = min(first_worker_seen.values())
    if first_worker_pending:
        metrics["ec2_first_worker_pending_s"] = min(first_worker_pending.values())
    if first_worker_running:
        metrics["ec2_first_worker_running_s"] = min(first_worker_running.values())
    worker_peak = max(count for _, count in running_worker_counts)
    if worker_peak > 0:
        metrics["ec2_peak_running_workers"] = float(worker_peak)
        metrics["ec2_observed_peak_running_workers_s"] = min(
            elapsed_s
            for elapsed_s, count in running_worker_counts
            if count == worker_peak
        )
    if first_worker_seen and first_worker_seen.keys() <= first_worker_running.keys():
        metrics["ec2_all_observed_workers_running_s"] = max(
            first_worker_running[instance_id] for instance_id in first_worker_seen
        )
    if len(running_counts) > 1:
        metrics["ec2_running_instance_seconds"] = sum(
            (right_elapsed - left_elapsed) * left_count
            for (left_elapsed, left_count), (right_elapsed, _right_count) in zip(
                running_counts, running_counts[1:]
            )
        )
        metrics["ec2_pending_instance_seconds"] = sum(
            (right_elapsed - left_elapsed) * left_count
            for (left_elapsed, left_count), (right_elapsed, _right_count) in zip(
                pending_counts, pending_counts[1:]
            )
        )
        metrics["ec2_running_worker_seconds"] = sum(
            (right_elapsed - left_elapsed) * left_count
            for (left_elapsed, left_count), (right_elapsed, _right_count) in zip(
                running_worker_counts, running_worker_counts[1:]
            )
        )
        metrics["ec2_pending_worker_seconds"] = sum(
            (right_elapsed - left_elapsed) * left_count
            for (left_elapsed, left_count), (right_elapsed, _right_count) in zip(
                pending_worker_counts, pending_worker_counts[1:]
            )
        )
    return metrics


def _timestamp_offset(value: object, origin: object) -> float | None:
    if not isinstance(value, str) or not isinstance(origin, str):
        return None
    try:
        return (datetime.fromisoformat(value) - datetime.fromisoformat(origin)).total_seconds()
    except (TypeError, ValueError):
        return None


def _case_timing_metrics(result: Mapping[str, object]) -> dict[str, float]:
    """Put EC2, Ray capacity, demand, and useful progress on one clock."""

    timeline = result.get("ec2_node_timeline")
    payload = result.get("workload_result")
    if not isinstance(timeline, Mapping) or not isinstance(payload, Mapping):
        return {}
    origin = timeline.get("started_at")
    metrics: dict[str, float] = {}

    def add_offset(name: str, value: object) -> None:
        offset = _timestamp_offset(value, origin)
        if offset is not None:
            metrics[name] = offset

    add_offset("workload_remote_start_offset_s", payload.get("started_at"))
    add_offset(
        "workload_demand_start_offset_s", payload.get("workload_demand_started_at")
    )
    resource_metrics = payload.get("resource_metrics")
    if isinstance(resource_metrics, Mapping):
        add_offset(
            "ray_first_gpu_visible_offset_s",
            resource_metrics.get("first_gpu_visible_at"),
        )
        add_offset(
            "ray_observed_peak_gpu_visible_offset_s",
            resource_metrics.get("observed_peak_gpus_visible_at"),
        )
        add_offset(
            "ray_topology_max_gpu_visible_offset_s",
            resource_metrics.get("topology_max_gpus_visible_at"),
        )
        add_offset(
            "first_useful_progress_offset_s",
            resource_metrics.get("first_useful_progress_at"),
        )
    demand = metrics.get("workload_demand_start_offset_s")
    progress = metrics.get("first_useful_progress_offset_s")
    if demand is not None and progress is not None:
        metrics["workload_demand_to_first_progress_s"] = progress - demand
    for ready_name in (
        "ray_first_gpu_visible_offset_s",
        "ray_observed_peak_gpu_visible_offset_s",
        "ray_topology_max_gpu_visible_offset_s",
    ):
        ready = metrics.get(ready_name)
        if ready is not None and progress is not None:
            metrics[f"{ready_name.removesuffix('_offset_s')}_to_first_progress_s"] = (
                progress - ready
            )
    ec2_metrics = _ec2_timeline_metrics(timeline)
    for ready_name in (
        "ec2_first_running_s",
        "ec2_observed_peak_running_s",
        "ec2_all_observed_instances_running_s",
        "ec2_first_worker_running_s",
        "ec2_observed_peak_running_workers_s",
        "ec2_all_observed_workers_running_s",
    ):
        ready = ec2_metrics.get(ready_name)
        if ready is not None and progress is not None:
            metrics[f"{ready_name.removesuffix('_s')}_to_first_progress_s"] = (
                progress - ready
            )
    return metrics


def _scale_profile_report(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_role: dict[str, list[dict[str, object]]] = {
        spec.role: [] for spec in SCALE_CASE_SPECS
    }
    for result in results:
        payload = result.get("workload_result")
        role = str(result.get("role", ""))
        if (
            role in by_role
            and result.get("status") == "completed"
            and isinstance(payload, Mapping)
            and isinstance(payload.get("elapsed_s"), (int, float))
        ):
            by_role[role].append(
                {
                    "repetition": result.get("repetition"),
                    "elapsed_s": float(payload["elapsed_s"]),
                }
            )

    def comparison(candidate_role: str, baseline_role: str) -> dict[str, object]:
        label = f"{candidate_role}-vs-{baseline_role}"
        seed = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
        return compare_randomized_runs(
            by_role[candidate_role],
            by_role[baseline_role],
            seed=seed,
        )

    scale_comparison = comparison(
        "scale-candidate-full", "scale-stock-workaround"
    )
    normal_comparison = comparison(
        "normal-candidate-equal", "normal-stock-equal"
    )
    scale_gate = speedup_gate(scale_comparison, minimum_speedup=0.10)
    regression = regression_gate(normal_comparison, max_regression=0.05)
    for gate, comparison_result in (
        (scale_gate, scale_comparison),
        (regression, normal_comparison),
    ):
        paired = comparison_result.get("mode") == "paired_by_repetition"
        gate["paired_by_repetition"] = paired
        if not paired:
            gate.update(status="unpaired", passed=False)

    blocks: dict[int, list[Mapping[str, object]]] = {}
    for result in results:
        repetition = result.get("repetition")
        if isinstance(repetition, int) and not isinstance(repetition, bool):
            blocks.setdefault(repetition, []).append(result)
    expected_roles = {spec.role for spec in SCALE_CASE_SPECS}
    block_issues = []
    for repetition, block in sorted(blocks.items()):
        roles = [str(item.get("role")) for item in block]
        orders = [item.get("randomization_order") for item in block]
        if set(roles) != expected_roles or len(roles) != len(expected_roles):
            block_issues.append(
                {"repetition": repetition, "reason": "role set mismatch", "roles": roles}
            )
        if set(orders) != set(range(1, len(expected_roles) + 1)):
            block_issues.append(
                {
                    "repetition": repetition,
                    "reason": "randomization order mismatch",
                    "orders": orders,
                }
            )
    expected_repetitions = max(
        (int(item.get("repetition", 0)) for item in results), default=0
    )
    block_gate = {
        "gate": "repetition_blocked_randomization",
        "passed": bool(blocks) and not block_issues,
        "blocks": len(blocks),
        "expected_repetitions": expected_repetitions,
        "issues": block_issues,
    }

    completion_issues = [
        {"case_id": item.get("case_id"), "status": item.get("status")}
        for item in results
        if item.get("status") != "completed"
    ]
    completion_gate = {
        "gate": "all_scale_controls_complete",
        "passed": bool(results) and not completion_issues,
        "completed": len(results) - len(completion_issues),
        "expected": len(results),
        "issues": completion_issues,
    }
    cleanup_issues = [
        item.get("case_id")
        for item in results
        if not isinstance(item.get("teardown"), Mapping)
        or item["teardown"].get("verified") is not True  # type: ignore[index]
    ]
    cleanup_gate = {
        "gate": "all_cloud_resources_cleaned",
        "passed": bool(results) and not cleanup_issues,
        "verified": len(results) - len(cleanup_issues),
        "expected": len(results),
        "issues": cleanup_issues,
    }

    correctness: dict[tuple[str, int], set[str]] = {}
    for result in results:
        payload = result.get("workload_result")
        group = result.get("correctness_group")
        repetition = result.get("repetition")
        if (
            result.get("status") == "completed"
            and isinstance(payload, Mapping)
            and isinstance(group, str)
            and isinstance(repetition, int)
        ):
            oracle = (
                payload.get("output_schema"),
                payload.get("output_rows"),
                payload.get("output_digest"),
            )
            correctness.setdefault((group, repetition), set()).add(
                json.dumps(oracle, sort_keys=True, default=str)
            )
    correctness_issues = [
        {"group": group, "repetition": repetition, "distinct_oracles": len(oracles)}
        for (group, repetition), oracles in sorted(correctness.items())
        if len(oracles) != 1
    ]
    correctness_gate = {
        "gate": "identical_correctness_oracles",
        "passed": bool(correctness) and not correctness_issues,
        "groups": len(correctness),
        "issues": correctness_issues,
    }

    role_metrics: dict[str, dict[str, object]] = {}
    for role in sorted(by_role):
        role_results = [item for item in results if item.get("role") == role]
        values: dict[str, list[float]] = {}

        def collect(name: str, value: object) -> None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(name, []).append(float(value))

        for item in role_results:
            collect("ray_up_s", item.get("ray_up_completed_s"))
            collect(
                "initial_cluster_preflight_s",
                item.get("initial_cluster_preflight_completed_s"),
            )
            collect(
                "workload_command_requested_s",
                item.get("workload_command_requested_s"),
            )
            payload = item.get("workload_result")
            if isinstance(payload, Mapping):
                for name in (
                    "elapsed_s",
                    "materialize_s",
                    "input_rows_per_second",
                    "global_bytes_spilled",
                    "global_bytes_restored",
                ):
                    collect(name, payload.get(name))
                resource = payload.get("resource_metrics")
                if isinstance(resource, Mapping):
                    for name in (
                        "cluster_gpu_seconds",
                        "owned_gpu_seconds",
                        "peak_cluster_gpus",
                    ):
                        collect(name, resource.get(name))
            timeline = item.get("ec2_node_timeline")
            if isinstance(timeline, Mapping):
                for name, value in _ec2_timeline_metrics(timeline).items():
                    collect(name, value)
            for name, value in _case_timing_metrics(item).items():
                collect(name, value)
        role_metrics[role] = {
            name: summarize_samples(
                samples,
                seed=int(hashlib.sha256(f"{role}:{name}".encode()).hexdigest()[:8], 16),
                resamples=2_000,
            )
            for name, samples in sorted(values.items())
        }

    return {
        "schema_version": 1,
        "merge_gates": {
            "scale_speedup": scale_gate,
            "equal_shape_regression": regression,
            "repetition_blocks": block_gate,
            "completion": completion_gate,
            "correctness": correctness_gate,
            "cleanup": cleanup_gate,
        },
        "comparisons": {
            "scale_candidate_vs_best_stock_workaround": scale_comparison,
            "equal_shape_actor_only": normal_comparison,
            "candidate_elastic_autoscale_vs_fixed": comparison(
                "autoscale-elastic-candidate", "fixed-elastic-candidate"
            ),
            "candidate_fixed_pool_autoscale_vs_fixed": comparison(
                "autoscale-fixed-candidate", "scale-candidate-full"
            ),
            "stock_autoscale_more_vs_candidate_bounded_autoscale": comparison(
                "autoscale-more-stock", "autoscale-elastic-candidate"
            ),
            "candidate_default_rank_vs_explicit_rank4_autoscale": comparison(
                "autoscale-default-rank-candidate",
                "autoscale-elastic-candidate",
            ),
        },
        "autoscale_more_resource_tradeoff": {
            "stock_max_gpus": 7,
            "candidate_max_gpus": 4,
            "extra_gpu_node_ceiling": 3,
            "maximum_gpu_capacity_ratio": 7 / 4,
            "observed_stock": role_metrics["autoscale-more-stock"],
            "observed_candidate": role_metrics["autoscale-elastic-candidate"],
        },
        "role_statistics": role_metrics,
    }


def _campaign_report(
    results: Sequence[Mapping[str, object]], *, profile: str = "full"
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    completed: dict[tuple[str, str, str, str, int, int, str], list[float]] = {}
    metrics_by_group: dict[
        tuple[str, str, str, str, int, int, str], dict[str, list[float]]
    ] = {}
    correctness: dict[tuple[object, int], dict[str, tuple[object, ...]]] = {}
    for result in results:
        status = str(result.get("status"))
        arm = str(result.get("arm", result.get("case_id", "").split("-")[1]))
        workload = str(result.get("workload", ""))
        payload = result.get("workload_result")
        if status == "workload-timeout":
            allowed_control_deadlock = (
                arm in {"stock", "pg-only"}
                and workload in {"incident", "actor-only", "shuffle-heavy"}
                and _structural_closed_wait(result)
            )
            if not allowed_control_deadlock:
                failures.append(
                    {"case_id": result.get("case_id"), "reason": "unaccepted timeout"}
                )
        elif status not in {"completed", "expected-failure"}:
            failures.append(
                {"case_id": result.get("case_id"), "reason": status}
            )
        if status != "completed" or not isinstance(payload, Mapping):
            continue
        topology = str(result.get("topology", ""))
        rank = str(result.get("shuffle_ranks", ""))
        map_min = int(result.get("map_actors_min", 0))
        map_max = int(result.get("map_actors_max", map_min))
        role = str(result.get("role", "full-matrix"))
        elapsed = payload.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            key = (arm, topology, workload, rank, map_min, map_max, role)
            completed.setdefault(key, []).append(float(elapsed))
            metrics = metrics_by_group.setdefault(key, {})

            def add_metric(name: str, value: object) -> None:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics.setdefault(name, []).append(float(value))

            add_metric("workload_completion_s", elapsed)
            add_metric("materialize_s", payload.get("materialize_s"))
            add_metric(
                "input_rows_per_second", payload.get("input_rows_per_second")
            )
            add_metric("global_bytes_spilled", payload.get("global_bytes_spilled"))
            add_metric(
                "global_bytes_restored", payload.get("global_bytes_restored")
            )
            add_metric("ray_up_s", result.get("ray_up_completed_s"))
            add_metric(
                "workload_command_requested_s",
                result.get("workload_command_requested_s"),
            )
            add_metric(
                "initial_cluster_preflight_s",
                result.get("initial_cluster_preflight_completed_s"),
            )
            resource_metrics = payload.get("resource_metrics")
            if isinstance(resource_metrics, Mapping):
                for name in (
                    "first_gpu_visible_s",
                    "time_to_observed_peak_gpus_s",
                    "time_to_topology_max_gpus_s",
                    "cluster_gpu_seconds",
                    "owned_gpu_seconds",
                    "available_gpu_seconds",
                ):
                    add_metric(name, resource_metrics.get(name))
            timeline = result.get("ec2_node_timeline")
            if isinstance(timeline, Mapping):
                for name, value in _ec2_timeline_metrics(timeline).items():
                    add_metric(name, value)
            for name, value in _case_timing_metrics(result).items():
                add_metric(name, value)
        repetition = int(result.get("repetition", 0))
        oracle = (
            payload.get("output_schema"),
            payload.get("output_rows"),
            payload.get("output_digest"),
        )
        correctness_group = result.get("correctness_group") or (
            topology,
            workload,
            rank,
        )
        oracle_label = arm if role == "full-matrix" else role
        correctness.setdefault((correctness_group, repetition), {})[
            oracle_label
        ] = oracle
    correctness_mismatches = []
    for key, by_arm in correctness.items():
        distinct = {json.dumps(value, sort_keys=True, default=str) for value in by_arm.values()}
        if len(distinct) > 1:
            correctness_mismatches.append(
                {"key": key, "by_arm": by_arm}
            )
    failures.extend(
        {"case_id": None, "reason": "correctness mismatch", "detail": item}
        for item in correctness_mismatches
    )
    statistics_rows = []
    for key, values in sorted(completed.items()):
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        row = {
                "arm": key[0],
                "topology": key[1],
                "workload": key[2],
                "shuffle_ranks": key[3],
                "map_actors_min": key[4],
                "map_actors_max": key[5],
                "role": key[6],
                "runs": len(values),
                "median_elapsed_s": median,
                "median_absolute_deviation_s": statistics.median(deviations),
                "min_elapsed_s": min(values),
                "max_elapsed_s": max(values),
            }
        for metric_name, metric_values in sorted(metrics_by_group.get(key, {}).items()):
            row[f"median_{metric_name}"] = statistics.median(metric_values)
        statistics_rows.append(row)
    report: dict[str, object] = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "correctness_mismatches": correctness_mismatches,
        "statistics": statistics_rows,
    }
    if profile == "scale":
        scale_evidence = _scale_profile_report(results)
        report["scale_profile"] = scale_evidence
        failed_gates = [
            name
            for name, gate in scale_evidence["merge_gates"].items()
            if gate.get("passed") is not True
        ]
        if failed_gates:
            failures.append(
                {
                    "case_id": None,
                    "reason": "scale profile merge gate did not pass",
                    "detail": failed_gates,
                }
            )
        report["passed"] = not failures
    return report


def execute_campaign(
    *,
    plan: Mapping[str, object],
    args: argparse.Namespace,
    config: Mapping[str, object],
    session: object,
) -> int:
    if not args.ray_cli.is_file():
        raise RuntimeError(f"Ray CLI does not exist: {args.ray_cli}")
    ec2 = session.client("ec2")
    results = []
    campaign_abort: dict[str, object] | None = None
    for case in plan["cases"]:
        result = execute_case(
            case=case, args=args, config=config, ec2_client=ec2
        )
        result.update(
            arm=case["arm"],
            topology=case["topology"],
            workload=case["workload"],
            shuffle_ranks=case["shuffle_ranks"],
            repetition=case["repetition"],
            map_actors_min=case["map_actors_min"],
            map_actors_max=case["map_actors_max"],
            role=case["role"],
            correctness_group=case["correctness_group"],
            randomization_block=case["randomization_block"],
            randomization_order=case["randomization_order"],
        )
        results.append(result)
        _atomic_json(args.artifact_root / "campaign-results.json", results)
        teardown = result.get("teardown")
        cleanup_proven = (
            isinstance(teardown, Mapping)
            and teardown.get("verified") is True
            and result.get("status") != "cleanup-error"
        )
        if not cleanup_proven:
            campaign_abort = {
                "case_id": result.get("case_id"),
                "status": result.get("status"),
                "reason": "cleanup could not be proven; later cloud cases were not started",
                "completed_cases": len(results),
                "planned_cases": len(plan["cases"]),
            }
            _atomic_json(args.artifact_root / "campaign-abort.json", campaign_abort)
            break
    report = _campaign_report(results, profile=args.profile)
    if campaign_abort is not None:
        report["campaign_abort"] = campaign_abort
        report.setdefault("failures", []).append(
            {
                "case_id": campaign_abort["case_id"],
                "reason": campaign_abort["reason"],
            }
        )
        report["passed"] = False
    _atomic_json(args.artifact_root / "evidence-report.json", report)
    return 0 if report["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_local_config(args.config, for_execute=args.execute)
        template = args.cluster_template.read_text()
        plan = build_plan(args, config=config, template=template)
    except (OSError, ValueError) as error:
        raise SystemExit(f"configuration error: {error}") from error
    _atomic_json(args.artifact_root / "plan.json", plan)
    summary = {
        "mode": plan["mode"],
        "profile": plan["matrix"]["profile"],
        "run_id": args.run_id,
        "cases": len(plan["cases"]),
        "rows": plan["workload"]["rows"],
        "blocks": plan["workload"]["blocks"],
        "repetitions": plan["matrix"]["repetitions"],
        "workload_timeout_seconds": plan["workload"]["timeout_seconds"],
        "plan": str(args.artifact_root / "plan.json"),
        "pending_artifacts": [
            name
            for name, artifact in plan["artifacts"].items()
            if not artifact["ready"]
        ],
        "artifact_separation_issues": plan["artifact_validation"][
            "separation_issues"
        ],
    }
    if plan["scale_profile"] is not None:
        summary["scale_campaign_budget"] = plan["scale_profile"][
            "campaign_budget"
        ]
    print(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    if not args.execute:
        return 0
    if (
        len(plan["cases"]) > args.max_execute_cases
        and args.confirm_large_run != args.run_id
    ):
        raise SystemExit(
            f"refusing {len(plan['cases'])} paid cases above --max-execute-cases "
            f"{args.max_execute_cases}; inspect the plan and pass "
            f"--confirm-large-run {args.run_id}"
        )
    session, preflight = aws_preflight(config, run_id=args.run_id)
    _atomic_json(args.artifact_root / "aws-preflight.json", preflight)
    return execute_campaign(
        plan=plan, args=args, config=config, session=session
    )


if __name__ == "__main__":
    raise SystemExit(main())
