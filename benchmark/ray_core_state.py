"""Dashboard-free Ray state reads directly from the core GCS tables.

The evidence wheels intentionally omit dashboard frontend assets.  Ray's HTTP
State API therefore cannot be a dependency of the benchmark harness, even
though the core actor, task-event, and placement-group tables are available.
This module keeps the pinned private API usage in one place and returns stable,
JSON-serializable records for the cloud and local evidence runners.
"""

from __future__ import annotations

from typing import Mapping


DEFAULT_LIMIT = 10_000
_DETACHED_RAY_DATA_SERVICES = {
    ("_AutoscalingCoordinatorActor", "AutoscalingCoordinator"),
    ("ActorLocationTracker", "ActorLocationTracker"),
    ("_StatsActor", "datasets_stats_actor"),
}


def _hex(identifier: object) -> str:
    if isinstance(identifier, bytes):
        return identifier.hex()
    try:
        return bytes(identifier).hex()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def _message_dict(message: object) -> dict[str, object]:
    from ray._private.protobuf_compat import message_to_dict

    return message_to_dict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=False,
    )


def _actor_record(message: object) -> dict[str, object]:
    from ray.core.generated import gcs_pb2

    node_id = _hex(message.node_id) or _hex(message.address.node_id)
    record = {
        "actor_id": _hex(message.actor_id),
        "class_name": str(message.class_name),
        "name": str(message.name),
        "state": gcs_pb2.ActorTableData.ActorState.Name(message.state),
        "job_id": _hex(message.job_id),
        "node_id": node_id,
        "pid": int(message.pid),
        "required_resources": {
            str(key): float(value) for key, value in message.required_resources.items()
        },
        "placement_group_id": _hex(message.placement_group_id),
        "is_detached": bool(message.is_detached),
        "num_restarts": int(message.num_restarts),
        "timestamp": float(message.timestamp),
        "start_time_ms": int(message.start_time),
        "end_time_ms": int(message.end_time),
    }
    if message.HasField("death_cause"):
        record["death_cause"] = _message_dict(message.death_cause)
    return record


def _task_record(message: object) -> dict[str, object]:
    from ray.core.generated import common_pb2

    info = message.task_info
    updates = message.state_updates
    state_events = sorted(
        (
            {
                "state": common_pb2.TaskStatus.Name(int(state)),
                "created_ms": int(timestamp_ns) // 1_000_000,
            }
            for state, timestamp_ns in updates.state_ts_ns.items()
        ),
        key=lambda event: (int(event["created_ms"]), str(event["state"])),
    )
    state = (
        str(state_events[-1]["state"])
        if state_events
        else common_pb2.TaskStatus.Name(info.scheduling_state)
    )

    def event_time(status: int) -> int | None:
        timestamp = updates.state_ts_ns.get(status)
        return int(timestamp) // 1_000_000 if timestamp is not None else None

    record = {
        "task_id": _hex(message.task_id) or _hex(info.task_id),
        "attempt_number": int(message.attempt_number),
        "name": str(info.name),
        "state": state,
        "type": common_pb2.TaskType.Name(info.type),
        "job_id": _hex(message.job_id) or _hex(info.job_id),
        "actor_id": _hex(info.actor_id),
        "parent_task_id": _hex(info.parent_task_id),
        "node_id": _hex(updates.node_id) or _hex(info.node_id),
        "worker_id": _hex(updates.worker_id),
        "worker_pid": int(updates.worker_pid),
        "required_resources": {
            str(key): float(value) for key, value in info.required_resources.items()
        },
        "placement_group_id": _hex(info.placement_group_id),
        "func_or_class_name": str(info.func_or_class_name),
        "creation_time_ms": event_time(common_pb2.PENDING_ARGS_AVAIL),
        "start_time_ms": event_time(common_pb2.RUNNING),
        "end_time_ms": event_time(common_pb2.FINISHED) or event_time(common_pb2.FAILED),
        "events": state_events,
    }
    if updates.HasField("error_info"):
        record["error"] = _message_dict(updates.error_info)
    return record


def _placement_group_record(message: object) -> dict[str, object]:
    from ray.core.generated import gcs_pb2

    bundles = [
        {
            "bundle_index": int(bundle.bundle_id.bundle_index),
            "resources": {
                str(key): float(value) for key, value in bundle.unit_resources.items()
            },
            "node_id": _hex(bundle.node_id),
        }
        for bundle in message.bundles
    ]
    stats = message.stats
    return {
        "placement_group_id": _hex(message.placement_group_id),
        "name": str(message.name),
        "state": gcs_pb2.PlacementGroupTableData.PlacementGroupState.Name(
            message.state
        ),
        "creator_job_id": _hex(message.creator_job_id),
        "creator_actor_id": _hex(message.creator_actor_id),
        "is_detached": bool(message.is_detached),
        "bundles": bundles,
        "stats": {
            "scheduling_attempt": int(stats.scheduling_attempt),
            "scheduling_state": (
                gcs_pb2.PlacementGroupStats.SchedulingState.Name(stats.scheduling_state)
            ),
            "scheduling_latency_ms": float(stats.scheduling_latency_us) / 1_000,
            "end_to_end_creation_latency_ms": (
                float(stats.end_to_end_creation_latency_us) / 1_000
            ),
            "highest_retry_delay_ms": float(stats.highest_retry_delay_ms),
        },
        "creation_time_ms": int(message.placement_group_creation_timestamp_ms),
        "ready_time_ms": int(
            message.placement_group_final_bundle_placement_timestamp_ms
        ),
    }


def _default_accessor() -> object:
    import ray
    from ray._private.state import state

    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before reading core GCS state")
    return state._connect_and_get_accessor()


def _bounded(records: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    if limit < 1:
        raise ValueError("state record limit must be positive")
    return records[:limit]


def is_permitted_detached_ray_data_service(record: Mapping[str, object]) -> bool:
    """Identify zero-resource Ray Data services that intentionally outlive a job.

    They are cluster-scoped and are still covered by the final node/process
    teardown audit. Unknown detached actors and any actor reserving a positive
    resource remain leaks.
    """

    resources = record.get("required_resources", {})
    has_positive_resource = not isinstance(resources, Mapping) or any(
        float(value) > 0 for value in resources.values()
    )
    identity = (str(record.get("class_name", "")), str(record.get("name", "")))
    return (
        bool(record.get("is_detached"))
        and not has_positive_resource
        and identity in _DETACHED_RAY_DATA_SERVICES
    )


class CoreGcsStateReader:
    """Read pinned Ray core state without importing dashboard components."""

    def __init__(self, accessor: object | None = None):
        self._accessor = accessor

    @property
    def accessor(self) -> object:
        if self._accessor is None:
            self._accessor = _default_accessor()
        return self._accessor

    def actors(
        self, *, job_id: str | None = None, limit: int = DEFAULT_LIMIT
    ) -> list[dict[str, object]]:
        from ray.core.generated import gcs_pb2

        records = [
            _actor_record(gcs_pb2.ActorTableData.FromString(serialized))
            for serialized in self.accessor.get_actor_table(None, None)
        ]
        records.sort(key=lambda record: str(record["actor_id"]))
        if job_id is not None:
            records = [record for record in records if record["job_id"] == job_id]
        return _bounded(records, limit)

    def tasks(
        self, *, job_id: str | None = None, limit: int = DEFAULT_LIMIT
    ) -> list[dict[str, object]]:
        from ray.core.generated import gcs_pb2

        records = [
            _task_record(gcs_pb2.TaskEvents.FromString(serialized))
            for serialized in self.accessor.get_task_events()
        ]
        records.sort(
            key=lambda record: (str(record["task_id"]), int(record["attempt_number"]))
        )
        if job_id is not None:
            records = [record for record in records if record["job_id"] == job_id]
        return _bounded(records, limit)

    def placement_groups(
        self, *, job_id: str | None = None, limit: int = DEFAULT_LIMIT
    ) -> list[dict[str, object]]:
        from ray.core.generated import gcs_pb2

        records = [
            _placement_group_record(
                gcs_pb2.PlacementGroupTableData.FromString(serialized)
            )
            for serialized in self.accessor.get_placement_group_table()
        ]
        records.sort(key=lambda record: str(record["placement_group_id"]))
        if job_id is not None:
            records = [
                record for record in records if record["creator_job_id"] == job_id
            ]
        return _bounded(records, limit)

    def snapshot(
        self,
        *,
        job_id: str | None = None,
        limit: int = DEFAULT_LIMIT,
        include_tasks: bool = True,
    ) -> dict[str, list[dict[str, object]]]:
        snapshot = {
            "actors": self.actors(job_id=job_id, limit=limit),
            "placement_groups": self.placement_groups(job_id=job_id, limit=limit),
        }
        if include_tasks:
            snapshot["tasks"] = self.tasks(job_id=job_id, limit=limit)
        return snapshot


def read_core_state(
    *,
    job_id: str | None = None,
    limit: int = DEFAULT_LIMIT,
    include_tasks: bool = True,
    accessor: object | None = None,
) -> Mapping[str, list[dict[str, object]]]:
    return CoreGcsStateReader(accessor).snapshot(
        job_id=job_id, limit=limit, include_tasks=include_tasks
    )
