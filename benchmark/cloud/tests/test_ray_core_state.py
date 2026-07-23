from __future__ import annotations

from benchmark.ray_core_state import (
    CoreGcsStateReader,
    is_permitted_detached_ray_data_service,
)


class FakeAccessor:
    def __init__(self, *, actors=(), tasks=(), placement_groups=()):
        self._actors = [item.SerializeToString() for item in actors]
        self._tasks = [item.SerializeToString() for item in tasks]
        self._placement_groups = [item.SerializeToString() for item in placement_groups]

    def get_actor_table(self, job_id, actor_state_name):
        assert job_id is None
        assert actor_state_name is None
        return self._actors

    def get_task_events(self):
        return self._tasks

    def get_placement_group_table(self):
        return self._placement_groups


def test_core_reader_decodes_detailed_actor_task_and_placement_group() -> None:
    from ray.core.generated import common_pb2, gcs_pb2

    job_id = b"job-a"
    actor = gcs_pb2.ActorTableData(
        actor_id=b"actor-a",
        job_id=job_id,
        state=gcs_pb2.ActorTableData.ALIVE,
        class_name="GpuWorker",
        name="worker",
        node_id=b"node-a",
        pid=123,
        required_resources={"GPU": 1},
        placement_group_id=b"pg-a",
        start_time=1000,
    )
    task = gcs_pb2.TaskEvents(
        task_id=b"task-a",
        job_id=job_id,
        attempt_number=2,
        task_info=common_pb2.TaskInfoEntry(
            task_id=b"task-a",
            job_id=job_id,
            name="GpuWorker.run",
            func_or_class_name="GpuWorker.run",
            type=common_pb2.ACTOR_TASK,
            actor_id=b"actor-a",
            placement_group_id=b"pg-a",
            required_resources={"GPU": 1},
        ),
        state_updates=gcs_pb2.TaskStateUpdate(
            node_id=b"node-a",
            worker_id=b"worker-a",
            worker_pid=123,
            state_ts_ns={
                common_pb2.PENDING_ARGS_AVAIL: 1_000_000,
                common_pb2.RUNNING: 2_000_000,
            },
        ),
    )
    group = gcs_pb2.PlacementGroupTableData(
        placement_group_id=b"pg-a",
        creator_job_id=job_id,
        state=gcs_pb2.PlacementGroupTableData.CREATED,
        name="shuffle",
        bundles=[
            common_pb2.Bundle(
                bundle_id=common_pb2.Bundle.BundleIdentifier(
                    placement_group_id=b"pg-a", bundle_index=0
                ),
                unit_resources={"GPU": 1},
                node_id=b"node-a",
            )
        ],
    )
    reader = CoreGcsStateReader(
        FakeAccessor(actors=[actor], tasks=[task], placement_groups=[group])
    )

    snapshot = reader.snapshot(job_id=job_id.hex())

    assert snapshot["actors"] == [
        {
            "actor_id": b"actor-a".hex(),
            "class_name": "GpuWorker",
            "name": "worker",
            "state": "ALIVE",
            "job_id": job_id.hex(),
            "node_id": b"node-a".hex(),
            "pid": 123,
            "required_resources": {"GPU": 1.0},
            "placement_group_id": b"pg-a".hex(),
            "is_detached": False,
            "num_restarts": 0,
            "timestamp": 0.0,
            "start_time_ms": 1000,
            "end_time_ms": 0,
        }
    ]
    assert snapshot["tasks"][0]["state"] == "RUNNING"
    assert snapshot["tasks"][0]["type"] == "ACTOR_TASK"
    assert snapshot["tasks"][0]["required_resources"] == {"GPU": 1.0}
    assert snapshot["tasks"][0]["start_time_ms"] == 2
    assert snapshot["placement_groups"][0]["state"] == "CREATED"
    assert snapshot["placement_groups"][0]["creator_job_id"] == job_id.hex()
    assert snapshot["placement_groups"][0]["bundles"] == [
        {
            "bundle_index": 0,
            "resources": {"GPU": 1.0},
            "node_id": b"node-a".hex(),
        }
    ]


def test_core_reader_filters_job_and_bounds_results() -> None:
    from ray.core.generated import gcs_pb2

    actors = [
        gcs_pb2.ActorTableData(
            actor_id=f"actor-{index}".encode(),
            job_id=(b"wanted" if index < 2 else b"other"),
            state=gcs_pb2.ActorTableData.DEAD,
        )
        for index in range(3)
    ]
    reader = CoreGcsStateReader(FakeAccessor(actors=actors))

    assert len(reader.actors(job_id=b"wanted".hex(), limit=1)) == 1
    assert reader.snapshot(include_tasks=False).keys() == {
        "actors",
        "placement_groups",
    }


def test_only_known_zero_resource_detached_ray_data_services_are_permitted() -> None:
    service = {
        "class_name": "_StatsActor",
        "name": "datasets_stats_actor",
        "is_detached": True,
        "required_resources": {},
    }

    assert is_permitted_detached_ray_data_service(service)
    assert not is_permitted_detached_ray_data_service(
        {**service, "required_resources": {"GPU": 1}}
    )
    assert not is_permitted_detached_ray_data_service(
        {**service, "class_name": "UserActor"}
    )
    assert not is_permitted_detached_ray_data_service({**service, "is_detached": False})
