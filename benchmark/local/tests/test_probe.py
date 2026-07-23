from __future__ import annotations

from benchmark import ray_core_state
from benchmark.local import probe


def test_job_cleanup_allows_only_known_detached_ray_data_services(monkeypatch) -> None:
    class Reader:
        def actors(self, *, job_id):
            assert job_id == "job-a"
            return [
                {
                    "actor_id": "service",
                    "class_name": "_StatsActor",
                    "name": "datasets_stats_actor",
                    "state": "ALIVE",
                    "pid": 1,
                    "is_detached": True,
                    "required_resources": {},
                },
                {
                    "actor_id": "gpu-leak",
                    "class_name": "UserActor",
                    "name": "",
                    "state": "ALIVE",
                    "pid": 2,
                    "is_detached": True,
                    "required_resources": {"GPU": 1},
                },
            ]

        def placement_groups(self, *, job_id):
            assert job_id == "job-a"
            return []

    monkeypatch.setattr(ray_core_state, "CoreGcsStateReader", Reader)

    resources = probe._active_job_resources("job-a")

    assert [actor["actor_id"] for actor in resources["active_actors"]] == ["gpu-leak"]
    assert [
        actor["actor_id"] for actor in resources["permitted_detached_ray_data_services"]
    ] == ["service"]
