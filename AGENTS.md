# Agent workspace notes

## Ray test temporary storage

- Do not use the host's base `/tmp` for Ray acceptance or spilling tests. It can
  exceed Ray's 95% disk-utilization guard even when the project filesystem has
  ample capacity.
- Use a short path under shared memory, normally `/dev/shm/ray-admission`.
  `scripts/run_ray_acceptance.py` does this by default; override it with
  `RAY_GPU_ACCEPTANCE_TMPDIR` when concurrent runs need separate directories.
- Keep `RAY_TMPDIR` short. Ray's plasma and raylet Unix socket paths must fit
  the Linux 107-byte `AF_UNIX` limit.
- Do not broadly delete `/tmp/ray`; it may contain sessions owned by other
  users or tasks.
