# ray-data-gpu-fusion

An opt-in Ray Data physical-optimizer and actor backend for modular cuDF
operators. Install it beside the repository's pinned derived Ray wheel, then
enable it before constructing a Dataset:

```python
import ray_data_gpu_fusion as rgf

rgf.enable()
```

Dataset construction and actions remain ordinary Ray Data APIs. See the
repository-level architecture and Phase-0 contract documents for supported
operator shapes, fallback behavior, and reproducible environment setup.

This Phase-0 wheel is not a standalone pip environment. cuDF, CuPy, RMM, and
their native CUDA libraries come from the repository's pinned Conda lock; use
the root `scripts/bootstrap.sh` entrypoint. Experimental S3 execution also
requires `botocore` and the RAPIDS Python `kvikio` package on every GPU worker.
Those S3 dependencies are intentionally not installed or hardware-validated by
the default bootstrap.
