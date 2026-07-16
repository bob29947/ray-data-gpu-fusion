#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$root/.venv/bin/python"
stock_wheel="$root/wheels/stock/ray-3.0.0.dev0-cp311-cp311-linux_x86_64.whl"

if [[ -n "${CONDA_EXE:-}" ]]; then
  conda_bin="$CONDA_EXE"
elif [[ -x /opt/miniconda3/bin/conda ]]; then
  conda_bin=/opt/miniconda3/bin/conda
elif command -v conda >/dev/null 2>&1; then
  conda_bin="$(command -v conda)"
else
  conda_bin=""
fi

git -C "$root" submodule update --init --recursive

if [[ ! -x "$python" ]]; then
  if [[ -z "$conda_bin" ]]; then
    echo "Conda was not found; set CONDA_EXE to its executable" >&2
    exit 1
  fi
  "$conda_bin" create \
    --yes \
    --prefix "$root/.venv" \
    --file "$root/environment/rapids-25.12-linux-64.explicit.txt"
fi

if [[ ! -f "$stock_wheel" ]]; then
  echo "Missing pinned stock Ray wheel: $stock_wheel" >&2
  exit 1
fi

(
  cd "$root"
  sha256sum --check pins/SHA256SUMS
)

"$python" -m pip install \
  --no-deps \
  --require-hashes \
  -r "$root/environment/ray-runtime-linux-py311.lock"

"$python" "$root/scripts/build_patched_ray_wheel.py"
patched_wheel="$root/wheels/patched/$(basename "$stock_wheel")"
if [[ ! -f "$patched_wheel" ]]; then
  echo "Patched Ray wheel was not produced" >&2
  exit 1
fi
(
  cd "$root"
  sha256sum --check pins/SHA256SUMS
)

"$python" -m pip install --force-reinstall --no-deps "$patched_wheel"
"$python" -m pip install --no-deps --no-build-isolation --editable "$root/plugin"
"$python" "$root/scripts/verify_install.py"
