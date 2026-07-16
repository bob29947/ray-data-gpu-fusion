#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$root/.venv/bin/python"

"$python" "$root/scripts/verify_install.py"
"$python" -m pytest -q \
  "$root/plugin/tests/unit" \
  "$root/plugin/tests/compatibility"
"$python" -m pytest -q -m "not gpu and not s3" \
  "$root/plugin/tests/integration"

if nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$python" -m pytest -q -m gpu "$root/plugin/tests"
else
  echo "No CUDA device is visible; GPU-marked tests were not run." >&2
fi

if [[ -n "$(git -C "$root/ray-stock" status --porcelain)" ]]; then
  echo "ray-stock became dirty during validation" >&2
  exit 1
fi
