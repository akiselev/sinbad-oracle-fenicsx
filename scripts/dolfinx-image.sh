#!/usr/bin/env bash
# Runs a command inside the official dolfinx image with this checkout mounted read-write at
# /adapter (installed editable into a throwaway per-invocation site) and the caller's uid, so
# nothing the container writes is root-owned on the host. Usage:
#   scripts/dolfinx-image.sh pytest tests
#   scripts/dolfinx-image.sh sinbad-oracle-fenicsx /io/request.json /io/result.json   (with IO_DIR set)
# Env: DOLFINX_IMAGE (default dolfinx/dolfinx:stable), IO_DIR (host dir mounted at /io).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${DOLFINX_IMAGE:-dolfinx/dolfinx:stable}"
mounts=(-v "$here":/adapter)
if [[ -n "${IO_DIR:-}" ]]; then
  mounts+=(-v "$(cd "$IO_DIR" && pwd)":/io)
fi
# The image locates dolfinx through its own PYTHONPATH; prepend, never replace.
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -w /adapter "${mounts[@]}" "$image" \
  bash -c 'export PYTHONPATH="/adapter/src:${PYTHONPATH:-}"; exec "$@"' -- "$@"
