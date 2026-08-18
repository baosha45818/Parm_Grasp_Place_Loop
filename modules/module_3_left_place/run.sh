#!/usr/bin/env bash
# Standalone Module 3: verified Module 1 handoff -> fixed Buffer placement.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOOP_ROOT="${PARM_LOOP_ROOT:-$(cd -- "${HERE}/../.." && pwd)}"
PYTHON_BIN="${LOOP_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "project Python is unavailable: ${PYTHON_BIN}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${HERE}/standalone_left_place.py" "$@"
