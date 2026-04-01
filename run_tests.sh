#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
INSTALL_DEPS=1

if [[ "${1:-}" == "--no-install" ]]; then
  INSTALL_DEPS=0
  shift
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r "$ROOT_DIR/requirements-test.txt"
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

if [[ "$#" -gt 0 ]]; then
  python -m pytest "$@"
else
  python -m pytest "$ROOT_DIR/tests"
fi
