#!/usr/bin/env bash
# Single entry point for the pipeline — forwards to src/run_all.py.
#   ./run.sh <subcommand> [args...]     e.g.  ./run.sh test --scheme original
#   ./run.sh --help                     list subcommands
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/src"
exec python3 run_all.py "$@"
