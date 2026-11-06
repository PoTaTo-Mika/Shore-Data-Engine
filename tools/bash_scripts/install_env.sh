#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m pip install -U pip
git submodule update --init
python -m pip install -e offline_deps/demucs
python -m pip install -e ".[all]"
