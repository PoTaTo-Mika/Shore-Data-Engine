#!/usr/bin/env bash
# Sync the sglang-omni submodule with upstream main; commit the gitlink to pin the result.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
git submodule update --init --remote engine/sglang-omni
git -C engine/sglang-omni log -1 --format='Pinned sglang-omni %H (%cs) %d'
