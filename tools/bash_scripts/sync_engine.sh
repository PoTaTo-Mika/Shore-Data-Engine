#!/usr/bin/env bash
# Sync all submodules with their tracked branches; commit the gitlinks to pin the result.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
git submodule update --init --remote
git submodule status
