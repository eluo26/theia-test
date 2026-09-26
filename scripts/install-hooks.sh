#!/usr/bin/env bash
# Point this clone at the committed hooks in .githooks/.
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
echo "Git hooks path set to .githooks"
