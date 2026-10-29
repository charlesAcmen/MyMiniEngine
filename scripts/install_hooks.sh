#!/usr/bin/env bash
# Enable this repository's tracked Git hooks for the current clone.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
git -C "${repo_root}" config --local core.hooksPath .githooks
printf '%s\n' 'Enabled repository hooks from .githooks.'
