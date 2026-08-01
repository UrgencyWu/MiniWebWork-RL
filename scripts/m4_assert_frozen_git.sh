#!/usr/bin/env bash
# Require every formal M4 job to run exactly the commit recorded at submission.
set -euo pipefail

: "${M4_EXPECTED_GIT_SHA:?M4_EXPECTED_GIT_SHA must be exported when submitting formal M4 jobs}"
actual_git_sha="$(git rev-parse HEAD)"
if [[ "$actual_git_sha" != "$M4_EXPECTED_GIT_SHA" ]]; then
  echo "M4 frozen commit mismatch: expected $M4_EXPECTED_GIT_SHA, found $actual_git_sha" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "M4 formal jobs require a clean tracked worktree at the frozen commit" >&2
  exit 1
fi
