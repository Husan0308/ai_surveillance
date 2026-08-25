#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
TARGET="${1:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
REL="${TARGET#$ROOT/}"
BASE="$(basename "$TARGET")"

if [[ -f "$TARGET" && -s "$TARGET" ]]; then
  echo "CAM01_TRT86_ENGINE_RESTORE status=present path=$TARGET bytes=$(stat -c%s "$TARGET")"
  exit 0
fi

mkdir -p "$(dirname "$TARGET")"

# First recover an already-existing local copy without touching Git state.
FOUND="$(find "$ROOT" -type f -name "$BASE" ! -path "$TARGET" -print -quit 2>/dev/null || true)"
if [[ -n "$FOUND" ]]; then
  cp --reflink=auto -- "$FOUND" "$TARGET"
  echo "CAM01_TRT86_ENGINE_RESTORE status=copied source=$FOUND target=$TARGET bytes=$(stat -c%s "$TARGET")"
  exit 0
fi

# `git stash push -u` stores untracked files in the stash's third parent. Search
# every stash entry and restore only the expected engine; do not pop/apply the
# stash, because that could reintroduce old source files after repository cleanup.
while IFS= read -r stash; do
  [[ -n "$stash" ]] || continue
  if git cat-file -e "${stash}^3:${REL}" 2>/dev/null; then
    TMP="${TARGET}.restore.$$"
    git show "${stash}^3:${REL}" > "$TMP"
    if [[ ! -s "$TMP" ]]; then
      rm -f "$TMP"
      echo "CAM01_TRT86_ENGINE_RESTORE ERROR: restored engine is empty from $stash" >&2
      exit 2
    fi
    mv -f "$TMP" "$TARGET"
    echo "CAM01_TRT86_ENGINE_RESTORE status=restored stash=$stash target=$TARGET bytes=$(stat -c%s "$TARGET")"
    exit 0
  fi
done < <(git stash list --format='%gd')

# Show any other TensorRT engine paths hidden in stashes to make recovery easy.
CANDIDATES="$(
  while IFS= read -r stash; do
    [[ -n "$stash" ]] || continue
    git ls-tree -r --name-only "${stash}^3" 2>/dev/null | grep -E '\.(engine|plan)$' | sed "s#^#${stash}:#" || true
  done < <(git stash list --format='%gd')
)"

if [[ -n "$CANDIDATES" ]]; then
  echo "CAM01_TRT86_ENGINE_RESTORE ERROR: exact engine not found; other stashed engine candidates:" >&2
  printf '%s\n' "$CANDIDATES" >&2
else
  echo "CAM01_TRT86_ENGINE_RESTORE ERROR: engine not found locally or in git stashes: $REL" >&2
fi
exit 1
