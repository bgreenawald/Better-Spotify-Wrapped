#!/usr/bin/env bash
set -euo pipefail

# Bootstrap symlinks for shared, untracked paths across git worktrees.
#
# Usage:
#   scripts/bootstrap_worktree.sh [--from <path>] [--force] [--absolute]
#                                 [--dry-run]
#
# - If --from is omitted, the script attempts to auto-detect a source worktree
#   by scanning `git worktree list` and selecting one that already contains
#   the majority of the requested paths.
# - By default, symlinks are relative to the current repo root. Use --absolute
#   to create absolute symlinks instead.
# - Use --force to replace existing files/symlinks at the destination.
# - Use --dry-run to print planned actions without making changes.

TARGETS=(
  ".venv"
  "build"
  "data"
  "resources"
  "tmp"
  ".cache"
  ".env"
)

usage() {
  cat <<EOF
Bootstrap symlinks for shared, untracked paths across git worktrees.

Options:
  --from <path>   Base directory to link from (primary worktree).
  --force         Replace existing destinations if present.
  --absolute      Use absolute symlinks (default: relative).
  --dry-run       Show actions without changing anything.
  -h, --help      Show this help and exit.

Examples:
  scripts/bootstrap_worktree.sh --from ../main-worktree
  scripts/bootstrap_worktree.sh --force
  scripts/bootstrap_worktree.sh --absolute --dry-run
EOF
}

log() { echo "[bootstrap] $*"; }
err() { echo "[bootstrap][error] $*" >&2; }

# Resolve to absolute path without relying on GNU realpath.
abspath() {
  local p="$1"
  if [ -d "$p" ]; then
    (cd "$p" && pwd -P)
  else
    local d
    d=$(dirname -- "$p")
    local f
    f=$(basename -- "$p")
    (cd "$d" 2>/dev/null && printf "%s/%s\n" "$(pwd -P)" "$f")
  fi
}

relpath() {
  # Compute relative path from $1 to $2
  local from="$1"
  local to="$2"
  python3 - "$from" "$to" <<'PY'
import os, sys
from_path, to_path = map(os.path.abspath, sys.argv[1:3])
try:
    print(os.path.relpath(to_path, start=from_path))
except Exception:
    print(to_path)
PY
}

require_git_root() {
  if ! git_root=$(git rev-parse --show-toplevel 2>/dev/null); then
    err "Not inside a git repository."
    exit 1
  fi
}

detect_source_worktree() {
  # Try to find a worktree that already contains most of TARGETS
  local best_path=""
  local best_score=-1
  if ! command -v git >/dev/null; then
    echo ""; return
  fi

  local line wt_path
  while IFS= read -r line; do
    case "$line" in
      worktree\ *)
        wt_path=${line#worktree };
        # Score by number of targets that exist in this worktree
        local score=0
        for item in "${TARGETS[@]}"; do
          if [ -e "$wt_path/$item" ] || [ -L "$wt_path/$item" ]; then
            score=$((score+1))
          fi
        done
        if [ "$score" -gt "$best_score" ]; then
          best_score=$score
          best_path=$wt_path
        fi
        ;;
    esac
  done < <(git worktree list --porcelain 2>/dev/null || true)

  echo "$best_path"
}

FROM_PATH=""
FORCE=0
ABSOLUTE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --from)
      shift
      [ $# -gt 0 ] || { err "--from requires a path"; exit 1; }
      FROM_PATH="$1"; shift ;;
    --force) FORCE=1; shift ;;
    --absolute) ABSOLUTE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

require_git_root
REPO_ROOT=$(git rev-parse --show-toplevel)

if [ -z "$FROM_PATH" ]; then
  FROM_PATH=$(detect_source_worktree)
  if [ -z "$FROM_PATH" ]; then
    err "Cannot auto-detect source worktree. Provide --from <path>."
    exit 1
  fi
  log "Auto-detected source: $FROM_PATH"
fi

# Normalize FROM_PATH to absolute path
FROM_PATH=$(abspath "$FROM_PATH")

if [ ! -d "$FROM_PATH" ]; then
  err "--from path does not exist or is not a directory: $FROM_PATH"
  exit 1
fi

log "Repository root: $REPO_ROOT"
log "Linking from: $FROM_PATH"
log "Mode: $( [ "$ABSOLUTE" -eq 1 ] && echo absolute || echo relative ) symlinks"
log "Force: $FORCE | Dry-run: $DRY_RUN"

made_changes=0
for item in "${TARGETS[@]}"; do
  src="$FROM_PATH/$item"
  dst="$REPO_ROOT/$item"

  # Compute link target path
  if [ "$ABSOLUTE" -eq 1 ]; then
    link_target="$src"
  else
    link_target=$(relpath "$REPO_ROOT" "$src")
  fi

  # Check source existence (warn if missing, still allow link if desired)
  if [ ! -e "$src" ] && [ ! -L "$src" ]; then
    log "Warning: source missing '$src' (will link anyway)"
  fi

  # Handle existing destination
  if [ -e "$dst" ] || [ -L "$dst" ]; then
    # If it's already a matching symlink, skip
    if [ -L "$dst" ] && [ "$(readlink "$dst" 2>/dev/null || true)" = "$link_target" ]; then
      log "Exists: $item -> $link_target (ok)"
      continue
    fi
    if [ "$FORCE" -eq 1 ]; then
      log "Removing existing: $dst"
      [ "$DRY_RUN" -eq 1 ] || rm -rf -- "$dst"
    else
      log "Skip (exists, use --force to replace): $dst"
      continue
    fi
  fi

  log "Link: $item -> $link_target"
  if [ "$DRY_RUN" -eq 0 ]; then
    ln -s -- "$link_target" "$dst"
    made_changes=1
  fi
done

if [ "$DRY_RUN" -eq 1 ]; then
  log "Dry-run complete. No changes made."
else
  if [ "$made_changes" -eq 1 ]; then
    log "Symlink setup complete."
  else
    log "Nothing to do. All links already in place."
  fi
fi

