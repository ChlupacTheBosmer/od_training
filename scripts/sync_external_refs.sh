#!/usr/bin/env bash
# Sync reference repositories listed in external/README.md.
#
# Usage:
#   bash scripts/sync_external_refs.sh
#   bash scripts/sync_external_refs.sh --pull
#   bash scripts/sync_external_refs.sh --disable-push

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README_PATH="$ROOT_DIR/external/README.md"
if [[ ! -f "$README_PATH" ]]; then
  README_PATH="$ROOT_DIR/external/readme.md"
fi
EXTERNAL_DIR="$ROOT_DIR/external"

PULL_EXISTING=false
DISABLE_PUSH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull)
      PULL_EXISTING=true
      shift
      ;;
    --disable-push)
      DISABLE_PUSH=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Sync reference repositories listed in external/README.md.

Options:
  --pull          Pull existing repositories after clone check.
  --disable-push  Set origin push URL to DISABLED for safety.
  -h, --help      Show this help.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$README_PATH" ]]; then
  echo "Missing external README file (expected external/README.md or external/readme.md)" >&2
  exit 1
fi

mkdir -p "$EXTERNAL_DIR"

CLONE_COUNT="$(grep -E '^[[:space:]]*git clone[[:space:]]+https?://[^[:space:]]+' "$README_PATH" | wc -l | tr -d ' ')"

if [[ "$CLONE_COUNT" == "0" ]]; then
  echo "No 'git clone ...' lines found in $README_PATH"
  exit 0
fi

echo "Syncing $CLONE_COUNT external reference repositories..."

grep -E '^[[:space:]]*git clone[[:space:]]+https?://[^[:space:]]+' "$README_PATH" | while IFS= read -r line; do
  # shellcheck disable=SC2086
  read -r _ _ url target _extra <<< "$line"

  repo_name="$(basename "${url%.git}")"

  if [[ -n "${target:-}" ]]; then
    if [[ "$target" = /* ]]; then
      dest="$target"
    else
      dest="$EXTERNAL_DIR/$target"
    fi
  else
    dest="$EXTERNAL_DIR/$repo_name"
  fi

  echo ""
  echo "Repo: $url"
  echo "Path: $dest"

  if [[ -d "$dest/.git" ]]; then
    echo "Status: already cloned"
    if [[ "$PULL_EXISTING" == true ]]; then
      echo "Action: pulling latest changes..."
      git -C "$dest" pull --ff-only
    fi
  else
    echo "Action: cloning..."
    git clone "$url" "$dest"
  fi

  if [[ "$DISABLE_PUSH" == true && -d "$dest/.git" ]]; then
    if git -C "$dest" remote get-url origin >/dev/null 2>&1; then
      git -C "$dest" remote set-url --push origin DISABLED
      echo "Action: disabled push URL for origin"
    fi
  fi
done

echo ""
echo "Done."
