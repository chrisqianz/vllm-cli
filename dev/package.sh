#!/usr/bin/env bash
# vllm-cli release zip builder
#
# Packages the repo into a clean, deployable source zip:
#   <repo-parent>/vllm-cli-<VERSION>.zip
#
# The exclusion list below is the cumulative result of past packaging
# reviews — do not remove entries without re-checking the resulting zip.
#
# Usage:  bash dev/package.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="$(tr -d '[:space:]' < "$REPO/VERSION")"
OUT_DIR="$(dirname "$REPO")"
ZIP="$OUT_DIR/vllm-cli-$VER.zip"
STAGE_PARENT="$(mktemp -d)"
STAGE="$STAGE_PARENT/vllm-cli-$VER"

trap 'rm -rf "$STAGE_PARENT"' EXIT
mkdir -p "$STAGE"

rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.v0*-check' \
  --exclude '.pi' \
  --exclude '.gsd' \
  --exclude '.claude' \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'examples' \
  --exclude 'PROFILES_v0.2.9.md' \
  --exclude 'dev' \
  --exclude 'RELEASE_NOTES_*' \
  "$REPO/" "$STAGE/"

# Only the current version's release notes ship with the package
if [ -f "$REPO/RELEASE_NOTES_v$VER.md" ]; then
  cp "$REPO/RELEASE_NOTES_v$VER.md" "$STAGE/"
else
  echo "WARN: RELEASE_NOTES_v$VER.md not found" >&2
fi

cd "$STAGE_PARENT"
rm -f "$ZIP"
zip -qr "$ZIP" "$(basename "$STAGE")"

echo "zip: $ZIP"
unzip -l "$ZIP" | awk 'END { print $2 " entries" }'

# --- verification gates (no grep -q on pipes: early exit + pipefail = false SIGPIPE failure) ---
LISTING="$(unzip -l "$ZIP")"
BAD="$(printf '%s\n' "$LISTING" | grep -cE '/(dev|\.venv|\.git|\.v0|examples|__pycache__)/|\.DS_Store|\.pyc$' || true)"
if [ "$BAD" -ne 0 ]; then
  echo "ERROR: forbidden path found inside zip (dev/, .venv, .git, .v0*-check, examples, __pycache__, .pyc, .DS_Store)" >&2
  exit 1
fi
case "$LISTING" in
  *"vllm-cli-$VER/VERSION"*) : ;;
  *) echo "ERROR: VERSION missing" >&2; exit 1 ;;
esac
[ "$(unzip -p "$ZIP" "vllm-cli-$VER/VERSION" | tr -d '[:space:]')" = "$VER" ] || { echo "ERROR: VERSION mismatch" >&2; exit 1; }
echo "verification OK: no dev/ or build artifacts, VERSION=$VER"
