#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <tag-name> <tag-message>"
  exit 1
fi

TAG_NAME="$1"
shift
TAG_MSG="$*"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is not clean; commit changes before tagging"
  exit 1
fi

if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
  echo "tag already exists: $TAG_NAME"
  exit 1
fi

git tag -a "$TAG_NAME" -m "$TAG_MSG"
git push origin "$TAG_NAME"

echo "created and pushed tag: $TAG_NAME"
