#!/usr/bin/env bash
# Release: set the version in pyproject.toml, commit, tag vX.Y.Z, push.
# Usage: ./release.sh 0.5.1
# If the given version already matches pyproject.toml, skips the bump and just tags.
# Pushing the v* tag triggers the image build workflow.
set -euo pipefail

cd "$(dirname "$0")"

VERSION="${1:?usage: ./release.sh X.Y.Z}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "error: version must be X.Y.Z, got '$VERSION'" >&2; exit 1; }
TAG="v$VERSION"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BRANCH" == "main" ]] || { echo "error: on branch '$BRANCH', expected main" >&2; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "error: working tree not clean" >&2; exit 1; }
! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "error: tag $TAG already exists" >&2; exit 1; }

CURRENT="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)"
if [[ "$VERSION" == "$CURRENT" ]]; then
    echo "pyproject.toml already at $VERSION; tagging HEAD without a bump commit"
else
    sed -i "s/^version = \"$CURRENT\"/version = \"$VERSION\"/" pyproject.toml
    uv lock
    git add pyproject.toml uv.lock
    git commit -m "Bump version to $VERSION"
fi

git tag "$TAG"
git push origin main "$TAG"
echo "released $TAG"
