#!/usr/bin/env bash
set -euo pipefail

SOURCE_BRANCH="${1:-codex/create-static-workflow-rag-cli-project-9zlwjk}"
TARGET_BRANCH="${2:-master}"
NEW_BRANCH="${3:-pr2-replayed-on-master}"

FILES=(
  .gitignore
  README.md
  app
  data/docs/novel.md
  data/eval_set.jsonl
  pyproject.toml
)

echo "[1/5] fetch latest refs"
git fetch origin

echo "[2/5] checkout target branch: ${TARGET_BRANCH}"
git checkout "${TARGET_BRANCH}"
git pull --ff-only origin "${TARGET_BRANCH}"

echo "[3/5] create replay branch: ${NEW_BRANCH}"
git checkout -B "${NEW_BRANCH}"

echo "[4/5] replay feature files from ${SOURCE_BRANCH}"
git checkout "${SOURCE_BRANCH}" -- "${FILES[@]}"

if git diff --quiet --cached && git diff --quiet; then
  echo "No changes to commit after replay."
  exit 0
fi

echo "[5/5] commit replayed changes"
git add "${FILES[@]}"
git commit -m "chore: replay PR2 changes on top of ${TARGET_BRANCH}"

echo
echo "Done. Push with:"
echo "  git push -u origin ${NEW_BRANCH} --force-with-lease"
