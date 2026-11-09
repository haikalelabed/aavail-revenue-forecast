#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=fd6182c4458bccb56024047a1a85f506a90caf98
source_dir="${1:-data/upstream}"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --no-checkout https://github.com/aavail/ai-workflow-capstone.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
  echo "Use the course repository at revision $revision" >&2
  exit 1
fi
python -m forecast.ingest "$source_dir/cs-train" data/train.csv
python -m forecast.ingest "$source_dir/cs-production" data/observed.csv
