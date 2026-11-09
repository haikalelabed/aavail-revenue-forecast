#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir="${1:-data/upstream}"
bash scripts/fetch_data.sh "$source_dir"
python -m forecast.evaluate "$source_dir/cs-train" "$source_dir/cs-production" reports
