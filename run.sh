#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm -it \
  -v "$PROJECT_DIR:$PROJECT_DIR" \
  -w "$PROJECT_DIR" \
  roof_analysis
