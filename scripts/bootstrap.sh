#!/usr/bin/env bash
# Bootstrap the deck-motion-forecast repo with the Claude Code kit.
# Usage: ./bootstrap.sh /path/to/deck-motion-forecast
set -euo pipefail

TARGET="${1:-$(pwd)}"
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Bootstrapping into: $TARGET"
mkdir -p "$TARGET"/{configs/{sim/vessels,data,model,experiment},src/dmf/{sim,data,models,train,eval,deploy,viz},scripts,tests,mss,artifacts,results,docs}

# Python package markers
find "$TARGET/src/dmf" -type d -exec touch {}/__init__.py \;

# Claude Code kit
cp -r "$KIT_DIR/.claude" "$TARGET/"
cp "$KIT_DIR/CLAUDE.md" "$TARGET/"

# gitignore
cat > "$TARGET/.gitignore" << 'GITIGNORE'
.venv/
__pycache__/
*.py[cod]
artifacts/
*.parquet
*.onnx
*.ckpt
.pytest_cache/
.mypy_cache/
.ruff_cache/
GITIGNORE

# Makefile
cat > "$TARGET/Makefile" << 'MAKEFILE'
CFG ?= configs/experiment/e01_baselines.yaml

.PHONY: data train eval bench test lint all
data:  ; python scripts/generate_corpus.py --config configs/data/default.yaml
train: ; python scripts/train.py --config $(CFG)
eval:  ; python scripts/evaluate.py --all-regimes
bench: ; python scripts/benchmark.py --export --parity --latency
test:  ; pytest -q
lint:  ; ruff check src tests && mypy src
all: data train eval bench
MAKEFILE

echo "Done. Next:"
echo "  cd $TARGET && git init"
echo "  cp /path/to/IMPLEMENTATION_PLAN.md docs/"
echo "  claude"
