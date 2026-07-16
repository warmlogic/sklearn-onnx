#!/usr/bin/env bash
# Sets up the venv used by repro_tree_nan_routing.py. See README.md for context.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
VENV="$HERE/.venv"

uv venv "$VENV" --python 3.13
uv pip install --python "$VENV/bin/python" \
    -e "$REPO" \
    "numpy==2.5.1" \
    "scikit-learn==1.9.0" \
    "onnx==1.22.0" \
    "onnxruntime==1.25.1"
#   numpy / scikit-learn / onnx are the latest releases as of 2026-07-16.
#   onnxruntime is held back from the true latest (1.27.0) on purpose --
#   see "Why onnxruntime is pinned" in README.md.

echo "Venv ready: $VENV/bin/python"
echo "Run the repro with: $VENV/bin/python $HERE/repro_tree_nan_routing.py"
