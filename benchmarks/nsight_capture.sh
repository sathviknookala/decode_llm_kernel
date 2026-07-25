#!/usr/bin/env bash
# Nsight Systems capture for the operator's launch structure.
# Usage: benchmarks/nsight_capture.sh [PYTHON]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-python}"
OUTDIR="$REPO_ROOT/results/profiling"
mkdir -p "$OUTDIR"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found on PATH; skipping Nsight capture." >&2
  exit 127
fi

nsys --version

for HEADS in mha gqa; do
  for IMPL in eager compile; do
    NAME="nsys_${IMPL}_${HEADS}_b32"
    echo "=== capturing $NAME ==="
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cuda-memory-usage=false \
      --force-overwrite=true \
      --output="$OUTDIR/$NAME" \
      "$PYTHON" "$REPO_ROOT/benchmarks/profile_operator.py" \
        --single "$HEADS" --single-batch 32 --impls "$IMPL" \
        --no-trace --iters 50 --warmup 25
    nsys stats --report cuda_gpu_kern_sum --format csv \
      --output "$OUTDIR/$NAME" "$OUTDIR/$NAME.nsys-rep" >/dev/null
    echo "--- $NAME kernel summary ---"
    cat "$OUTDIR/${NAME}_cuda_gpu_kern_sum.csv" 2>/dev/null || true
  done
done

echo "Nsight artifacts written to $OUTDIR"
