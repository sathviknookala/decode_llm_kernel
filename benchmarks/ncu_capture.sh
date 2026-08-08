#!/usr/bin/env bash
# Nsight Compute counter capture: measured DRAM and L2 traffic for the operator.
# The sweep's logical_eff_gbps is a logical floor with rows at 180% of empirical bandwidth;
# these counters are what would turn it into a measurement.
# Usage: benchmarks/ncu_capture.sh [PYTHON]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-python}"
OUTDIR="$REPO_ROOT/results/profiling"
mkdir -p "$OUTDIR"

# dram__ is the DRAM traffic the logical count is not; lts__ is the L2 traffic that is the
# standing explanation for rows above 100%, so both are needed to settle it.
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
gpu__time_duration.sum"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found on PATH; skipping Nsight Compute capture." >&2
  exit 127
fi

ncu --version | head -3

# Counter access is a driver-level permission, not a flag. Probe once with a throwaway kernel
# so a denied run fails here with the fix rather than after profiling every configuration.
PERM_LOG="$(mktemp)"
trap 'rm -f "$PERM_LOG"' EXIT
if ! ncu --metrics dram__bytes_read.sum --csv --target-processes all \
     "$PYTHON" -c "import torch; (torch.randn(256, device='cuda') * 2).sum().item()" \
     >"$PERM_LOG" 2>&1; then
  :
fi
if grep -q ERR_NVGPUCTRPERM "$PERM_LOG"; then
  cat >&2 <<'MSG'
ERR_NVGPUCTRPERM: this user cannot read GPU performance counters, so no counter capture is
possible regardless of the metrics requested. This is the same class of block as --lock-clocks
on this machine. It needs a root-level change, one of:

  sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
      > /etc/modprobe.d/nvidia-profiling.conf' && sudo update-initramfs -u && reboot
  # or, without a reboot: run the capture itself as root

Until then results/LIMITATIONS.md is correct that no measured DRAM traffic exists.
MSG
  exit 3
fi

for HEADS in mha gqa; do
  for IMPL in eager compile; do
    NAME="ncu_${IMPL}_${HEADS}_b32"
    echo "=== capturing $NAME ==="
    ncu --metrics "$METRICS" \
      --csv \
      --target-processes all \
      --print-summary per-kernel \
      --log-file "$OUTDIR/${NAME}.csv" \
      "$PYTHON" "$REPO_ROOT/benchmarks/profile_operator.py" \
        --single "$HEADS" --single-batch 32 --impls "$IMPL" \
        --no-trace --iters 5 --warmup 3
    echo "--- $NAME ---"
    "$PYTHON" "$REPO_ROOT/benchmarks/ncu_report.py" --csv "$OUTDIR/${NAME}.csv" \
      --head-label "$HEADS" --batch 32 || true
  done
done

echo "Nsight Compute artifacts written to $OUTDIR"
