import argparse
import csv
import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_utils import REPO_ROOT, logical_bytes, write_json
from benchmarks.workload import DTYPES, HEAD_CONFIGS

L2_SECTOR_BYTES = 32

DRAM_READ = "dram__bytes_read.sum"
DRAM_WRITE = "dram__bytes_write.sum"
L2_READ_SECTORS = "lts__t_sectors_op_read.sum"
L2_WRITE_SECTORS = "lts__t_sectors_op_write.sum"

# ncu prefixes its CSV with banner lines before the header row; the header is the first line
# carrying the metric columns.
HEADER_MARKERS = ("Metric Name", "Metric Value")


def parse_ncu_csv(text):
    """Rows of {kernel, metric, unit, value} from `ncu --csv` output.

    Keyed on column names rather than positions, because ncu's column set moves between
    versions. Non-numeric metric values (ncu emits a units row under some versions) are
    dropped rather than coerced to zero.
    """
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if all(m in l for m in HEADER_MARKERS)), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    out = []
    for row in reader:
        raw = (row.get("Metric Value") or "").replace(",", "").strip()
        try:
            value = float(raw)
        except ValueError:
            continue
        out.append({
            "kernel": (row.get("Kernel Name") or "").strip(),
            "metric": (row.get("Metric Name") or "").strip(),
            "unit": (row.get("Metric Unit") or "").strip(),
            "value": value,
        })
    return out


def aggregate(rows):
    """Sum each metric over every kernel launch, and keep the per-kernel split.

    Summing is right here because one invocation of the eager path is 18 kernels; what the
    logical count is being compared against is the traffic of the whole invocation.
    """
    totals = {}
    per_kernel = {}
    # ncu emits one row per (launch, metric), so a single metric's row count is the launch
    # count. max over metrics rather than any one of them, in case a metric is unavailable for
    # some launches; dividing the total by the metric count would understate it the same way.
    rows_per_metric = {}
    launches_per_kernel = {}
    for r in rows:
        totals[r["metric"]] = totals.get(r["metric"], 0.0) + r["value"]
        per_kernel.setdefault(r["kernel"], {})
        k = per_kernel[r["kernel"]]
        k[r["metric"]] = k.get(r["metric"], 0.0) + r["value"]
        rows_per_metric[r["metric"]] = rows_per_metric.get(r["metric"], 0) + 1
        launches_per_kernel.setdefault(r["kernel"], {})
        lk = launches_per_kernel[r["kernel"]]
        lk[r["metric"]] = lk.get(r["metric"], 0) + 1
    return {"totals": totals, "per_kernel": per_kernel,
            "kernel_launches": max(rows_per_metric.values(), default=0),
            "launches_per_kernel": {k: max(v.values(), default=0)
                                    for k, v in launches_per_kernel.items()}}


def measured_bytes(totals):
    """DRAM bytes as counted, and L2 bytes reconstructed from sectors."""
    dram = totals.get(DRAM_READ, 0.0) + totals.get(DRAM_WRITE, 0.0)
    l2_sectors = totals.get(L2_READ_SECTORS, 0.0) + totals.get(L2_WRITE_SECTORS, 0.0)
    return {
        "dram_read_bytes": totals.get(DRAM_READ, 0.0),
        "dram_write_bytes": totals.get(DRAM_WRITE, 0.0),
        "dram_total_bytes": dram,
        "l2_sectors": l2_sectors,
        "l2_bytes": l2_sectors * L2_SECTOR_BYTES,
    }


def logical_bytes_for(head_label, batch, cache_alloc_len, dtype_label, iters=1):
    """The sweep's logical count for one configuration, without allocating anything.

    Meta tensors carry dtype and shape, which is all logical_bytes reads, so this runs with no
    GPU and no permission to profile.
    """
    head = next((h for h in HEAD_CONFIGS if h[0] == head_label), None)
    if head is None:
        raise ValueError(f"unknown head label {head_label!r}, expected one of "
                         f"{[h[0] for h in HEAD_CONFIGS]}")
    _, num_q_heads, num_kv_heads, head_dim = head
    dtype = DTYPES[dtype_label]

    def meta(*shape, dt=dtype):
        return torch.empty(shape, dtype=dt, device="meta")

    lb = logical_bytes(
        meta(batch, num_q_heads, head_dim), meta(batch, num_kv_heads, head_dim),
        meta(batch, num_kv_heads, head_dim),
        meta(cache_alloc_len, head_dim, dt=torch.float32),
        meta(cache_alloc_len, head_dim, dt=torch.float32),
        meta(batch, cache_alloc_len, num_kv_heads, head_dim),
        meta(batch, cache_alloc_len, num_kv_heads, head_dim))
    return {k: v * iters if isinstance(v, int) else v for k, v in lb.items()
            if k.endswith("_bytes")}


def compare(measured, logical):
    """Measured traffic against the logical count. A DRAM ratio below 1 is the L2 service the
    sweep's >100%-of-bandwidth rows were attributed to but never shown."""
    total = logical["total_bytes"]
    return {
        **measured,
        "logical_total_bytes": total,
        "dram_over_logical": measured["dram_total_bytes"] / total if total else "",
        "l2_over_logical": measured["l2_bytes"] / total if total else "",
    }


def main():
    ap = argparse.ArgumentParser(
        description="Measured DRAM and L2 traffic from an ncu CSV, against the logical count")
    ap.add_argument("--csv", required=True, help="output of `ncu --csv --log-file ...`")
    ap.add_argument("--head-label", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--cache-alloc-len", type=int, default=2048)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--iters", type=int, default=1,
                    help="timed invocations inside the profiled process")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"no such ncu csv: {args.csv}")
    with open(args.csv) as f:
        rows = parse_ncu_csv(f.read())
    if not rows:
        raise SystemExit(
            f"{args.csv} carries no metric rows -- an ERR_NVGPUCTRPERM run produces a log with "
            f"no counters, which is not the same as a configuration that moved no bytes")

    agg = aggregate(rows)
    logical = logical_bytes_for(args.head_label, args.batch, args.cache_alloc_len,
                                args.dtype, args.iters)
    result = compare(measured_bytes(agg["totals"]), logical)

    print(f"kernels: {len(agg['per_kernel'])} distinct")
    print(f"logical total : {logical['total_bytes']:>14,.0f} B")
    print(f"measured DRAM : {result['dram_total_bytes']:>14,.0f} B  "
          f"({result['dram_over_logical']:.2f}x logical)")
    print(f"measured L2   : {result['l2_bytes']:>14,.0f} B  "
          f"({result['l2_over_logical']:.2f}x logical)")
    if args.json_out:
        write_json(args.json_out, {"source_csv": os.path.relpath(args.csv, REPO_ROOT),
                                   "per_kernel": agg["per_kernel"], **result})
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
