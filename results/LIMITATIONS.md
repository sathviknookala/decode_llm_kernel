# Limitations and blockers, per results file

What each result set in this directory does **not** establish. Read this before quoting any
number from these files. Methodology is in `docs/benchmark_methodology.md`.

Everything here was measured on a single machine: one RTX PRO 4000 Blackwell (`sm_120`,
23.4 GiB, driver 575.64.03), torch 2.11.0+cu128, CUDA runtime 12.8 / nvcc 12.9. Exact
provenance for each run is in the adjacent `*.env.json`.

---

## `raw/operator_baseline_v2.csv` (+ `operator_baseline_v2.env.json`)

**`logical_eff_gbps` and `pct_of_empirical_bw` are not physical DRAM traffic.** The numerator
is a logical floor: it excludes intermediates an implementation materializes (eager's
`rotate_half` concatenation, gathered trig rows, cast temporaries), excludes whole-sector
overhead on scattered writes, and counts bytes that may be served from L2. The denominator is
streaming throughput on a different access pattern. Treat the ratio as orientation, not an
efficiency score. **Blocker:** measured traffic needs Nsight Compute counters, not yet run.

**v2 has only the streaming denominator.** The current rig additionally emits
`pct_of_scattered_write_bw`, based on random K/V writes into an MHA b=128, alloc=2048, FP32 cache.
That fixes the access-pattern mismatch but not the byte-mix mismatch: the operator numerator counts
all logical traffic, while the scattered reference counts K/V source reads and K/V cache writes.
Neither column is an efficiency score, and the scattered ratio can exceed 100%.

**The timed region is an operator call, not a bare kernel.** It includes PyTorch dispatch and
one iterator step for position rotation. That is identical across implementations, so
comparisons are fair, but these are not pure kernel times.

**This file predates the CUDA-graph baseline now available in the rig.** Its direct-path
"dispatch share" remains arithmetic, not a graph comparison. In subsequent runs `graph_eager`
and `graph_compile` bind every other input pointer at capture and copy each next position set into
a static buffer before replay. That copy is a real launch inside the timed region, matching the
dynamic-position serving contract rather than presenting replay alone as the operator cost.

**This file predates start/end clock-state capture.** The rig now records SM and memory clocks,
temperature, power draw, and active throttle reasons at both boundaries. On this machine the
opt-in clock lock is unavailable to the current user: `nvidia-smi -lgc` reports that permission to
change clocks is required, so the rig logs the denial and continues unlocked before restoring only
locks it actually acquired. Do not retroactively read v2's few-percent deltas as clock-controlled.

**Two configurations are unmeasured, not measured-and-fast.** `mha b=128 alloc=2048 fp32`
(both position modes) is skipped: it needs 17.2 GB peak contiguous cache (2 sets × 8.6 GB)
against a 12 GB budget on a 23.4 GiB card. Rows are present with `impl=skipped` and the
reason. This is a preview of why paged caches exist, not a gap in the sweep.

**Coverage bounds of the matrix.** `head_dim` is fixed at 128 (the Llama-2 anchor) and rotary
is always full — no partial-rotary or awkward head dims (Checkpoint D). Contiguous
token-major cache only; no paged, fragmented, or head-major layouts (Checkpoint F). Head
configurations are MHA 32/32 and GQA 32/8 only; MQA is covered in the byte-formula tests but
not benchmarked. `torch.compile` is measured with one backend and default mode, recorded in
the env JSON; other modes are unexplored.

**GQA shapes are real; GQA tensors are not.** The swept GQA layout (32 q heads : 8 kv heads,
`head_dim` 128, `rope_theta` 10000) is the attention shape of **Mistral-7B-v0.1**
(Apache-2.0, ungated; Jiang et al. 2023, arXiv:2310.06825), cross-checked against its
published `config.json` by `tests/test_anchor_models.py`, and our RoPE tables are bit-exact
against `MistralRotaryEmbedding`. That establishes the *configuration* is real, **not** that
the numbers are: every timed tensor is still synthetic `randn`, no model weights are loaded,
and no real activation distribution or decode loop is involved. Latency here is dtype- and
shape-driven, so synthetic inputs are appropriate — but nothing in this file is a measurement
of Mistral-7B-v0.1 the model.

**Timed rows use the identity request mapping and packed Q/K/V.** Non-identity mappings and
fused-projection strides are exercised by the validation gate, not by the timing path, so this
file says nothing about their cost. The current rig adds explicit columns and an adjacent-control
supplemental cross at b∈{1,32}; those measurements belong to the subsequent baseline, not v2.

**The attention consumer is not modelled at all.** This operator only writes the cache. A
layout or kernel that speeds writes may slow the attention reads that follow, and that
trade-off is invisible in this file.

**`cache_alloc_len` is an allocation size.** Flat latency across it is the expected result and
is not evidence about launch-bounding — the operator touches one RoPE row and one K/V slot per
request and never scans the context.

---

## `raw/bandwidth_reference.json`

**The continuity reference is streaming, not a bound on this operator.** Large contiguous `copy_`
and `add_` remain the friendliest pattern. `reference_gbps` is deliberately still the 512 MiB copy
median so old and new result files stay comparable.

**The size sweep locates the knee, but only for FP32 copies.** The 1/4/16/64/256/512 MiB sweep
measures cache-resident throughput through 16 MiB and DRAM throughput from 64 MiB upward on this
48 MiB-L2 GPU. It does not generalize the knee or throughput to other dtypes or GPUs.

**The scattered reference matches one important shape, not every row.** It uses random K/V slot
writes at MHA b=128, alloc=2048, FP32 granularity with one 8 GiB K+V cache set. That is the shape
behind the old 88%-of-streaming headline and stays under the 12 GB budget, but batch size, head
layout, dtype, and allocation length differ for other operator rows.

**Medians of a short run** (50 iterations after warmup) with no clock-state provenance or control.
The current rig records boundary state and can request a lock, but this historical file predates
that change; treat its ±few-percent differences as noise.

---

## `profiling/` (`profile_summary.json`, `trace_*.json`, `nsys_*_cuda_gpu_kern_sum.csv`)

**Kernel counts are exact; the attribution of the remaining time is inferred.** The traces
establish kernels per invocation and device time per invocation directly. That the rest of the
measured span is host-side dispatch follows from the arithmetic (1.8 µs of GPU work inside a
25.5 µs call). These v2-era traces predate the graph comparison; use the subsequent baseline's
graph-vs-direct latency for the measured recoverable launch share.

**Profiling perturbs what it measures.** Reported device times come from profiled runs and are
not the timing path's numbers; use `operator_baseline_v2.csv` for latency.

**Narrow config coverage.** bf16, ragged positions, `alloc=2048`, b∈{1,32}, MHA and GQA only.
Kernel counts for other dtypes and batch sizes are not established, though the inductor fusion
count is expected to vary (already 1 vs 2 kernels between MHA b=32 and GQA b=32).

**nsys traces cover the whole process.** Setup allocation and RNG kernels appear alongside the
operator; the NVTX range marks the measured region. The `.nsys-rep` and `.sqlite` files are
gitignored as large regenerable binaries — regenerate with `benchmarks/nsight_capture.sh`.
Only the kernel-summary CSVs are tracked.

**No Nsight Compute.** No occupancy, register pressure, sector efficiency, or achieved-bandwidth
counters — so no statement here explains *why* a kernel takes the time it does.

---

## `raw/archive/operator_baseline_v1_preB1.csv`

Superseded and **not comparable** to v2. See `raw/archive/README.md` for the five reasons
(byte accounting, latency naming, uniform-only positions, no validation gate, column rename).
Retained only so earlier claims can be traced to the data that produced them.
