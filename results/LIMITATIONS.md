# Limitations and blockers, per results file

What each result set in this directory does **not** establish. Read this before quoting any
number from these files. Methodology is in `docs/benchmark_methodology.md`.

Everything here was measured on a single machine: one RTX PRO 4000 Blackwell (`sm_120`,
23.4 GiB, driver 575.64.03), torch 2.11.0+cu128, CUDA runtime 12.8 / nvcc 12.9. Exact
provenance for each run is in the adjacent `*.env.json`.

---

## `raw/operator_baseline_v3.csv` (+ `operator_baseline_v3.env.json`)

1908 rows (1904 timed), four impls, four head layouts, 0 validation failures. Provenance is a
clean `df96e9c`; each row names the gate cases that cleared it in `validation_cases`.

**Every row in this file is overhead-bound, so the bandwidth columns describe almost nothing.**
At b=128 the four layouts move KV traffic differing by 64x (mha 8192 bytes per token, mqa 128)
and the compiled device medians are 0.0272 ms and 0.0304 ms respectively -- the layout carrying
64x the bytes is the faster one. No configuration here is limited by memory traffic. Do not read
any latency in this file as evidence about a memory system.

**This file's own writes never leave L2, and that turns out not to matter.** The timed loop cycles
8 position sets and reuses one set of input tensors, so it rewrites the same slots on every
invocation -- a write working set of 0.67x L2 even at the largest configuration that fits. The
flatness above could therefore have been an artifact of cache residency rather than a property of
the operator. It is not: `raw/l2_residency_probe.csv` sweeps that working set to 42.67x L2 and
latency moves 1.9%. Read the flatness as real, but read it knowing this file alone could not
establish it.

**`pct_of_empirical_bw` is not an efficiency score, and this file proves it.** 16 rows exceed
100%, peaking at **180% of empirical bandwidth and 273% of the scattered reference**
(`graph_compile`, mha, b=128, fp32, uniform). The numerator is a logical byte count that ignores
where bytes are served from; uniform positions rewrite one slot per request, which stays
L2-resident. v2 reported 88% on this column and it read like saturation. It was not.
**Blocker:** real traffic needs Nsight Compute counters, still not run.

**The launch share is measured, but its floor is not decomposed.** `graph_eager` and
`graph_compile` remove 62-69% of the corresponding direct path's device median, and
`graph_compile` sits at ~0.0107 ms in essentially every configuration. That flatness suggests a
fixed cost rather than work, but the graph thunk is **two** launches -- `positions.copy_()` into
the static buffer, then `replay()` -- and this file does not separate them. Until it does, the
addressable headroom below the graph path is unknown.

**The graph path's timed region includes the position copy, deliberately.** Capture binds every
other input pointer, so serving with dynamic positions requires that copy. Presenting replay
alone would measure a workload that cannot be run.

**The ragged-positions finding changed shape and is not settled.** Compiled b=1 reproduces v2's
penalty at 1.41x ragged-over-uniform, but `graph_eager` and `graph_compile` show 1.00x at the
same configurations. The access pattern is identical, so v2's "rewriting one slot kept it
L2-resident" explanation cannot be the whole story -- but the graph floor may simply be too
coarse to resolve a difference this small. Treat the mechanism as open.

**Strided QKV was measured and costs nothing here.** strided-over-packed device median is 0.993
at b=1 and 1.002 at b=32. That is a real answer to v2's open question, bounded to those two
batch sizes: the serving cross runs only at b in {1, 32}, so nothing is established at b=8 or
b=128.

**Four configurations are unmeasured, not measured-and-fast.** `mha` and `mha96` at b=128,
alloc=2048, fp32 need 17.2 GB and 12.9 GB peak contiguous cache (2 sets each) against a 12 GB
budget. Rows are present with `impl=skipped`. A preview of why paged caches exist.

**The timed region is an operator call, not a bare kernel.** For direct impls it includes
PyTorch dispatch and one iterator step for position rotation. Identical across implementations,
so comparisons are fair, but these are not pure kernel times.

**Clock state is recorded at both boundaries but not controlled.** `nvidia-smi -lgc` is
unavailable to this user (permission denied); the rig logs the denial and continues unlocked.
Treat few-percent differences as noise.

**Shapes are real; tensors are not.** All four layouts are real released models' attention
shapes -- `mha` Llama-2-7B, `gqa` Mistral-7B-v0.1, `mqa` Falcon-7B, `mha96` Phi-3-mini-4k --
cross-checked against published `config.json` by `tests/test_anchor_models.py`, and the RoPE
tables are bit-exact against `MistralRotaryEmbedding`. Every timed tensor is still synthetic
`randn`: no weights, no real activation distribution, no decode loop.

**Coverage bounds.** Rotary is always full -- no partial-rotary (Checkpoint D). Contiguous
token-major cache only; no paged, fragmented, or head-major layouts (Checkpoint F).
`torch.compile` runs one backend at default mode, recorded in the env JSON. `rope_theta` is
10000 for every row, which is why anchors at other thetas were not adopted.

**The attention consumer is not modelled at all.** This operator only writes the cache. A layout
or kernel that speeds writes may slow the attention reads that follow, and that trade-off is
invisible here.

**`cache_alloc_len` is an allocation size.** Flat latency across it is expected and is not
evidence about launch-bounding -- the operator touches one RoPE row and one K/V slot per request
and never scans the context.

---

## `raw/l2_residency_probe.csv` (+ `l2_residency_probe.env.json`)

Sweeps the cache-write working set from 0.00x to 42.67x this device's 50.3 MB L2, by drawing each
cycled position set from its own disjoint slot range. Exists to test whether the baseline sweep's
L2-resident writes were flattering its flat-latency result.

**It answers one question and is not a baseline.** No oracle is allocated and no validation gate
runs -- the impls were already cleared by the v3 sweep, and skipping the second cache set is what
lets `mha b=128 fp32` fit the footprint budget here. Do not quote its latencies as operator
timings; they were measured under a deliberately hostile addressing pattern that no baseline uses.

**Only the write side is swept.** Q/K/V, the trig tables and the graph's captured output buffer
are the same allocations on every invocation, so the read side stays cache-resident throughout.
That is a fair model of decode, where Q/K/V arrive hot from the projection, but it means the probe
bounds cache-write cost specifically, not total memory cost.

**`compile` and `graph_compile` only**, b=128 and alloc=2048 only, ragged packed/identity only,
one seed, 100 iterations per point. Nothing here speaks to small batch, and small batch is where
the operator is most overhead-dominated to begin with.

**The marginal-rate figure is an inference, not a counter reading.** Comparing across head layouts,
a 127x range of cache-write traffic costs 2.0 us, which implies a marginal rate near 2 TB/s and
therefore that the stores are not being paid for at memory speed. That is arithmetic over two
measured medians; confirming *why* still needs Nsight Compute.

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
not the timing path's numbers; use `operator_baseline_v3.csv` for latency.

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

## `raw/archive/operator_baseline_v2_preV3.csv`

Superseded and **not comparable** to v3, and not regenerable from any committed state: its
`env.json` records a dirty tree at `1db04be`, a commit containing neither `validation.py` nor
`workload.py`. See `raw/archive/README.md` for the six reasons (provenance, autograd state,
added impls, added head layouts, added serving-layout columns, gate strength). Several claims
derived from it -- "3.8-4.0x eager at b>=8", "88% of achievable, near-saturated" -- are
contradicted by v3.

---

## `raw/archive/operator_baseline_v1_preB1.csv`

Superseded and **not comparable** to anything later. See `raw/archive/README.md` for the five
reasons (byte accounting, latency naming, uniform-only positions, no validation gate, column
rename).
Retained only so earlier claims can be traced to the data that produced them.
