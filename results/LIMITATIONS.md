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

**The launch share is measured; the floor beneath it is decomposed elsewhere.** `graph_eager` and
`graph_compile` remove 62-69% of the corresponding direct path's device median, and
`graph_compile` sits at ~0.0107 ms in essentially every configuration. What that flat number
consists of is not visible in this file: `raw/graph_floor_probe.csv` splits it into a 4.3 us
CUDA-event harness cost, a ~3.0 us position copy, a ~2.0 us replay launch, and 1.4-4.6 us of
operator work depending on size.

**Every `device_median_ms` in this file carries that 4.3 us harness cost.** It is common to all
impls, so the ratios between them are sound, but no absolute latency here should be read as the
operator's cost in a real decode loop, and differences smaller than a few microseconds are below
the instrument. `amortized_call_ms` is the column to use for those.

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

## `raw/graph_floor_probe.csv` (+ `graph_floor_probe.env.json`)

A six-rung ladder isolating what a `graph_compile` sample consists of: a Python call that launches
nothing, the position copy alone, replay of a graph holding one trivial kernel, both launches
together, the operator graph replayed with positions frozen, and `graph_compile` itself. Operator
work is the top rung minus the two-launch rung.

**The two latency definitions disagree, and the file reports both rather than choosing.** By event
median the operator does 0.03 us of work at b=32; by amortized loop it does ~1.4 us. Event
bracketing costs 4.3 us per sample and partially masks GPU execution, so the amortized figures are
the ones to use here. Neither is a counter reading.

**The frozen-position rung cannot serve decode** and is labelled `NOT SERVABLE` in the stage table.
It exists only to separate the copy from the replay. Its cache writes are also L2-resident, worth
about 2% per `raw/l2_residency_probe.csv`, so it is measured under friendlier conditions than the
serving path.

**"Minimal replay" is one trivial kernel, not an empty graph**, so the ~2.0 us attributed to replay
launch includes that kernel's own cost. It is an upper bound on pure replay overhead.

**`compile` lineage only**, five configurations, one seed. Nothing here covers `eager` or
`graph_eager`, and the 4.3 us harness figure is specific to this device and torch build.

**Subtraction across rungs assumes the components do not interact.** They share a stream and
serialise, which is why the arithmetic roughly closes, but the derived numbers carry about
+/- 0.5 us of run-to-run noise -- visible directly in the ladder, where nominally identical rungs
differ by that much between configurations.

---

## `raw/amdahl_probe.csv` (+ `amdahl_probe.env.json`)

91 rows (90 timed, 1 skipped, 0 errors) over Mistral-7B-v0.1 in HF transformers 5.12.1: six
ablation rungs x three execution modes x five surviving configurations. Exists to answer the one
question the operator microbenchmarks cannot -- what fraction of a real decode step this operation
is, and therefore what a fused kernel could win end to end.

**There is no serving engine anywhere in this file, and the bias is not one-directional.** The
denominator is HF's own decode loop; vLLM, FlashAttention, SGLang and TensorRT-LLM are not
installed. Attention is PyTorch SDPA, the cache is HF's contiguous `StaticCache`, and there is no
paged KV or continuous batching. Both halves of the ratio differ from a serving stack, in opposite
directions: HF's RoPE + append is unoptimized PyTorch (`op_removed` deletes 672 of 2123 traced
calls) where vLLM has hand-written kernels, which **inflates the numerator**; while at b=32 HF's
step is 62-108 ms against a ~27 ms weight-bandwidth floor, which **inflates the denominator**. At
b=1 the denominator is honest (27.9 ms measured against ~26.2 ms analytic) so only the numerator
caveat applies. **Do not quote any number here as "X% of a serving step"** without naming the
stack; it is a statement about HF's decode path.

**Five of the six rungs are numerically invalid by construction** and are marked
`numerically_valid=False`. `op_removed`, `rope_removed` and `append_removed` corrupt the model's
output; `op_doubled` double-rotates Q, which feeds attention directly. All hold shapes, dtypes and
allocations fixed, and `tests/test_amdahl_probe.py` asserts each one actually changes the logits --
a patch that silently did nothing would report a clean 0% operator share.

**The headline saving is an upper bound that is not achievable, and the file says so.** Removing
the operation saves 2.0-5.3% depending on configuration, but **doubling it costs 0.0-0.9%** under
both compiled modes. Mirror ratio 0.046-0.275. The two do not mirror, so most of what removal buys
is the compiler restructuring around a deleted mutation, which no faster kernel reproduces. The
realizable figure is the doubling slope: **0.81% at the gate configuration**, 15.7 us per layer.

**The same check reads 0.99-1.65 under eager, which is the positive control.** `op_doubled` is not
an insensitive instrument -- in eager, removal and doubling mirror each other, as a serial
operation should. It registers ~0 only under `torch.compile` and cudagraphs, on the same operation
in the same runs. Without those eager rows the demotion would rest on a null result.

**The 0.5 mirror cutoff was chosen after seeing the data.** The measured ratio at the gate is
0.275, so the demotion holds for any cutoff above that; `test_the_verdict_is_stable_across_
plausible_cutoffs` checks 0.35/0.5/0.75. It is not a wide margin and the ratio is reported beside
every verdict so the choice can be judged.

**Sliding-window eviction is disabled, so this models plain-append decode, not Mistral as
shipped.** Mistral-7B-v0.1 declares `sliding_window` 4096, which makes `StaticCache` build
`StaticSlidingWindowLayer` for every layer; that `update()` rolls the whole cache tensor once full,
O(window) per layer per step. Patching it away in a rung would have attributed eviction cost to
RoPE + append and inflated the saving. `decode_loop.load_model` sets `sliding_window = None` and
`layer_types` to all-`full_attention`, every row carries `layer_types_forced=True`, and the
constructed layers are asserted non-sliding. Phi-3-mini-4k declares 2047, so no ungated anchor
escapes this.

**`op_replaced_ref` is this project's reference, and it loses badly** -- 3% slower at b=1, 19% at
b=8, 21-29% at b=32, worsening with batch. The seam is numerically equivalent to HF's own path by
test, so this is a real result about the reference, not an artifact. Two deviations are baked into
it: HF gathers the cos/sin rows once per model step in `MistralRotaryEmbedding` rather than once
per layer, so the per-layer operation does not include the table gather the locked signature
implies; and HF's cache is `[B, Hkv, S, D]`, so the append writes through a transposed view.

**One configuration is unmeasured, not measured-and-fast.** b=32 alloc-equivalent ctx=1024 needs
weights 14.5 + KV 5.8 + ~3.0 GB workspace = 23.2 GB against a 22.5 GB budget on a 23.4 GB card.
It is present with `rung=skipped` and the arithmetic as its reason. The b=32 arm therefore stops
at ctx=512, and the batch arm at b=8/b=1 runs only at ctx=1024.

**The b=32 ctx sweep goes downward on purpose.** The operator's own cost is context-independent
but attention's KV read grows with context, so the operator's *share* peaks at small ctx --
2.96% at 128 falling to 2.02% at 512. Gating at ctx=1024, as the plan first proposed, would have
understated the operator's best case.

**Provenance is a dirty tree, and the delta is known.** `env.json` records `fd89bfed` with
`git_dirty=True`; the uncommitted delta is exactly the crash-safety fix, footprint guard and gate
licensing committed as `9e9f3c4`, which is the state that regenerates this file. Unlike the
archived v2, the producing code is committed and named. `env.json` for this probe is also written
flat rather than nested under an `environment` key as `graph_floor_probe.env.json` is; the code was
left as it ran rather than edited after the fact, so shape and data stay consistent.

**Clocks unlocked but stable**: 2647 MHz SM at both boundaries, throttle reasons `0x0` at both.
`nvidia-smi -lgc` remains denied to this user.

**Timing is amortized step time only.** Step scale is milliseconds so the 4.3 us event floor is
irrelevant here, but no per-layer figure in this file is measured directly -- every one is a step
delta divided by 32. Single seed, single GPU, single model, bf16 only, one prompt length per
configuration, no prefill measurements.

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
