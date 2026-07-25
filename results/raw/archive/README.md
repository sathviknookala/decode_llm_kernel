# Archived results

## `operator_baseline_v1_preB1.csv`

Superseded by `results/raw/operator_baseline_v2.csv` (Checkpoint B.1). Do **not** compare
the two directly: v1 used different byte accounting and different latency naming.

Differences that make v1 numbers non-comparable:

- **Byte accounting.** v1 hard-coded FP32 for the cos/sin term and derived every other term
  from a single dtype argument. v2 reads `element_size()` from each real tensor.
- **Latency naming.** v1's `op_call_ms` was an amortized loop measurement mislabelled as an
  operator-call latency; v2 reports `amortized_call_ms` and a separate
  `synchronized_call_ms`. v1 also carried a redundant `p50_ms` alongside `median_ms`.
- **Positions.** v1 pinned every request to `max_seq - 1`, so every timed invocation rewrote
  one cache slot (an L2-resident best case). v2 defaults to deterministic ragged positions and
  rotates position sets across invocations.
- **No validation gate.** v1 timed `torch.compile` without checking its output against the
  oracle. v2 refuses to record timings for a configuration that fails validation.
- **Column rename.** v1's `context_len` is v2's `cache_alloc_len` (the operator never scans
  the preceding context; the value only sets cache allocation and the valid position range).
