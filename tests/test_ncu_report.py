import pytest

from benchmarks.ncu_report import (
    DRAM_READ,
    DRAM_WRITE,
    L2_READ_SECTORS,
    L2_SECTOR_BYTES,
    L2_WRITE_SECTORS,
    aggregate,
    compare,
    logical_bytes_for,
    measured_bytes,
    parse_ncu_csv,
)

# Shape of `ncu --csv --log-file` output: banner lines, then a header, then one row per
# (kernel, metric). Fixture-derived: this machine denies counter access (ERR_NVGPUCTRPERM), so
# the parser has never seen real ncu output and this format is from the documented schema.
NCU_CSV = '''==PROF== Connected to process 1234
==PROF== Profiling "triton_poi_fused_0" - 0: 0%....50%....100% - 1 pass
"ID","Process ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","triton_poi_fused_0","Command line profiler metrics","dram__bytes_read.sum","byte","1,024"
"0","1234","triton_poi_fused_0","Command line profiler metrics","dram__bytes_write.sum","byte","2,048"
"0","1234","triton_poi_fused_0","Command line profiler metrics","lts__t_sectors_op_read.sum","sector","64"
"0","1234","triton_poi_fused_0","Command line profiler metrics","lts__t_sectors_op_write.sum","sector","32"
"1","1234","CatArrayBatchedCopy","Command line profiler metrics","dram__bytes_read.sum","byte","512"
"1","1234","CatArrayBatchedCopy","Command line profiler metrics","dram__bytes_write.sum","byte","512"
'''


def test_the_parser_skips_the_banner_and_finds_the_header():
    rows = parse_ncu_csv(NCU_CSV)
    assert len(rows) == 6
    assert rows[0]["kernel"] == "triton_poi_fused_0"
    assert rows[0]["metric"] == DRAM_READ


def test_thousands_separators_are_not_read_as_text():
    """ncu writes 1,024 for a kilobyte; float() rejects it and a coerced zero would read as a
    kernel that moved nothing."""
    rows = parse_ncu_csv(NCU_CSV)
    assert rows[0]["value"] == pytest.approx(1024.0)


def test_a_units_row_is_dropped_rather_than_coerced_to_zero():
    text = NCU_CSV + '"2","1234","k","Command line profiler metrics","dram__bytes_read.sum","byte","n/a"\n'
    assert len(parse_ncu_csv(text)) == 6


def test_output_without_a_header_yields_nothing():
    """An ERR_NVGPUCTRPERM run writes a log with no counters at all; that must not aggregate
    to a confident zero."""
    assert parse_ncu_csv("==ERROR== ERR_NVGPUCTRPERM - no permission\n") == []


def test_metrics_are_summed_over_every_kernel_in_the_invocation():
    """Eager runs 18 kernels per invocation, so a per-kernel figure is not what the logical
    count is being compared against."""
    totals = aggregate(parse_ncu_csv(NCU_CSV))["totals"]
    assert totals[DRAM_READ] == pytest.approx(1024 + 512)
    assert totals[DRAM_WRITE] == pytest.approx(2048 + 512)


def test_the_per_kernel_split_survives_aggregation():
    per_kernel = aggregate(parse_ncu_csv(NCU_CSV))["per_kernel"]
    assert set(per_kernel) == {"triton_poi_fused_0", "CatArrayBatchedCopy"}
    assert per_kernel["CatArrayBatchedCopy"][DRAM_READ] == pytest.approx(512)


def test_l2_sectors_become_bytes_at_the_documented_sector_size():
    got = measured_bytes(aggregate(parse_ncu_csv(NCU_CSV))["totals"])
    assert got["l2_sectors"] == pytest.approx(96)
    assert got["l2_bytes"] == pytest.approx(96 * L2_SECTOR_BYTES)
    assert got["dram_total_bytes"] == pytest.approx(1024 + 2048 + 512 + 512)


def test_missing_metrics_count_as_absent_not_as_zero_traffic():
    got = measured_bytes({DRAM_READ: 100.0})
    assert got["dram_total_bytes"] == pytest.approx(100.0)
    assert got["l2_sectors"] == 0


def test_logical_bytes_need_no_gpu_and_no_allocation():
    """Meta tensors carry dtype and shape, which is all logical_bytes reads -- so this runs on
    a machine that cannot profile at all."""
    lb = logical_bytes_for("mha", 32, 2048, "bf16")
    assert lb["total_bytes"] > 0
    assert lb["read_bytes"] + lb["write_bytes"] == lb["total_bytes"]


def test_logical_bytes_follow_the_dtype_and_the_head_layout():
    bf16 = logical_bytes_for("mha", 32, 2048, "bf16")["total_bytes"]
    fp32 = logical_bytes_for("mha", 32, 2048, "fp32")["total_bytes"]
    mqa = logical_bytes_for("mqa", 32, 2048, "bf16")["total_bytes"]
    assert fp32 > bf16
    assert mqa < bf16                       # one KV head instead of 32


def test_an_unknown_head_label_is_refused():
    with pytest.raises(ValueError, match="unknown head label"):
        logical_bytes_for("mha256", 32, 2048, "bf16")


def test_a_dram_ratio_below_one_is_what_l2_service_looks_like():
    """The sweep's rows above 100% of empirical bandwidth are attributed to L2 service. That
    claim predicts measured DRAM below the logical count, which is the comparison this makes."""
    got = compare(measured_bytes({DRAM_READ: 400.0, DRAM_WRITE: 100.0,
                                  L2_READ_SECTORS: 30.0, L2_WRITE_SECTORS: 20.0}),
                  {"total_bytes": 1000})
    assert got["dram_over_logical"] == pytest.approx(0.5)
    assert got["l2_over_logical"] == pytest.approx(50 * L2_SECTOR_BYTES / 1000)


def test_a_zero_logical_count_leaves_the_ratio_blank():
    got = compare(measured_bytes({DRAM_READ: 1.0}), {"total_bytes": 0})
    assert got["dram_over_logical"] == ""
