import json

from benchmarks.benchmark_utils import read_env_doc

FLAT = {"timestamp_utc": "2026-08-05T00:00:00Z", "gpu_name": "X", "git_sha": "abc"}
NESTED = {"environment": FLAT, "rungs": [{"rung": "full"}]}


def _write(tmp_path, doc, name="e.env.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def test_reads_the_nested_shape_every_probe_writes(tmp_path):
    assert read_env_doc(_write(tmp_path, NESTED))["gpu_name"] == "X"


def test_still_reads_the_flat_shape_the_committed_amdahl_file_uses(tmp_path):
    """amdahl_probe.env.json was left flat so the committed code matches the committed data.
    Normalising the writer must not orphan the file it already produced."""
    assert read_env_doc(_write(tmp_path, FLAT))["git_sha"] == "abc"


def test_a_doc_that_is_neither_yields_nothing_rather_than_garbage(tmp_path):
    assert read_env_doc(_write(tmp_path, {"rungs": []})) == {}


def test_a_missing_file_is_not_an_error(tmp_path):
    assert read_env_doc(str(tmp_path / "nope.json")) == {}
    assert read_env_doc(None) == {}


def test_the_committed_amdahl_env_json_parses(tmp_path):
    import os
    from benchmarks.benchmark_utils import REPO_ROOT
    p = os.path.join(REPO_ROOT, "results/raw/amdahl_probe.env.json")
    if os.path.exists(p):
        assert read_env_doc(p).get("model_id") == "mistralai/Mistral-7B-v0.1"
