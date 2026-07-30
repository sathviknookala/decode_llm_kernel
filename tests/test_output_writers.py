import json

from benchmarks.benchmark_utils import write_csv, write_json


def test_write_csv_accepts_a_bare_filename(tmp_path, monkeypatch):
    """Regression: os.path.dirname('bare.csv') == '', and os.makedirs('') used to raise."""
    monkeypatch.chdir(tmp_path)
    write_csv("bare.csv", [{"a": 1, "b": 2}])
    assert (tmp_path / "bare.csv").exists()


def test_write_json_accepts_a_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json("bare.json", {"a": 1})
    with open(tmp_path / "bare.json") as f:
        assert json.load(f) == {"a": 1}
