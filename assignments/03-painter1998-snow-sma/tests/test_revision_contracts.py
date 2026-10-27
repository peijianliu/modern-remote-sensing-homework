"""空验证、构建失败保留旧结果、系数可行性与配对误差区间回归。"""

import hashlib
import json

import numpy as np
import pytest

from scripts import build_notebooks as builder
from scripts.verify_artifacts import verify_runs
from snowmix.diagnostics import coefficient_status, paired_rmse_difference


def test_empty_experiment_verification_fails(tmp_path):
    with pytest.raises(ValueError, match="No experiment manifests"):
        verify_runs(tmp_path)


def test_verification_requires_each_stage_and_nonempty_metrics(tmp_path):
    run = tmp_path / "example_r001"
    run.mkdir()
    manifest = {"stage": "example", "status": "complete"}
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "metrics.json").write_text("{}", encoding="utf-8")
    result = verify_runs(tmp_path, ("example",))
    assert result["status"] == "failed" and "Empty experiment metrics" in result["errors"][0]
    (run / "metrics.json").write_text('{"pixels": 10}', encoding="utf-8")
    assert verify_runs(tmp_path, ("example",))["status"] == "passed"
    assert verify_runs(tmp_path, ("example", "missing"))["status"] == "failed"
    manifest["status"] = "running"
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_runs(tmp_path, ("example",))["status"] == "failed"


def write_mock_site(path, filenames):
    path.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in filenames:
        (path / name).write_text(name, encoding="utf-8")
        manifest[name] = hashlib.sha256(name.encode()).hexdigest()
    (path / "site_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_atomic_build_drops_removed_files_and_keeps_failed_old_site(tmp_path, monkeypatch):
    output = tmp_path / "site"
    write_mock_site(output, ["old.html"])

    def fail(path, timeout):
        write_mock_site(path, ["partial.html"])
        raise ValueError("simulated notebook failure")

    monkeypatch.setattr(builder, "render_site", fail)
    with pytest.raises(ValueError, match="simulated"):
        builder.build(output)
    assert (output / "old.html").read_text(encoding="utf-8") == "old.html"
    assert not (output / "partial.html").exists()
    monkeypatch.setattr(
        builder, "render_site", lambda path, timeout: write_mock_site(path, ["new.html"])
    )
    monkeypatch.setattr("scripts.verify_site.verify", lambda path: None)
    builder.build(output)
    assert (output / "new.html").is_file() and not (output / "old.html").exists()
    assert not list(tmp_path.glob(".site-stage-*"))


def test_changed_generated_site_is_preserved(tmp_path):
    write_mock_site(tmp_path, ["page.html"])
    (tmp_path / "page.html").write_text("my comments", encoding="utf-8")
    with pytest.raises(ValueError, match="edited"):
        builder.check_previous_output(tmp_path)


def test_physical_status_includes_background_and_shade():
    coefficients = np.array([[0.4, 0.3], [-0.01, 0.6], [1.2, -0.2], [0.4, 0.8], [np.nan, 0]])
    original = coefficients.copy()
    result = coefficient_status(coefficients, 1 - coefficients.sum(1))
    np.testing.assert_array_equal(result["status"], [1, 2, 3, 4, 0])
    assert result["feasible"].sum() == 1
    np.testing.assert_array_equal(coefficients, original)
    with pytest.raises(ValueError):
        coefficient_status(coefficients, np.zeros(1))


def test_paired_rmse_interval_uses_same_blocks_and_checks_support():
    positions = np.column_stack((np.arange(10) * 50, np.zeros(10, int)))
    truth, first, second = np.zeros(10), np.ones(10) * 0.2, np.ones(10) * 0.1
    value = paired_rmse_difference(truth, first, second, positions)
    assert value["difference"] == pytest.approx(-0.1)
    assert value["ci_low"] == pytest.approx(-0.1)
    assert value["ci_high"] == pytest.approx(-0.1)
    sparse = paired_rmse_difference(truth[:2], first[:2], second[:2], positions[:2])
    assert sparse["ci_low"] is None and sparse["spatial_blocks"] == 2
