"""MATLAB 定位、缓存复用及命令行参数传递检查。"""

import json
from pathlib import Path

import pytest

from scripts import run_supplement
from snowmix import __main__ as cli


@pytest.mark.parametrize("source", ["cli", "environment", "config", "path"])
def test_matlab_location_priority(tmp_path, monkeypatch, source):
    executables = {}
    for name in ["cli", "environment", "config", "path"]:
        executable = tmp_path / f"{name} MATLAB" / "matlab.exe"
        executable.parent.mkdir()
        executable.touch()
        executables[name] = executable
    monkeypatch.delenv("MATLAB_EXECUTABLE", raising=False)
    monkeypatch.setattr(run_supplement.shutil, "which", lambda _: str(executables["path"]))
    config = (
        {"runtime": {"matlab_executable": str(executables["config"])}} if source != "path" else {}
    )
    if source in {"cli", "environment"}:
        monkeypatch.setenv("MATLAB_EXECUTABLE", str(executables["environment"]))
    requested = str(executables["cli"]) if source == "cli" else None
    assert run_supplement.resolve_matlab(requested, config) == executables[source].resolve()


def test_relative_matlab_path_is_resolved_from_package_root(tmp_path, monkeypatch):
    executable = tmp_path / "tools" / "matlab.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(run_supplement, "ROOT", tmp_path)
    monkeypatch.delenv("MATLAB_EXECUTABLE", raising=False)
    assert (
        run_supplement.resolve_matlab(config={"runtime": {"matlab_executable": "tools/matlab.exe"}})
        == executable.resolve()
    )


def test_invalid_explicit_matlab_path_does_not_fall_back(tmp_path, monkeypatch):
    fallback = tmp_path / "matlab.exe"
    fallback.touch()
    missing = tmp_path / "missing" / "matlab.exe"
    monkeypatch.setattr(
        run_supplement.shutil,
        "which",
        lambda command: str(fallback) if command == "matlab" else None,
    )
    with pytest.raises(FileNotFoundError, match="MATLAB executable not found"):
        run_supplement.resolve_matlab(str(missing))


def test_missing_matlab_reports_configuration_options(monkeypatch):
    monkeypatch.delenv("MATLAB_EXECUTABLE", raising=False)
    monkeypatch.setattr(run_supplement.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="MATLAB_EXECUTABLE"):
        run_supplement.resolve_matlab()


def prepare_input_tree(root: Path) -> None:
    """建立缓存存在性检查所需的占位文件；不模拟科学数据内容。"""
    relative_files = [
        "data/raw/public_validation/zenodo_3580078/Darklab_snow_SSA.zip",
        "data/raw/public_validation/zenodo_3580825/reflectance_spectra_field_2010_2011.zip",
        "data/raw/public_validation/zenodo_4031446/16FEB13190431_landsat_30m_fsca.tif",
        "data/raw/public_validation/USGS_L8_OLI/Landsat8OLI7.json",
        "data/cache/public_validation/scene/source_manifest.json",
    ]
    for relative in relative_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (root / "configs").mkdir()
    (root / "configs/worldview_validation.json").write_text(
        json.dumps({"worldview": {"scenes": [{"id": "scene", "solar_zenith_deg": 30.25}]}}),
        encoding="utf-8",
    )


def test_complete_caches_need_no_matlab_or_download(tmp_path, monkeypatch):
    prepare_input_tree(tmp_path)
    for name in ["forward_models_55deg.mat", "forward_models_worldview_exact_angles.mat"]:
        (tmp_path / "data/cache" / name).touch()
    monkeypatch.setattr(run_supplement, "ROOT", tmp_path)

    def unexpected_call(*args, **kwargs):
        pytest.fail("Cached inputs must not start downloads or MATLAB")

    monkeypatch.setattr(run_supplement, "command", unexpected_call)
    monkeypatch.setattr(run_supplement, "resolve_matlab", unexpected_call)
    run_supplement.prepare()


def test_missing_caches_use_matlab_argument_list_with_spaces(tmp_path, monkeypatch):
    prepare_input_tree(tmp_path)
    executable = tmp_path / "MATLAB tools" / "matlab.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(run_supplement, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        run_supplement.subprocess,
        "run",
        lambda arguments, **kwargs: calls.append((arguments, kwargs)),
    )
    run_supplement.prepare(matlab_executable=str(executable))
    assert len(calls) == 2
    for arguments, options in calls:
        assert arguments[:2] == [str(executable.resolve()), "-batch"]
        assert options == {"cwd": tmp_path, "check": True}
    assert "[55]" in calls[0][0][2]
    assert "[30.25]" in calls[1][0][2]


@pytest.mark.parametrize("stage", ["supplement", "review-report"])
def test_cli_passes_matlab_and_config(tmp_path, monkeypatch, stage):
    config = {"runtime": {"matlab_executable": "configured-matlab"}}
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_supplement, "run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        "sys.argv", ["snowmix", stage, "--config", str(config_path), "--matlab", "selected-matlab"]
    )
    cli.execute()
    assert calls == [
        {
            "report_only": stage == "review-report",
            "config": config,
            "matlab_executable": "selected-matlab",
        }
    ]
