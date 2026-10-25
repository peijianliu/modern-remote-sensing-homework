"""复用已有缓存，准备缺失输入并运行补充实验。"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT


def command(script: str) -> None:
    """运行项目脚本；子进程失败时终止当前批次。"""
    # 逐行转发标准输出，使子进程输出同时写入主命令日志。
    arguments = [sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts" / script)]
    with subprocess.Popen(
        arguments,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, arguments)


def resolve_matlab(executable: str | None = None, config: dict | None = None) -> Path:
    """按命令行、环境变量、配置文件、PATH 的顺序定位 MATLAB。

    配置项为 runtime.matlab_executable；相对路径以附件根目录为基准。
    显式配置无效时抛出 FileNotFoundError，便于定位路径错误。
    """
    configured = (
        executable
        or os.environ.get("MATLAB_EXECUTABLE")
        or (config or {}).get("runtime", {}).get("matlab_executable")
    )
    if configured:
        path = Path(configured).expanduser()
        candidate = path if path.is_absolute() else ROOT / path
        if candidate.is_file():
            return candidate.resolve()
        located = shutil.which(str(configured))
        if located:
            return Path(located).resolve()
        raise FileNotFoundError(f"MATLAB executable not found: {configured}")
    located = shutil.which("matlab")
    if located:
        return Path(located).resolve()
    raise FileNotFoundError(
        "MATLAB cache absent; set --matlab, MATLAB_EXECUTABLE, "
        "runtime.matlab_executable, or add matlab to PATH"
    )


def prepare(config: dict | None = None, matlab_executable: str | None = None) -> None:
    """复用观测资料，仅为缺失的正演缓存调用 MATLAB。"""
    needed = [
        ROOT / "data/raw/public_validation/zenodo_3580078/Darklab_snow_SSA.zip",
        ROOT / "data/raw/public_validation/zenodo_3580825/reflectance_spectra_field_2010_2011.zip",
        ROOT / "data/raw/public_validation/zenodo_4031446/16FEB13190431_landsat_30m_fsca.tif",
        ROOT / "data/raw/public_validation/USGS_L8_OLI/Landsat8OLI7.json",
    ]
    if not all(path.exists() for path in needed):
        command("acquire_public_validation.py")
    scenes = json.loads((ROOT / "configs/worldview_validation.json").read_text(encoding="utf-8"))[
        "worldview"
    ]["scenes"]
    if not all(
        (ROOT / "data/cache/public_validation" / scene["id"] / "source_manifest.json").exists()
        for scene in scenes
    ):
        command("acquire_landsat_reference.py")
    caches = [
        ("data/cache/forward_models_55deg.mat", "[55]"),
        (
            "data/cache/forward_models_worldview_exact_angles.mat",
            "[" + " ".join(str(scene["solar_zenith_deg"]) for scene in scenes) + "]",
        ),
    ]
    for cache, angles in caches:
        if not (ROOT / cache).exists():
            matlab = resolve_matlab(matlab_executable, config)
            subprocess.run(
                [str(matlab), "-batch", f"addpath('matlab'); run_forward('{cache}',{angles});"],
                cwd=ROOT,
                check=True,
            )


def run(
    report_only: bool = False, config: dict | None = None, matlab_executable: str | None = None
) -> None:
    """运行补充实验，或核验已有结果并生成汇总报告。"""
    if not report_only:
        prepare(config, matlab_executable)
        for script in [
            "publication_figures.py",
            "empirical_validation.py",
            "aviris_band_sensitivity.py",
            "worldview_validation.py",
            "empirical_mixtures.py",
            "review_reanalysis.py",
            "conservative_maps.py",
        ]:
            command(script)
    command("verify_supplement.py")
    command("build_review_report.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run snow-mixture supplement experiments")
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.json"))
    parser.add_argument("--matlab", help="MATLAB executable path or command")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    configuration = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run(args.report_only, configuration, args.matlab)
