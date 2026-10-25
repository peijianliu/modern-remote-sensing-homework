"""归档实验配置、源码哈希、指标和图件，并计算配对差异统计。"""

from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def configure_plots():
    """设置静态图件的字体、输出分辨率和坐标轴样式。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 10,
            "svg.fonttype": "none",
        }
    )


def write_json(path: Path, value):
    """以 UTF-8 写入 JSON，转换 NumPy 类型并拒绝非有限数值。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item.relative_to(ROOT))
        raise TypeError(type(item).__name__)

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def new_run(config: dict, stage: str) -> Path:
    """按项目、日期和版本创建新运行目录，保存源码快照与环境信息。"""
    base = ROOT / "experiments" / config["project"] / config["run_date"] / config["version"]
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while (base / f"{stage}_r{number:03d}").exists():
        number += 1
    path = base / f"{stage}_r{number:03d}"
    path.mkdir()
    for name in ("figures", "tables", "arrays", "rasters"):
        (path / name).mkdir()
    code = {}
    for directory in ("snowmix", "scripts", "matlab", "tests", "configs"):
        for source in (ROOT / directory).glob("**/*"):
            if source.is_file() and source.suffix in {".py", ".m", ".json"}:
                code[str(source.relative_to(ROOT))] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
    with zipfile.ZipFile(
        path / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in code:
            archive.write(ROOT / relative, relative)
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "matplotlib",
        "rasterio",
        "pyproj",
        "netCDF4",
        "shapely",
        "spectral",
    ):
        packages[name] = importlib.metadata.version(name)
    write_json(
        path / "run_manifest.json",
        {
            "stage": stage,
            "config": config,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "code_sha256": code,
            "packages": packages,
            "status": "running",
        },
    )
    print(f"RUN: {path.relative_to(ROOT)}", flush=True)
    return path


def finish_run(path: Path, metrics: dict):
    """写入指标和完成时间，更新当前项目的运行检查点。"""
    write_json(path / "metrics.json", metrics)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    manifest.update(status="complete", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(path / "run_manifest.json", manifest)
    (path / "run_summary.log").write_text(
        "Structured execution summary.\n"
        + json.dumps(
            {
                "started_utc": manifest["started_utc"],
                "completed_utc": manifest["completed_utc"],
                "stage": manifest["stage"],
                "status": "complete",
                "metrics_file": "metrics.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "last_completed_run": str(path.relative_to(ROOT)),
        "status": "active",
        "updated_utc": manifest["completed_utc"],
    }
    write_json(ROOT / "experiments/checkpoint.json", checkpoint)


def save_figure(fig, path: Path, name: str):
    """将图件同时保存为 PNG 和 SVG，然后关闭绘图对象。"""
    fig.savefig(path / "figures" / f"{name}.png", bbox_inches="tight")
    fig.savefig(path / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def paired_metrics(reference, estimate) -> dict:
    """在共同有限值上计算 bias、MAE、RMSE 和 1−SSE/SST。"""
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    valid = np.isfinite(reference) & np.isfinite(estimate)
    difference = estimate[valid] - reference[valid]
    if len(difference) == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None, "r2": None}
    variance = np.sum((reference[valid] - reference[valid].mean()) ** 2)
    return {
        "n": int(valid.sum()),
        "bias": float(difference.mean()),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "r2": float(1 - np.sum(difference**2) / variance) if variance > 0 else None,
    }
