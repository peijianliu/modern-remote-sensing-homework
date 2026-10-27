"""核验指定实验批次；空批次、缺少关键阶段或损坏产物均失败。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_STAGES = ("02_03_paper", "04_stability", "05_07_image")


def check_run(run: Path, manifest: dict) -> dict:
    """检查一份完整运行的指标与已声明栅格，返回实际检查数量。"""
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete run: {run.name}")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Empty experiment metrics: {run.name}")
    for relative, expected in manifest.get("output_sha256", {}).items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe declared output: {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Declared output checksum mismatch: {relative}")
    raster_count = 0
    for path in sorted((run / "rasters").glob("*.tif")):
        import rasterio

        with rasterio.open(path) as dataset:
            if dataset.crs is None or dataset.count == 0 or not all(dataset.descriptions):
                raise ValueError(f"Missing raster coordinates or band descriptions: {path.name}")
            array = dataset.read()
            good = np.isfinite(array).all(0)
            if not good.any():
                raise ValueError(f"Raster has no finite pixels: {path.name}")
            constrained = "fcls_spectral_fractions" in path.name or (
                manifest["stage"] == "08_qa_sensitivity"
                and path.name.endswith("_spectral_fractions.tif")
            )
            if constrained:
                if np.min(array[:, good]) < -1e-6:
                    raise ValueError(f"Negative constrained fraction: {path.name}")
                np.testing.assert_allclose(array[:, good].sum(0), 1, atol=2e-6)
            if "_grain_rms_condition" in path.name and np.nanmin(array[1]) < 0:
                raise ValueError(f"Negative spectral RMS: {path.name}")
            if "_qa.tif" in path.name and dataset.dtypes != ("uint16",):
                raise ValueError(f"Unexpected QA storage type: {path.name}")
        raster_count += 1
    if "synthetic_shape" in metrics:
        if metrics["synthetic_shape"] != [146, 101, 224]:
            raise ValueError("Synthetic grid differs from the prescribed experiment")
        for name in ("matched_200um_max_rms", "minimum_max_rms"):
            if not np.isfinite(metrics[name]) or metrics[name] >= 1e-8:
                raise ValueError(f"Matched synthetic recovery failed: {name}")
    if "coarse_shape" in metrics:
        import pandas as pd

        table = pd.read_csv(run / "tables/coarse_pixel_comparison.csv")
        if len(table) != np.prod(metrics["coarse_shape"]):
            raise ValueError("Coarse-pixel table shape differs from its metrics")
        if int(table.paired_valid.sum()) != metrics["matched_pixels"]:
            raise ValueError("Paired sample count differs from its metrics")
        if table.aviris_valid_area_fraction.max() > 1 + 1e-6:
            raise ValueError("Area coverage exceeds one")
    return {
        "run": run.name,
        "stage": manifest["stage"],
        "status": "complete",
        "rasters_checked": raster_count,
        "png_count": len(list((run / "figures").glob("*.png"))),
    }


def verify_runs(base: Path, required_stages: tuple[str, ...] = DEFAULT_STAGES) -> dict:
    """验证批次内每个运行并确认关键阶段齐全，不将零次循环视为通过。

    base 是包含各运行子目录的批次目录；required_stages 使用 manifest 中
    的 stage 字段匹配。历史未完成运行也会报告，需显式选取有效批次。
    """
    manifests = sorted(base.glob("*/run_manifest.json"))
    if not manifests:
        raise ValueError(f"No experiment manifests found: {base}")
    if not required_stages or len(set(required_stages)) != len(required_stages):
        raise ValueError("At least one unique required stage must be specified")
    results = {
        "runs": [],
        "errors": [],
        "required_stages": list(required_stages),
        "unselected_runs": [],
    }
    complete_stages = set()
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest.get("stage"), str):
                raise ValueError(f"Missing stage: {path.parent.name}")
            if manifest["stage"] not in required_stages:
                results["unselected_runs"].append(path.parent.name)
                continue
            results["runs"].append(check_run(path.parent, manifest))
            complete_stages.add(manifest["stage"])
        except (OSError, ValueError, KeyError, AssertionError) as error:
            results["errors"].append(f"{path.parent.name}: {error}")
    missing = sorted(set(required_stages) - complete_stages)
    if missing:
        results["errors"].append("Missing completed stages: " + ", ".join(missing))
    results["status"] = "failed" if results["errors"] else "passed"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        required=True,
        help="Batch containing run subdirectories with run_manifest.json",
    )
    parser.add_argument(
        "--required-stage", action="append", help="Repeat to name every required manifest stage"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional new verification JSON; existing files are preserved"
    )
    args = parser.parse_args()
    try:
        result = verify_runs(args.experiments_dir, tuple(args.required_stage or DEFAULT_STAGES))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
