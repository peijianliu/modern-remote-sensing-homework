"""从完整公开 AVIRIS 输入重算当前三模型，不依赖历史解混数组。

示例：python scripts/rebuild_public_case.py --source-root <prepared-inputs> --output build/full-case
source-root 只需本脚本列明的公开输入；不会读取 ECOSTRESS 或历史实验统计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from snowmix import image
from snowmix import teaching as t
from snowmix.case_analysis import fixed_radius_scan, select_scene_endmembers
from snowmix.diagnostics import coefficient_status


def digest(path: Path) -> str:
    """以分块读取核对完整影像，不把大型文件一次性载入内存。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def heldout_indices(
    cube: np.ndarray, valid: np.ndarray, wavelength: np.ndarray, excluded: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按预定指数和中位亮度选取四条留出光谱，排除端元周围 3 个像元。

    输入为全波段 (rows, columns, bands) 反射率；输出扁平索引、NDSI、NDVI。
    指数只用于选谱命名，不作为独立雪覆盖真值。选择过程不使用拟合残差。
    """

    def band(target: float) -> np.ndarray:
        return cube[..., np.argmin(abs(wavelength - target))]

    green, red, nir, swir = (band(v) for v in (0.55, 0.65, 0.86, 1.64))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi, ndvi = (green - swir) / (green + swir), (nir - red) / (nir + red)
    eligible = valid.copy()
    for row, column in excluded:
        eligible[max(0, row - 3) : row + 4, max(0, column - 3) : column + 4] = False
    candidates = [
        eligible & (ndsi > 0.7) & (green > 0.65) & (abs(ndvi) < 0.15),
        eligible & (ndsi >= 0.1) & (ndsi <= 0.7),
        eligible & (ndvi > 0.5) & (ndsi < 0),
        eligible & (abs(ndvi) < 0.1) & (ndsi < 0),
    ]
    selected = []
    for mask in candidates:
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError("Prespecified held-out spectral class has no candidates")
        distance = abs(green.ravel()[indices] - np.median(green.ravel()[indices]))
        selected.append(int(indices[np.lexsort((indices, distance))[0]]))
    return np.array(selected), ndsi, ndvi


def rebuild(source_root: Path, output: Path) -> dict:
    """重读 ENVI、逐像元重解、输出三模型共同域诊断及可携带的示例数据。"""
    if output.exists():
        raise FileExistsError("Choose a new output directory; previous experiments are preserved")
    started = time.perf_counter()
    config = json.loads((CASE / "configs/baseline.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (source_root / "data/cache/aviris_metadata.json").read_text(encoding="utf-8")
    )
    # 元数据可由 Windows 或 Linux 生成，文件字段均按项目相对路径解释。
    binary = source_root / metadata["binary_path"].replace("\\", "/")
    metadata["binary_path"] = binary.relative_to(source_root).as_posix()
    image.ROOT = source_root
    case = t.load_aviris_case()
    cube, valid, bands, transform, shape, crs, _ = image.prepare_roi(config, metadata)
    np.testing.assert_array_equal(valid, case["valid_mask"])
    wavelengths = np.array(metadata["wavelength_um"])
    np.testing.assert_array_equal(wavelengths[bands], case["wavelength_um"])
    observed = cube[..., bands][valid].astype(float)
    scene = select_scene_endmembers(cube, valid, wavelengths, bands)
    # 端元从完整观测按规则重新提取，冻结值仅用于事后回归核对。
    np.testing.assert_array_equal(scene["backgrounds"], case["backgrounds"])
    np.testing.assert_array_equal(scene["snow_spectra"], case["image_snow_spectra"])
    output.mkdir(parents=True)
    input_hash = digest(binary)
    manifest = {
        "stage": "public_aviris_rebuild",
        "status": "running",
        "inputs": {
            binary.relative_to(source_root).as_posix(): input_hash,
            "data/demo/aviris_case.npz": digest(CASE / "data/demo/aviris_case.npz"),
        },
    }

    def write(name: str, value: dict) -> None:
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    write("run_manifest.json", manifest)
    try:
        methods = [
            ("physical_MGS", case["snow_spectra"], case["radii_um"], "original"),
            ("image_MGS", scene["snow_spectra"], np.arange(1, 6), "original"),
            ("physical_FCLS", case["snow_spectra"], case["radii_um"], "fcls"),
        ]
        results = []
        for name, snow, radii, solver in methods:
            result = image.unmix_pixels(
                observed, snow, radii, scene["backgrounds"], config, solver=solver, progress=name
            )
            results.append(result)
        common = np.logical_and.reduce([np.isfinite(r["normalized"]).all(1) for r in results])
        positions = np.column_stack(np.nonzero(valid))
        selected, ndsi, ndvi = heldout_indices(
            cube,
            valid,
            wavelengths,
            np.vstack((case["background_row_col"], case["image_endmember_row_col"])),
        )
        # 保留原始系数与所有端元；零残差与有效分母不能代替面积比例可行性。
        maps = {
            "method_names": np.array([m[0] for m in methods]),
            "valid_mask": valid,
            "row_column": positions,
            "ndsi": ndsi.astype(np.float32),
            "ndvi": ndvi.astype(np.float32),
            "transform": np.array(list(transform)[:6]),
        }
        for field in ("fractions", "shade", "normalized", "rms", "qa", "radius_index"):
            maps[field] = np.stack([r[field] for r in results])
        np.savez_compressed(output / "coefficient_diagnostics.npz", **maps)
        rows = []
        for (name, _, _, _), result in zip(methods, results):
            status = coefficient_status(result["fractions"], result["shade"], tolerance=1e-7)
            for subset, keep in [("all_input", np.ones(len(observed), bool)), ("common", common)]:
                snow = result["normalized"][keep, 0]
                negative = result["fractions"][keep, 0] < -1e-7
                rows.append(
                    {
                        "method": name,
                        "subset": subset,
                        "n": int(keep.sum()),
                        "raw_snow_negative_fraction": float(negative.mean()),
                        "normalized_snow_negative_fraction": float(np.mean(snow < -1e-7)),
                        "normalized_snow_over_one_fraction": float(np.mean(snow > 1 + 1e-7)),
                        "physical_feasible_fraction": float(status["feasible"][keep].mean()),
                        "rms_mean": float(result["rms"][keep].mean()),
                        "negative_snow_median": float(
                            np.median(result["fractions"][keep, 0][negative])
                        )
                        if negative.any()
                        else 0.0,
                    }
                )
        pd.DataFrame(rows).to_csv(output / "coefficient_feasibility.csv", index=False)
        groups = np.full(len(observed), "other", dtype="U16")
        groups[
            (ndsi[valid] > 0.4) & (cube[..., np.argmin(abs(wavelengths - 0.55))][valid] > 0.1)
        ] = "high_ndsi"
        groups[(ndvi[valid] > 0.3) & (ndsi[valid] <= 0.4)] = "high_ndvi"
        group_rows = []
        for (name, _, _, _), result in zip(methods, results):
            for group in np.unique(groups):
                keep = common & (groups == group)
                raw = result["fractions"][keep, 0]
                group_rows.append(
                    {
                        "method": name,
                        "spectral_group": group,
                        "n": int(keep.sum()),
                        "negative_fraction": float(np.mean(raw < -1e-7)),
                        "raw_snow_mean": float(raw.mean()),
                        "normalized_snow_mean": float(result["normalized"][keep, 0].mean()),
                        "rms_mean": float(result["rms"][keep].mean()),
                    }
                )
        pd.DataFrame(group_rows).to_csv(output / "coefficient_spectral_groups.csv", index=False)
        with threadpool_limits(limits=1):
            scan = fixed_radius_scan(
                cube[..., bands].reshape(-1, bands.sum())[selected],
                case["snow_spectra"],
                case["radii_um"],
                case["backgrounds"],
            )
        np.savez_compressed(
            output / "heldout_spectra.npz",
            spectra=cube[..., bands].reshape(-1, bands.sum())[selected],
            row_column=np.column_stack((selected // shape[1], selected % shape[1])),
            ndsi=ndsi.ravel()[selected],
            ndvi=ndvi.ravel()[selected],
            **scan,
        )
        for result, key in zip(
            results, ("physical_snow_fraction", "image_snow_fraction", "fcls_snow_fraction")
        ):
            np.testing.assert_allclose(
                result["normalized"][:, 0], case[key][valid], atol=5e-7, rtol=2e-6, equal_nan=True
            )
        metrics = {
            "pixels": len(observed),
            "bands": int(bands.sum()),
            "background_and_snow_endmembers_reselected": True,
            "common_n": int(common.sum()),
            "matches_frozen_three_models": True,
            "heldout_pixels": selected.tolist(),
            "seconds": time.perf_counter() - started,
        }
        write("metrics.json", metrics)
        manifest.update(
            status="complete",
            output_sha256={
                p.name: digest(p) for p in sorted(output.iterdir()) if p.name != "run_manifest.json"
            },
        )
        write("run_manifest.json", manifest)
        print(json.dumps(metrics), flush=True)
        return metrics
    except Exception as error:
        manifest.update(status="failed", failure_type=type(error).__name__)
        write("run_manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.source_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
