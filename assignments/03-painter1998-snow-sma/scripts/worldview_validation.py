"""以 WorldView 雪覆盖参考评价两景 Landsat C2 L2 的迁移结果。

运行：.venv/Scripts/python.exe scripts/worldview_validation.py
使用预定光谱库、采样规则和方法，按场景及雪覆盖分层统计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from matplotlib.patches import Patch
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from rasterio.io import MemoryFile
from scipy.io import loadmat
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import (
    ROOT,
    configure_plots,
    finish_run,
    new_run,
    paired_metrics,
    save_figure,
    write_json,
)
from snowmix.image import unmix_pixels, write_raster
from snowmix.spectra import read_ecostress, read_usgs_ascii, response_matrix

QA_SOURCE = "https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands"
RSR_SOURCE = "https://landsat.usgs.gov/spectral-characteristics-viewer"
REFERENCE_SOURCE = "https://doi.org/10.5281/zenodo.4031446"
STRATA = ("no_snow", "mixed", "full_snow")


def load_json(path: Path) -> dict:
    """以 UTF-8 编码读取清单或配置。"""
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    """计算本地输入文件的哈希。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def landsat_quality(
    raw: np.ndarray, qa: np.ndarray, saturation: np.ndarray, reject_bits: list[int]
) -> np.ndarray:
    """使用共同 QA 掩膜，保留雪位 bit 5 及大于 1 的反射率。"""
    reject_mask = sum(1 << bit for bit in reject_bits)
    return ((qa & reject_mask) == 0) & (saturation == 0) & (raw != 0).all(axis=-1)


def align_reference(
    path: Path | str, shape: tuple[int, int], transform: Affine, crs: rasterio.crs.CRS
) -> tuple[np.ndarray, np.ndarray, dict]:
    """解析参考填充值，必要时按面积平均重投影并计算有效覆盖率。

    参考足迹外补零覆盖率，使边缘部分相交像元按完整目标面积计算支持。
    当前配对数据网格一致；跨 CRS 分支由常量守恒合成案例检查。
    """
    with rasterio.open(path) as dataset:
        raw = dataset.read(1)
        valid = (raw != -32768) & (raw >= 0) & (raw <= 1000)
        reference = np.full(raw.shape, np.nan, dtype=np.float32)
        reference[valid] = raw[valid] / 1000.0
        same = dataset.crs == crs and dataset.shape == shape and dataset.transform == transform
        metadata = {
            "source_crs": str(dataset.crs),
            "target_crs": str(crs),
            "source_transform": list(dataset.transform)[:6],
            "target_transform": list(transform)[:6],
            "source_shape": list(dataset.shape),
            "target_shape": list(shape),
            "source_nodata_attribute": dataset.nodata,
            "explicit_fill": -32768,
            "scale": 0.001,
            "source_valid_pixels": int(valid.sum()),
            "grid_exactly_matched": same,
            "reprojected": not same,
        }
        if same:
            return reference, valid.astype(np.float32), metadata
        output = np.full(shape, np.nan, dtype=np.float32)
        coverage = np.zeros(shape, dtype=np.float32)
        reproject(
            reference,
            output,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        support = np.pad(valid.astype(np.float32), 2, constant_values=0)
        reproject(
            support,
            coverage,
            src_transform=dataset.transform * Affine.translation(-2, -2),
            src_crs=dataset.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=None,
            dst_nodata=0,
            resampling=Resampling.average,
        )
        return output, coverage, metadata


def load_oli_responses() -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """读取 USGS 官方 OLI B2–B7 相对光谱响应，并记录负响应尾部样本。"""
    wavelengths, responses, notes = [], [], []
    for band in range(2, 8):
        file = ROOT / f"data/raw/public_validation/USGS_L8_OLI/Landsat8OLI{band}.json"
        rows = load_json(file)
        keys = list(rows[0])
        values = np.array([[float(row[keys[0]]), float(row[keys[1]])] for row in rows])
        values = values[np.argsort(values[:, 0])]
        assert np.all(np.diff(values[:, 0]) > 0)
        wavelengths.append(values[:, 0])
        responses.append(np.maximum(values[:, 1], 0))
        notes.append(
            {
                "band": band,
                "file": str(file.relative_to(ROOT)),
                "sha256": file_hash(file),
                "negative_tail_samples_set_to_zero": int((values[:, 1] < 0).sum()),
            }
        )
    return (
        wavelengths,
        responses,
        {
            "source": RSR_SOURCE,
            "wavelength_unit": "um",
            "convolution": "response-only integration on source cells",
            "bands": notes,
        },
    )


def endmembers(config: dict, angle: float, rw: list[np.ndarray], rs: list[np.ndarray]):
    """以相同 OLI 响应卷积积雪、松针和花岗岩光谱。"""
    cfg = config["worldview"]
    file = ROOT / cfg["forward_cache"]
    data = loadmat(file)
    angles = data["angles_deg"].ravel()
    angle_index = int(np.argmin(abs(angles - angle)))
    if abs(angles[angle_index] - angle) > 0.001:
        raise ValueError("Exact scene-angle forward cache unavailable")
    wanted = np.arange(cfg["radius_min_um"], cfg["radius_max_um"] + 1, cfg["radius_step_um"])
    full_radii = data["radii_um"].ravel()
    indices = [int(np.flatnonzero(full_radii == radius)[0]) for radius in wanted]
    ice_indices = np.flatnonzero(data["ice_options"].ravel() == cfg["ice_option"])
    if len(ice_indices) != 1:
        raise ValueError("Requested ice option is absent")
    wm, snow_coverage = response_matrix(data["wavelength_um"].ravel(), rw, rs)
    snow = data["snicar_albedo"][indices, :, angle_index, int(ice_indices[0])] @ wm
    background, records = [], []
    for name, filename, reader in [
        ("vegetation", "lodgepole_pine.asc", read_usgs_ascii),
        ("bare_granite", "granite_h2.txt", read_ecostress),
    ]:
        path = ROOT / "data/raw/spectra" / filename
        wavelength, reflectance, metadata = reader(path)
        weights, coverage = response_matrix(wavelength, rw, rs)
        background.append(reflectance @ weights)
        records.append(
            {
                "class": name,
                "file": str(path.relative_to(ROOT)),
                "sha256": file_hash(path),
                "metadata": metadata,
                "response_support": coverage.tolist(),
            }
        )
    if not np.isfinite(snow).all() or not np.isfinite(background).all():
        raise ValueError("OLI spectral support is incomplete")
    record = {
        "forward_cache": str(file.relative_to(ROOT)),
        "sha256": file_hash(file),
        "requested_and_effective_zenith_deg": float(angles[angle_index]),
        "snow_response_support": snow_coverage.tolist(),
        "radii_um": wanted.tolist(),
        "ice_option": cfg["ice_option"],
        "backgrounds": records,
        "model": "pure spherical optically thick SNICAR DHR approximates sensor surface reflectance",
        "endmembers_fitted_to_validation_labels": False,
    }
    return wanted, snow, np.column_stack(background), record


def strata(reference: np.ndarray, config: dict) -> np.ndarray:
    """依据独立参考划分预定雪覆盖层级。"""
    cuts = config["worldview"]["snow_strata"]
    return np.where(
        reference < cuts["no_snow_upper"],
        "no_snow",
        np.where(reference < cuts["full_snow_lower"], "mixed", "full_snow"),
    )


def block_bootstrap(
    reference: np.ndarray, estimate: np.ndarray, block: np.ndarray, seed: int, repeats: int
) -> dict:
    """按空间区块重采样，并由误差充分统计量计算区间。"""
    good = np.isfinite(reference) & np.isfinite(estimate)
    difference = estimate[good] - reference[good]
    ids, inverse = np.unique(block[good], return_inverse=True)
    if len(ids) < 5:
        return {
            "spatial_blocks": len(ids),
            "bias_interval_95": None,
            "mae_interval_95": None,
            "rmse_interval_95": None,
        }
    totals = np.column_stack(
        [
            np.bincount(inverse),
            np.bincount(inverse, weights=difference),
            np.bincount(inverse, weights=abs(difference)),
            np.bincount(inverse, weights=difference * difference),
        ]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), (repeats, len(ids)))
    samples = totals[indices].sum(axis=1)
    metrics = np.column_stack(
        (
            samples[:, 1] / samples[:, 0],
            samples[:, 2] / samples[:, 0],
            np.sqrt(samples[:, 3] / samples[:, 0]),
        )
    )
    limits = np.quantile(metrics, [0.025, 0.975], axis=0)
    return {
        "spatial_blocks": len(ids),
        "bias_interval_95": limits[:, 0].tolist(),
        "mae_interval_95": limits[:, 1].tolist(),
        "rmse_interval_95": limits[:, 2].tolist(),
        "meaning": "sampling distribution conditional on this scene, fixed endmembers and reference",
    }


def paired_delta(
    reference: np.ndarray,
    fixed: np.ndarray,
    multi: np.ndarray,
    block: np.ndarray,
    seed: int,
    repeats: int,
) -> dict:
    """在相同像元及相同重采样区块上配对比较方法。"""
    good = np.isfinite(reference) & np.isfinite(fixed) & np.isfinite(multi)
    first, second = fixed[good] - reference[good], multi[good] - reference[good]
    ids, inv = np.unique(block[good], return_inverse=True)
    value = {
        "n": int(good.sum()),
        "spatial_blocks": len(ids),
        "multi_minus_fixed_rmse": float(np.sqrt(np.mean(second**2)) - np.sqrt(np.mean(first**2)))
        if good.any()
        else None,
        "multi_minus_fixed_mae": float(np.mean(abs(second)) - np.mean(abs(first)))
        if good.any()
        else None,
    }
    if len(ids) < 5:
        value["rmse_delta_interval_95"] = None
        return value
    totals = np.column_stack(
        (
            np.bincount(inv),
            np.bincount(inv, weights=first**2),
            np.bincount(inv, weights=second**2),
            np.bincount(inv, weights=abs(first)),
            np.bincount(inv, weights=abs(second)),
        )
    )
    draws = np.random.default_rng(seed).integers(0, len(ids), (repeats, len(ids)))
    sums = totals[draws].sum(axis=1)
    delta = np.sqrt(sums[:, 2] / sums[:, 0]) - np.sqrt(sums[:, 1] / sums[:, 0])
    value["rmse_delta_interval_95"] = np.quantile(delta, [0.025, 0.975]).tolist()
    return value


def save_sample_maps(
    path: Path,
    scene: dict,
    reference: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    fits: dict,
    transform: Affine,
    crs,
    table: pd.DataFrame,
):
    """并列展示完整参考图与抽样像元的预测结果。"""
    x = (transform.c + (cols + 0.5) * transform.a) / 1000
    y = (transform.f + (rows + 0.5) * transform.e) / 1000
    bottom = (transform.f + reference.shape[0] * transform.e) / 1000
    right = (transform.c + reference.shape[1] * transform.a) / 1000
    extent = (transform.c / 1000, right, bottom, transform.f / 1000)
    fig, axes = plt.subplots(1, 4, figsize=(12, 10), layout="constrained")
    ramp = plt.get_cmap("Blues").copy()
    ramp.set_bad("#d9d9d9")
    im = axes[0].imshow(
        reference, extent=extent, interpolation="nearest", cmap=ramp, vmin=0, vmax=1
    )
    axes[0].set_title("WorldView 独立可见雪比例\n完整参考足迹", fontsize=10)
    for ax, method in zip(axes[1:3], ("fixed200_six_bands", "multi_six_bands")):
        ax.set_facecolor("#d9d9d9")
        im = ax.scatter(
            x, y, c=fits[method]["normalized"][:, 0], s=2.5, cmap=ramp, vmin=0, vmax=1, linewidths=0
        )
        ax.set_title(
            "固定 200 μm / FCLS" if method.startswith("fixed") else "多粒径 MIN RMS / FCLS",
            fontsize=10,
        )
    delta = fits["multi_six_bands"]["normalized"][:, 0] - table.reference_fsca.to_numpy()
    axes[3].set_facecolor("#d9d9d9")
    error = axes[3].scatter(
        x, y, c=delta * 100, s=2.5, cmap="PuOr_r", vmin=-100, vmax=100, linewidths=0
    )
    axes[3].set_title("多粒径 − WorldView\n覆盖差异 / pp", fontsize=10)
    for ax in axes:
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect("equal")
        ax.set_xlabel("UTM Easting / km")
        ax.set_ylabel("UTM Northing / km")
        ax.grid(False)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes[:3], shrink=0.5, label="雪比例（0–1；阴影归一化模型结果）")
    fig.colorbar(error, ax=axes[3], shrink=0.5, label="覆盖差异 / 百分点")
    axes[0].legend(
        handles=[Patch(facecolor="#d9d9d9", label="无数据 / 未抽样")], fontsize=8, loc="lower left"
    )
    bar_x, bar_y = extent[0] + 1, extent[2] + 3
    axes[0].plot([bar_x, bar_x + 5], [bar_y, bar_y], color="black", linewidth=2)
    axes[0].text(bar_x + 2.5, bar_y + 1, "5 km", ha="center", fontsize=8)
    fig.suptitle(
        f"{scene['roi_label']}；Landsat {scene['landsat_date']}；{crs}\n"
        f"后3幅仅{len(table):,}个预定随机样本，30 m；栅格北向上；预定随机像元结果",
        fontsize=11,
    )
    save_figure(fig, path, scene["id"] + "_sample_maps")


def scatter_and_residuals(path: Path, scene: dict, table: pd.DataFrame, fits: dict):
    """绘制按独立参考分层的散点图及光谱残差诊断。"""
    methods = list(fits)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    colors = {"no_snow": "#666666", "mixed": "#0072b2", "full_snow": "#cc79a7"}
    for ax, method in zip(axes.ravel(), methods):
        fit = fits[method]
        for label in STRATA:
            keep = table.stratum.to_numpy() == label
            count = int((keep & np.isfinite(fit["normalized"][:, 0])).sum())
            ax.scatter(
                table.reference_fsca[keep],
                fit["normalized"][keep, 0],
                s=5,
                alpha=0.28,
                color=colors[label],
                linewidths=0,
                label=f"{label} 有效{count}/{int(keep.sum())}",
            )
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="独立 WorldView 可见雪比例",
            ylabel="Landsat 阴影归一化雪比例",
            title=method,
        )
        legend = ax.legend(fontsize=8, markerscale=2)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
    fig.suptitle(
        f"{scene['landsat_date']}：预定抽样，日期偏移{scene['day_offset']} d；端元未按参考调参",
        fontsize=11,
    )
    save_figure(fig, path, scene["id"] + "_stratified_scatter")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    bands = np.arange(2, 8)
    for label in STRATA:
        keep = table.stratum.to_numpy() == label
        if not keep.any():
            continue
        for method, style in [("fixed200_six_bands", "--"), ("multi_six_bands", "-")]:
            residual = fits[method]["residual"][keep]
            axes[0].plot(
                bands,
                residual.mean(axis=0) * 100,
                linestyle=style,
                color=colors[label],
                label=label + " / " + method.split("_")[0],
            )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(
        xlabel="OLI band",
        ylabel="平均 E_c / 反射率百分点",
        xticks=bands,
        title="原始系数重建的光谱残差；6 bands",
    )
    axes[0].legend(fontsize=7)
    histogram_colors = ("#666666", "#0072b2", "#cc79a7", "#e69f00")
    for method, color in zip(methods, histogram_colors):
        difference = fits[method]["normalized"][:, 0] - table.reference_fsca.to_numpy()
        axes[1].hist(
            difference[np.isfinite(difference)] * 100,
            bins=np.linspace(-100, 100, 61),
            histtype="step",
            linewidth=1.4,
            color=color,
            label=method,
        )
    axes[1].set(
        xlabel="模型 − WorldView / 覆盖百分点",
        ylabel="有效样本数（对数）",
        title="覆盖误差分布",
        yscale="log",
    )
    axes[1].legend(fontsize=7)
    save_figure(fig, path, scene["id"] + "_residual_diagnostics")


def run_scene(path: Path, config: dict, scene: dict, rw, rs) -> dict:
    """对一个预定场景均匀抽样，并运行四种预定方法。"""
    cfg = config["worldview"]
    folder = ROOT / "data/cache/public_validation" / scene["id"]
    source = load_json(folder / "source_manifest.json")
    stac = load_json(folder / "stac_item_unsigned.json")
    actual_sza = 90 - stac["properties"]["view:sun_elevation"]
    if abs(actual_sza - scene["solar_zenith_deg"]) > 1e-7:
        raise ValueError("Configured solar angle differs from STAC acquisition metadata")
    arrays = []
    inputs = []
    for filename in [f"SR_B{band}.tif" for band in range(2, 8)] + [
        "QA_PIXEL.tif",
        "QA_RADSAT.tif",
        "SR_QA_AEROSOL.tif",
    ]:
        file = folder / filename
        with rasterio.open(file) as dataset:
            if not arrays:
                shape, transform, crs = dataset.shape, dataset.transform, dataset.crs
            if dataset.shape != shape or dataset.transform != transform or dataset.crs != crs:
                raise ValueError("Landsat input bands do not share a grid")
            arrays.append(dataset.read(1))
        actual_hash = file_hash(file)
        declared = next(
            row["sha256"] for row in source["files"] if Path(row["path"]).name == filename
        )
        if actual_hash != declared:
            raise ValueError(f"Input hash mismatch: {filename}")
        inputs.append({"path": str(file.relative_to(ROOT)), "sha256": actual_hash})
    raw = np.stack(arrays[:6], axis=-1)
    qa, saturation, aerosol = arrays[6:]
    reference_file = ROOT / scene["reference"]
    if file_hash(reference_file) != source["reference_sha256"]:
        raise ValueError("Independent reference hash differs from acquisition manifest")
    reference, coverage, alignment = align_reference(reference_file, shape, transform, crs)
    input_valid = landsat_quality(raw, qa, saturation, cfg["qa_pixel_reject_bits"])
    good = input_valid & (coverage >= cfg["minimum_reference_coverage"]) & np.isfinite(reference)
    indices = np.flatnonzero(good)
    if len(indices) == 0:
        raise ValueError("No common valid independent-reference pixels")
    sample_seed = config["seed"] + int(scene["landsat_date"].replace("-", ""))
    selected = np.sort(
        np.random.default_rng(sample_seed).choice(
            indices, min(cfg["sample_pixels_per_scene"], len(indices)), replace=False
        )
    )
    rows, cols = np.unravel_index(selected, shape)
    observed = (
        raw.reshape(-1, 6)[selected].astype(float) * cfg["surface_reflectance_scale"]
        + cfg["surface_reflectance_offset"]
    )
    truth = reference.ravel()[selected]
    block_size = cfg["spatial_block_size_pixels"]
    block = (rows // block_size) * (int(np.ceil(shape[1] / block_size))) + cols // block_size
    table = pd.DataFrame(
        {
            "row": rows,
            "column": cols,
            "block_id": block,
            "reference_fsca": truth,
            "reference_coverage": coverage.ravel()[selected],
            "stratum": strata(truth, config),
            "qa_pixel": qa.ravel()[selected],
            "qa_radsat": saturation.ravel()[selected],
            "sr_qa_aerosol": aerosol.ravel()[selected],
            "aerosol_valid_retrieval": (aerosol.ravel()[selected] >> 1) & 1,
            "aerosol_interpolated": (aerosol.ravel()[selected] >> 5) & 1,
            "aerosol_level": (aerosol.ravel()[selected] >> 6) & 3,
            "qa_water_bit": (qa.ravel()[selected] >> 7) & 1,
        }
    )
    for index, band in enumerate(range(2, 8)):
        table[f"observed_B{band}"] = observed[:, index]
    radii, snow, background, record = endmembers(config, actual_sza, rw, rs)
    fits = {}
    metrics = []
    residual_rows = []
    aerosol_rows = []
    fixed = int(np.flatnonzero(radii == cfg["fixed_radius_um"])[0])
    for band_name, band_ids in cfg["band_sets"].items():
        band_indices = np.array(band_ids) - 2
        for model_name in ("fixed200", "multi"):
            name = model_name + "_" + band_name
            ss = snow[fixed : fixed + 1] if model_name == "fixed200" else snow
            rr = radii[fixed : fixed + 1] if model_name == "fixed200" else radii
            fit = unmix_pixels(
                observed[:, band_indices],
                ss[:, band_indices],
                rr,
                background[band_indices],
                config,
                solver="fcls",
                progress=scene["id"] + "/" + name,
            )
            chosen = fit["radius_index"]
            modeled = (
                fit["fractions"][:, :1] * ss[chosen][:, band_indices]
                + fit["fractions"][:, 1:] @ background[band_indices].T
            )
            fit["residual"] = observed[:, band_indices] - modeled
            np.testing.assert_allclose(
                np.sqrt(np.mean(fit["residual"] ** 2, axis=1)), fit["rms"], rtol=1e-6, atol=1e-8
            )
            fits[name] = fit
            estimate = fit["normalized"][:, 0]
            table[name + "_snow"] = estimate
            table[name + "_raw_snow"] = fit["fractions"][:, 0]
            table[name + "_shade"] = fit["shade"]
            table[name + "_radius_um"] = fit["radius_um"]
            table[name + "_rms"] = fit["rms"]
            table[name + "_qa"] = fit["qa"]
            for label in ("all", *STRATA):
                keep = (
                    np.ones(len(truth), bool)
                    if label == "all"
                    else table.stratum.to_numpy() == label
                )
                summary = {
                    "scene": scene["id"],
                    "day_offset": scene["day_offset"],
                    "method": name,
                    "bands": band_ids,
                    "stratum": label,
                    "sample_count": int(keep.sum()),
                    "cover": paired_metrics(truth[keep], estimate[keep]),
                    "raw_snow_cover_diagnostic": paired_metrics(
                        truth[keep], fit["fractions"][keep, 0]
                    ),
                    "normalization_invalid_count": int((keep & ~np.isfinite(estimate)).sum()),
                    "spectral_rms_mean": float(np.mean(fit["rms"][keep])) if keep.any() else None,
                    "grain_reported_count": int((keep & np.isfinite(fit["radius_um"])).sum()),
                    "grain_boundary_count": int((keep & ((fit["qa"] & 16) != 0)).sum()),
                    "block_bootstrap": block_bootstrap(
                        truth[keep],
                        estimate[keep],
                        block[keep],
                        sample_seed,
                        cfg["bootstrap_repeats"],
                    ),
                }
                metrics.append(summary)
                if keep.any():
                    for b, mean, std in zip(
                        band_ids,
                        fit["residual"][keep].mean(axis=0),
                        fit["residual"][keep].std(axis=0),
                    ):
                        residual_rows.append(
                            {
                                "method": name,
                                "stratum": label,
                                "band": b,
                                "mean_E_c": mean,
                                "std_E_c": std,
                                "n": int(keep.sum()),
                            }
                        )
            for (level, valid_aerosol, interpolated), group in table.groupby(
                ["aerosol_level", "aerosol_valid_retrieval", "aerosol_interpolated"]
            ):
                keep = group.index.to_numpy()
                aerosol_rows.append(
                    {
                        "method": name,
                        "aerosol_level": int(level),
                        "valid_retrieval": int(valid_aerosol),
                        "interpolated": int(interpolated),
                        **paired_metrics(truth[keep], estimate[keep]),
                    }
                )
            grid = np.full((6, *shape), np.nan, np.float32)
            grid[0].ravel()[selected] = estimate
            grid[1].ravel()[selected] = fit["radius_um"]
            grid[2].ravel()[selected] = fit["rms"]
            grid[3].ravel()[selected] = estimate - truth
            grid[4].ravel()[selected] = fit["fractions"][:, 0]
            grid[5].ravel()[selected] = fit["shade"]
            write_raster(
                path / "rasters" / (scene["id"] + "_" + name + "_sampled.tif"),
                grid,
                transform,
                crs,
                [
                    "sampled_shade_normalized_snow",
                    "sampled_optical_radius_um_no_grain_truth",
                    "sampled_spectral_rms",
                    "sampled_cover_error_model_minus_worldview",
                    "sampled_raw_snow_spectral_coefficient",
                    "sampled_zero_shade_coefficient",
                ],
            )
    common_methods = np.logical_and.reduce(
        [np.isfinite(fit["normalized"][:, 0]) for fit in fits.values()]
    )
    table["common_four_methods_normalization_valid"] = common_methods
    for item in metrics:
        label = item["stratum"]
        keep = common_methods.copy()
        if label != "all":
            keep &= table.stratum.to_numpy() == label
        estimate = fits[item["method"]]["normalized"][:, 0]
        item["common_methods_cover"] = paired_metrics(truth[keep], estimate[keep])
        item["common_methods_raw_snow_diagnostic"] = paired_metrics(
            truth[keep], fits[item["method"]]["fractions"][keep, 0]
        )
        item["common_methods_block_bootstrap"] = block_bootstrap(
            truth[keep], estimate[keep], block[keep], sample_seed, cfg["bootstrap_repeats"]
        )
    deltas = {}
    for band_name in cfg["band_sets"]:
        first = fits["fixed200_" + band_name]["normalized"][:, 0]
        second = fits["multi_" + band_name]["normalized"][:, 0]
        deltas[band_name] = paired_delta(
            truth[common_methods],
            first[common_methods],
            second[common_methods],
            block[common_methods],
            sample_seed,
            cfg["bootstrap_repeats"],
        )
    table.to_csv(path / "tables" / (scene["id"] + "_sample_pixels.csv"), index=False)
    pd.DataFrame(residual_rows).to_csv(
        path / "tables" / (scene["id"] + "_residual_summary.csv"), index=False
    )
    pd.DataFrame(aerosol_rows).to_csv(
        path / "tables" / (scene["id"] + "_aerosol_groups.csv"), index=False
    )
    write_raster(
        path / "rasters" / (scene["id"] + "_independent_reference.tif"),
        reference,
        transform,
        crs,
        ["worldview_viewable_fsca_0_1"],
    )
    np.savez_compressed(
        path / "arrays" / (scene["id"] + "_sample_solutions.npz"),
        selected_flat_indices=selected,
        reference=truth,
        observed=observed,
        **{name + "_" + key: value for name, fit in fits.items() for key, value in fit.items()},
    )
    save_sample_maps(path, scene, reference, rows, cols, fits, transform, crs, table)
    scatter_and_residuals(path, scene, table, fits)
    full_labels = strata(reference[good], config)
    center_x = transform.c + shape[1] * transform.a / 2
    center_y = transform.f + shape[0] * transform.e / 2
    center_lon, center_lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(
        center_x, center_y
    )
    result = {
        "id": scene["id"],
        "reference_date": scene["reference_date"],
        "landsat_date": scene["landsat_date"],
        "day_offset": scene["day_offset"],
        "landsat_acquisition_utc": source["acquisition_utc"],
        "reference_definition": "independent WorldView viewable fSCA",
        "center_lon_lat": [center_lon, center_lat],
        "grid_crs": str(crs),
        "shape": list(shape),
        "input_valid_pixels": int(input_valid.sum()),
        "common_valid_pixels": int(good.sum()),
        "sampled_pixels": len(selected),
        "sample_seed": sample_seed,
        "common_four_methods_normalized_pixels": int(common_methods.sum()),
        "common_four_methods_excluded_pixels": int((~common_methods).sum()),
        "paired_multi_minus_fixed": deltas,
        "common_footprint_strata_counts": {
            label: int((full_labels == label).sum()) for label in STRATA
        },
        "sample_strata_counts": {label: int((table.stratum == label).sum()) for label in STRATA},
        "reference_alignment": alignment,
        "input_reflectance_beyond_1_count": int((observed > 1).sum()),
        "input_negative_reflectance_count": int((observed < 0).sum()),
        "input_reflectance_was_clipped": False,
        "input_sources": inputs,
        "endmembers": record,
        "methods": metrics,
        "no_training_or_validation_label_tuning": True,
        "radius_absolute_validation_available": False,
    }
    write_json(path / (scene["id"] + "_scene_summary.json"), result)
    print(
        "RESULT",
        scene["id"],
        [
            (m["method"], m["cover"]["n"], m["cover"]["rmse"])
            for m in metrics
            if m["stratum"] == "all"
        ],
        flush=True,
    )
    return result


def focused_checks() -> dict:
    """检查雪位保留、剔除位、DN 缩放和跨 CRS 常量守恒。"""
    raw = np.full((7, 6), 60000, np.uint16)
    qa = np.array([1 << 5] + [1 << b for b in range(5)] + [0], np.uint16)
    saturation = np.zeros(7, np.uint16)
    saturation[-1] = 1 << 11
    np.testing.assert_array_equal(
        landsat_quality(raw, qa, saturation, [0, 1, 2, 3, 4]),
        [True, False, False, False, False, False, False],
    )
    assert raw[0, 0] * 0.0000275 - 0.2 > 1  # 按比例和偏移缩放，保留大于 1 的反射率。
    source = np.full((40, 40), 420, np.int16)
    source[15:20, 15:20] = -32768
    source_transform = Affine(30, 0, 750000, 0, -30, 4390000)
    tx = Transformer.from_crs(32610, 32611, always_xy=True)
    x, y = tx.transform(750000, 4390000)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=40,
            width=40,
            count=1,
            dtype="int16",
            transform=source_transform,
            crs=32610,
        ) as dataset:
            dataset.write(source, 1)
        destination, coverage, notes = align_reference(
            memory.name,
            (45, 45),
            Affine(30, 0, x - 90, 0, -30, y + 90),
            rasterio.crs.CRS.from_epsg(32611),
        )
        assert notes["reprojected"] and not notes["grid_exactly_matched"]
    assert np.isfinite(destination).any()
    np.testing.assert_allclose(destination[np.isfinite(destination)], 0.42, atol=1e-7)
    assert (coverage < 0.95).any() and (coverage >= 0.95).any()
    assert coverage.min() >= 0 and coverage.max() <= 1 + 1e-7
    return {
        "qa_and_scale": "passed",
        "synthetic_cross_crs_constant_preservation_and_support": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/worldview_validation.json")
    args = parser.parse_args()
    config = load_json(args.config)
    checks = focused_checks()
    path = new_run(config, "13_worldview_validation")
    configure_plots()
    plt.rcParams["savefig.dpi"] = 300
    rw, rs, response_notes = load_oli_responses()
    write_json(path / "response_functions.json", response_notes)
    results = [run_scene(path, config, scene, rw, rs) for scene in config["worldview"]["scenes"]]
    flattened = []
    for scene in results:
        for row in scene["methods"]:
            flattened.append(
                {
                    "scene": scene["id"],
                    "reference_date": scene["reference_date"],
                    "day_offset": scene["day_offset"],
                    "method": row["method"],
                    "stratum": row["stratum"],
                    **row["cover"],
                    "spectral_rms_mean": row["spectral_rms_mean"],
                    "grain_boundary_count": row["grain_boundary_count"],
                    "normalization_invalid_count": row["normalization_invalid_count"],
                    **{"common_" + key: val for key, val in row["common_methods_cover"].items()},
                    "raw_snow_rmse_diagnostic": row["raw_snow_cover_diagnostic"]["rmse"],
                }
            )
    pd.DataFrame(flattened).to_csv(path / "tables/stratified_metrics.csv", index=False)
    metrics = {
        "benchmark": "third-sensor transfer to independent WorldView viewable fractional snow cover",
        "not_new_aviris_modis_scene_validation": True,
        "reference_source": REFERENCE_SOURCE,
        "qa_source": QA_SOURCE,
        "scene_date_offsets_not_pooled": True,
        "checks": checks,
        "scenes": results,
    }
    lines = [
        "# WorldView 独立雪覆盖：第三传感器迁移实验",
        "",
        "使用公开 WorldView 可见雪覆盖比例作独立参考，固定雪/实验室松针/花岗岩/零阴影字典，未训练或按标签调参。",
        "2016 同日与 2017 相差 1 天分别评价，构成第三传感器的覆盖迁移检验。",
        "",
        "| 场景 | 方法 | n | bias / pp | MAE / pp | RMSE / pp |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scene in results:
        for method in scene["methods"]:
            if method["stratum"] == "all":
                m = method["common_methods_cover"]
                lines.append(
                    f"| {scene['landsat_date']} (+{scene['day_offset']} d) | {method['method']} | {m['n']} | {m['bias'] * 100:.3f} | {m['mae'] * 100:.3f} | {m['rmse'] * 100:.3f} |"
                )
    lines.extend(
        [
            "",
            "上表使用四种方法共同归一化有效子集；预定12000样本按输入及归一化有效性筛选，失效值保留为空。各方法条件有效指标、筛除率、raw雪系数诊断另存，结果同时报告覆盖误差和失效率。",
            "各雪覆盖分层与气溶胶分组见 tables；空间区块为 50×50 个 Landsat 像元（1.5 km），1000 次区块 bootstrap。",
            "覆盖精度区间仅条件于当前场景、固定字典和 WorldView 参考；不包含参考分类、BRDF、地形或时差的不确定性。",
            "",
            "QA_PIXEL 剔除 fill/dilated cloud/cirrus/cloud/shadow（bits 0–4），保留 snow bit 5；所有方法共用 QA_RADSAT=0 掩膜。",
            "SR_QA_AEROSOL 按气溶胶等级、有效反演和插值标记分组；各组均报告误差与样本数。",
            "反射率使用 DN×0.0000275−0.2；负值与超过 1 的值保留。参考 Int16 手动剔除 −32768，再除 1000；源 nodata 元数据为空。",
            "实际两场源影像与参考分别在 UTM10N/UTM11N 完全匹配；未伪造跨 CRS 处理。通用平均重投影分支保留有效面积并有跨 CRS 常量检查。",
            "雪谱为官方 SNICAR 精确场景太阳天顶角、球形厚雪、Warren1984；DHR近似传感器反射率，未做完整方向性/地形修正。",
            "",
            f"[独立参考]({REFERENCE_SOURCE})；[官方 QA]({QA_SOURCE})；[官方 OLI RSR]({RSR_SOURCE})。",
            "",
            "## 图注",
            "",
        ]
    )
    for scene in results:
        for suffix, description in [
            (
                "sample_maps",
                "左图是完整独立参考，后三幅仅预定随机样本；灰色为无数据/未抽样。统一0–1雪比例色标；差值以0为中心。投影、坐标和5 km比例尺标在图上。",
            ),
            (
                "stratified_scatter",
                "按独立参考定义无雪<0.1、混合0.1–0.9、近纯雪≥0.9；黑线为1∶1。图例给出有效数/预定层内数，低分母NaN不画成零。四种模型在相同预定样本上比较，未挑选低RMS点。",
            ),
            (
                "residual_diagnostics",
                "左侧E_c由原始系数重建，先按参考覆盖分层取波段均值；右侧为独立覆盖误差分布。光谱拟合良好不自动表示覆盖准确。",
            ),
        ]:
            file = path / "figures" / (scene["id"] + "_" + suffix + ".png")
            lines.extend(
                [f"![{scene['landsat_date']} {suffix}]({file.as_posix()})", "", description, ""]
            )
    (path / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finish_run(path, metrics)
    print("COMPLETE", path.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
