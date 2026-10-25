"""处理 AVIRIS 影像，开展分辨率控制实验与同日 MODIS 对比。"""

from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from scipy.io import loadmat
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
from .artifacts import (
    ROOT,
    new_run,
    finish_run,
    configure_plots,
    write_json,
    save_figure,
    paired_metrics,
)
from .linear import select_snow
from .cache_contract import validate_forward_settings
from .spectra import (
    gaussian_band_matrix,
    read_usgs_ascii,
    read_ecostress,
    modis_responses,
    response_matrix,
)
from .geo import MODIS_CRS, modis_roi_grid, area_weights, aggregate
from .modis import quality_mask


def prepare_roi(config, metadata):
    """提取研究区并面积平均到目标分辨率，返回反射率、有效像元与波段掩膜。"""
    cfg = config["image"]
    with rasterio.open(ROOT / metadata["binary_path"]) as source:
        x, y = Transformer.from_crs(4326, source.crs, always_xy=True).transform(
            *cfg["roi_center_lon_lat"]
        )
        size = cfg["roi_width_m"]
        resolution = cfg["target_resolution_m"]
        left = math.floor((x - size / 2) / resolution) * resolution
        top = math.ceil((y + size / 2) / resolution) * resolution
        shape = (math.ceil(size / resolution), math.ceil(size / resolution))
        transform = from_origin(left, top, resolution, resolution)
        bounds = (left, top - shape[0] * resolution, left + shape[1] * resolution, top)
        window = from_bounds(*bounds, source.transform).round_offsets().round_lengths()
        raw = source.read(window=window, boundless=True, fill_value=0, out_dtype="float32")
        source_transform = source.window_transform(window)
        wavelengths = np.array(metadata["wavelength_um"])
        usable = (wavelengths >= 0.42) & (wavelengths <= 2.45)
        usable &= ~((wavelengths >= 1.35) & (wavelengths <= 1.45))
        usable &= ~((wavelengths >= 1.8) & (wavelengths <= 1.98))
        bbl = metadata["header_metadata"].get("bbl")
        if bbl is not None:
            usable &= np.array(bbl, dtype=int) == 1
        invalid = (raw == 0).all(0) | (raw[usable] > 16000).any(0) | (raw[usable] < 0).any(0)
        reflectance = raw / metadata["scale_factor"]
        reflectance[:, invalid] = -9999
        cube = np.full((source.count, *shape), np.nan, dtype=np.float32)
        reproject(
            reflectance,
            cube,
            src_transform=source_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=source.crs,
            src_nodata=-9999,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=2,
        )
        native_coverage = np.zeros(shape, np.float32)
        reproject(
            (~invalid).astype("float32"),
            native_coverage,
            src_transform=source_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=source.crs,
            dst_nodata=0,
            resampling=Resampling.average,
            num_threads=2,
        )
        valid = np.isfinite(cube[usable]).all(0) & np.any(cube[usable] > 0, axis=0)
        valid &= native_coverage >= 0.95
        return np.moveaxis(cube, 0, -1), valid, usable, transform, shape, source.crs, bounds


def sensor_library(wavelengths, fwhm, angle, config=None):
    """按配置选择雪粒半径与入射角，将雪和背景光谱卷积到同一传感器网格。"""
    data = loadmat(ROOT / "data/cache/forward_models.mat")
    if angle == 25:
        data = loadmat(ROOT / "data/cache/forward_models_25deg.mat")
    validate_forward_settings(data, config or {})
    angles = data["angles_deg"].ravel()
    ai = int(np.argmin(abs(angles - angle)))
    if abs(angles[ai] - angle) > 0.5:
        raise ValueError("Requested forward angle unavailable")
    radii = data["radii_um"].ravel()
    cfg = (config or {}).get(
        "image", {"radius_min_um": 30, "radius_max_um": 1100, "radius_step_um": 10}
    )
    if cfg["radius_step_um"] <= 0 or cfg["radius_min_um"] > cfg["radius_max_um"]:
        raise ValueError("Invalid radius range or step")
    requested = np.arange(
        cfg["radius_min_um"],
        cfg["radius_max_um"] + cfg["radius_step_um"] / 2,
        cfg["radius_step_um"],
    )
    if not np.isin(requested, radii).all():
        raise ValueError("Requested image radius grid is absent from frozen optical cache")
    keep = np.isin(radii, requested)
    ice_option = (config or {}).get("forward", {}).get("ice_ri", 1)
    optical_indices = np.flatnonzero(data["ice_options"].ravel() == ice_option)
    if len(optical_indices) != 1:
        raise ValueError("Requested ice option absent from cache")
    weights = gaussian_band_matrix(data["wavelength_um"].ravel(), wavelengths, fwhm)
    snow = data["snicar_albedo"][keep, :, ai, int(optical_indices[0])] @ weights
    records = []
    background = []
    definitions = [
        ("vegetation", "lodgepole_pine.asc", read_usgs_ascii),
        ("rock", "granite_h2.txt", read_ecostress),
        ("soil", "cryumbrept_87p3855.txt", read_ecostress),
        ("quartz_proxy", "quartz_hs32.5854.asc", read_usgs_ascii),
    ]
    for name, filename, reader in definitions:
        w, r, record = reader(ROOT / "data/raw/spectra" / filename)
        convolution = gaussian_band_matrix(w, wavelengths, fwhm)
        background.append(r @ convolution)
        records.append({"class": name, "file": filename, "metadata": record})
    return radii[keep], snow, np.column_stack(background), records


def unmix_pixels(y, snow, radii, background, config, solver="original", progress=""):
    """分块进行候选粒径解混，汇总原始系数、归一化比例、粒径、RMS 和 QA。"""
    n = len(y)
    k = background.shape[1] + 1
    output = {
        "fractions": np.full((n, k), np.nan, np.float32),
        "normalized": np.full((n, k), np.nan, np.float32),
        "shade": np.full(n, np.nan, np.float32),
        "radius_um": np.full(n, np.nan, np.float32),
        "radius_index": np.full(n, -1, np.int16),
        "rms": np.full(n, np.nan, np.float32),
        "condition": np.full(n, np.nan, np.float32),
        "qa": np.zeros(n, np.uint16),
    }
    cfg = config["image"]
    with threadpool_limits(limits=1):
        for start in range(0, n, cfg["block_size"]):
            stop = min(start + cfg["block_size"], n)
            result = select_snow(
                y[start:stop],
                snow,
                radii,
                background,
                solver=solver,
                minimum_denominator=cfg["minimum_denominator"],
                minimum_snow=cfg["minimum_snow_fraction"],
                condition_warning=cfg["condition_warning"],
            )
            for name in output:
                output[name][start:stop] = getattr(result, name)
            if progress and start % (cfg["block_size"] * 8) == 0:
                print(f"{progress}: {stop}/{n} pixels", flush=True)
    return output


def write_raster(path, array, transform, crs, descriptions):
    """写出带坐标参考、无效值及波段说明的压缩 GeoTIFF。"""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    if array.dtype == np.uint16:
        nodata = 65535
    else:
        array = array.astype(np.float32)
        nodata = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        transform=transform,
        crs=crs,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    ) as dataset:
        dataset.write(array)
        for i, name in enumerate(descriptions, 1):
            dataset.set_band_description(i, name)


def save_solution(path, label, result, shape, transform, crs, classes):
    """保存解混数组，并分别导出原始系数、归一化比例、粒径诊断和 QA 栅格。"""
    np.savez_compressed(path / "arrays" / f"{label}.npz", **result)
    raw = np.column_stack((result["fractions"], result["shade"])).reshape(*shape, -1)
    write_raster(
        path / "rasters" / f"{label}_spectral_fractions.tif",
        np.moveaxis(raw, -1, 0),
        transform,
        crs,
        classes + ["shade"],
    )
    normalized = result["normalized"].reshape(*shape, -1)
    write_raster(
        path / "rasters" / f"{label}_normalized_fractions.tif",
        np.moveaxis(normalized, -1, 0),
        transform,
        crs,
        classes,
    )
    extra = np.column_stack((result["radius_um"], result["rms"], result["condition"])).reshape(
        *shape, 3
    )
    write_raster(
        path / "rasters" / f"{label}_grain_rms_condition.tif",
        np.moveaxis(extra, -1, 0),
        transform,
        crs,
        ["radius_um", "spectral_rms_reflectance", "condition_number"],
    )
    write_raster(
        path / "rasters" / f"{label}_qa.tif",
        result["qa"].reshape(shape),
        transform,
        crs,
        ["qa_bitmask"],
    )


def map_panel(path, name, arrays, titles, shape, limits, cmap="viridis"):
    """按共同色标范围绘制一组栅格结果。"""
    fig, axes = plt.subplots(1, len(arrays), figsize=(4.2 * len(arrays), 4.7), squeeze=False)
    for ax, array, title in zip(axes.ravel(), arrays, titles):
        im = ax.imshow(np.asarray(array).reshape(shape), vmin=limits[0], vmax=limits[1], cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(name.replace("_", " "), fontsize=12)
    save_figure(fig, path, name)


def summarize_result(result):
    """汇总有效像元、光谱 RMS、比例越界及粒径质量标记。"""
    valid = np.isfinite(result["rms"])
    snow = result["normalized"][:, 0]
    return {
        "valid_pixels": int(valid.sum()),
        "spectral_rms_mean": float(np.mean(result["rms"][valid])) if valid.any() else None,
        "snow_fraction_mean": float(np.nanmean(snow)) if np.isfinite(snow).any() else None,
        "snow_underflow_rate": float(np.mean(snow[valid] < -1e-8)) if valid.any() else None,
        "snow_overflow_rate": float(np.mean(snow[valid] > 1 + 1e-8)) if valid.any() else None,
        "ill_conditioned_rate": float(np.mean((result["qa"][valid] & 32) != 0))
        if valid.any()
        else None,
        "grain_valid_pixels": int(np.isfinite(result["radius_um"]).sum()),
        "radius_boundary_pixels": int(((result["qa"] & 16) != 0).sum()),
    }


def bootstrap_blocks(reference, estimate, mask, shape, seed):
    """以 3 × 3 像元块重采样配对差值，计算 bias 和 RMSE 的 95% 区间。

    每次抽取与原样本相同数量的块，再合并所抽块内的有效像元计算统计量。
    """
    ref = np.asarray(reference).reshape(shape)
    est = np.asarray(estimate).reshape(shape)
    mask = mask.reshape(shape)
    blocks = []
    for r in range(0, shape[0], 3):
        for c in range(0, shape[1], 3):
            m = mask[r : r + 3, c : c + 3]
            if m.any():
                blocks.append((est[r : r + 3, c : c + 3] - ref[r : r + 3, c : c + 3])[m])
    if len(blocks) < 5:
        return {"blocks": len(blocks), "bias_interval_95": None, "rmse_interval_95": None}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1000):
        values = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        samples.append([values.mean(), np.sqrt(np.mean(values * values))])
    quantile = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "blocks": len(blocks),
        "bias_interval_95": quantile[:, 0].tolist(),
        "rmse_interval_95": quantile[:, 1].tolist(),
    }


def run_image(config):
    """执行 AVIRIS 解混、光谱与空间分辨率控制实验，以及真实 MODIS 比对。"""
    path = new_run(config, "05_07_image")
    configure_plots()
    metadata = json.loads((ROOT / "data/cache/aviris_metadata.json").read_text(encoding="utf-8"))
    cube, valid, band_mask, transform, shape, crs, bounds = prepare_roi(config, metadata)
    w = np.array(metadata["wavelength_um"])
    fwhm = np.array(metadata["fwhm_um"])
    write_json(
        path / "band_selection.json",
        {
            "valid_indices_zero_based": np.flatnonzero(band_mask),
            "band_count": int(band_mask.sum()),
            "modern_mask": "0.42-2.45 um minus [1.35,1.45] and [1.80,1.98], plus supplied bbl",
            "historical_175_mask_recovered": False,
        },
    )
    print(f"AVIRIS ROI {shape}, valid={valid.sum()}, bands={band_mask.sum()}", flush=True)
    radii, snow, library, records = sensor_library(w[band_mask], fwhm[band_mask], 22, config)
    write_json(
        path / "endmembers.json",
        {
            "records": records,
            "snow_solar_zenith_deg": 22,
            "gravel_status": "quartz_proxy: mineral powder; a measured gravel endmember remains required.",
        },
    )
    data = cube[..., band_mask].reshape(-1, int(band_mask.sum())).astype(float)
    data[~valid.ravel()] = np.nan
    fixed = np.flatnonzero(radii == 200)[0]
    classes = ["snow", "vegetation", "rock", "soil"]
    background = library[:, :3]
    fixed_result = unmix_pixels(
        data, snow[fixed : fixed + 1], [200], background, config, progress="AVIRIS fixed"
    )
    original = unmix_pixels(data, snow, radii, background, config, progress="AVIRIS original")
    constrained = unmix_pixels(
        data, snow, radii, background, config, solver="fcls", progress="AVIRIS FCLS"
    )
    for label, result in [
        ("aviris_fixed200_original", fixed_result),
        ("aviris_multi_original", original),
        ("aviris_multi_fcls", constrained),
    ]:
        save_solution(path, label, result, shape, transform, crs, classes)
    extended_original = unmix_pixels(
        data, snow, radii, library, config, progress="AVIRIS expanded proxy"
    )
    save_solution(
        path,
        "aviris_expanded_proxy_original",
        extended_original,
        shape,
        transform,
        crs,
        ["snow", "vegetation", "rock", "soil", "quartz_proxy"],
    )
    rgb = np.stack([cube[..., np.argmin(abs(w - wave))] for wave in (0.65, 0.55, 0.47)], axis=-1)
    rgb = np.nan_to_num(rgb)
    rgb = np.clip(rgb / 0.65, 0, 1)
    rgb[~valid] = 0.92
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(f"AVIRIS L2：2013-05-03，{config['image'].get('roi_label', 'Sierra')}，30 m")
    ax.axis("off")
    save_figure(fig, path, "aviris_true_color")
    map_panel(
        path,
        "aviris_snow_fraction",
        [
            fixed_result["normalized"][:, 0],
            original["normalized"][:, 0],
            constrained["normalized"][:, 0],
        ],
        ["固定200 μm / 原文约束", "MIN RMS / 原文约束", "MIN RMS / FCLS"],
        shape,
        (0, 1),
        "Blues",
    )
    map_panel(
        path,
        "aviris_spectral_rms_pp",
        [fixed_result["rms"] * 100, original["rms"] * 100, constrained["rms"] * 100],
        ["固定200 μm", "MIN RMS / 原文约束", "MIN RMS / FCLS"],
        shape,
        (0, 5),
        "magma",
    )
    map_panel(
        path,
        "aviris_grain_radius_um",
        [original["radius_um"], constrained["radius_um"]],
        ["原文约束", "FCLS"],
        shape,
        (30, 1100),
        "viridis",
    )
    map_panel(
        path,
        "aviris_endmember_fractions",
        [constrained["normalized"][:, i] for i in range(1, 4)],
        ["植被", "岩石", "土壤"],
        shape,
        (0, 1),
    )
    map_panel(
        path,
        "aviris_shade_qa",
        [constrained["shade"], ((original["qa"] & 64) != 0).astype(float)],
        ["FCLS 阴影比例", "原文约束比例越界标记"],
        shape,
        (0, 1),
    )
    # 响应积分使用有效 AVIRIS 中心波长，强水吸收波段按掩膜剔除。
    rw, rs = modis_responses(ROOT / "data/raw/Terra_RSR_in-band.xlsx")
    unique, indices = np.unique(w, return_index=True)
    response_unique, rsr_coverage = response_matrix(unique, rw, rs, band_mask[indices])
    response = np.zeros((len(w), 7))
    response[indices] = response_unique
    assert np.isfinite(response).all(), f"MODIS RSR lacks valid AVIRIS support: {rsr_coverage}"
    simulated = np.nan_to_num(cube.reshape(-1, len(w)), nan=0) @ response
    simulated[~valid.ravel()] = np.nan
    # 端元与观测谱使用同一离散 AVIRIS → MODIS 卷积算子。
    kept_response = response[band_mask]
    snow_modis = snow @ kept_response
    library_modis = library.T @ kept_response
    background_modis = library_modis[:2].T  # 植被与代表裸地的岩石光谱
    fine_broad = unmix_pixels(
        simulated,
        snow_modis,
        radii,
        background_modis,
        config,
        solver="fcls",
        progress="spectral-only MODIS",
    )
    save_solution(
        path,
        "spectral_only_modis_fcls",
        fine_broad,
        shape,
        transform,
        crs,
        ["snow", "vegetation", "bare"],
    )
    coarse_transform, coarse_shape, tile_window = modis_roi_grid(bounds, crs)
    weights = area_weights(transform, shape, crs, coarse_transform, coarse_shape, MODIS_CRS)
    save_npz(path / "arrays/area_weights.npz", weights)
    coarse_high, coverage = aggregate(weights, data, valid)
    coarse_broad, coverage_broad = aggregate(weights, simulated, valid)
    minimum = config["image"]["minimum_coarse_coverage"]
    full = coverage >= minimum
    coarse_high[~full] = np.nan
    coarse_broad[~full] = np.nan
    spatial_only = unmix_pixels(coarse_high, snow, radii, background, config, solver="fcls")
    combined = unmix_pixels(
        coarse_broad, snow_modis, radii, background_modis, config, solver="fcls"
    )
    save_solution(
        path,
        "spatial_only_aviris_fcls",
        spatial_only,
        coarse_shape,
        coarse_transform,
        MODIS_CRS,
        classes,
    )
    save_solution(
        path,
        "combined_simulated_modis_fcls",
        combined,
        coarse_shape,
        coarse_transform,
        MODIS_CRS,
        ["snow", "vegetation", "bare"],
    )
    reference, reference_coverage = aggregate(weights, constrained["normalized"][:, 0])
    reference = reference[:, 0]
    broad_reference, broad_ref_coverage = aggregate(weights, fine_broad["normalized"][:, 0])
    reference_full = full & (reference_coverage >= minimum)
    broad_reference_full = full & (broad_ref_coverage >= minimum)
    grain_valid = np.isfinite(constrained["radius_um"]) & np.isfinite(
        constrained["normalized"][:, 0]
    )
    ref_radius_numerator = weights @ np.where(
        grain_valid, constrained["radius_um"] * constrained["normalized"][:, 0], 0
    )
    ref_radius_denominator = weights @ np.where(grain_valid, constrained["normalized"][:, 0], 0)
    # 有效粒径像元对应的雪面积占总雪面积的比例。
    total_snow_area = weights @ np.nan_to_num(constrained["normalized"][:, 0], nan=0)
    grain_snow_support = np.divide(
        ref_radius_denominator,
        total_snow_area,
        out=np.full(len(reference), np.nan),
        where=total_snow_area > 0,
    )
    reference_radius = np.full(len(reference), np.nan)
    np.divide(
        ref_radius_numerator,
        ref_radius_denominator,
        out=reference_radius,
        where=reference_full
        & (ref_radius_denominator > 0)
        & (grain_snow_support >= minimum)
        & (reference > config["image"]["minimum_snow_fraction"]),
    )
    map_panel(
        path,
        "controlled_degradation_snow",
        [reference, spatial_only["normalized"][:, 0], combined["normalized"][:, 0]],
        ["先解混再聚合：AVIRIS参照", "仅空间聚合后解混", "光谱＋空间退化后解混"],
        coarse_shape,
        (0, 1),
        "Blues",
    )
    # 扩展端元模型：雪、植被、岩石、土壤与 quartz_proxy。
    expanded = unmix_pixels(coarse_broad, snow_modis, radii, library_modis.T, config, solver="fcls")
    save_solution(
        path,
        "expanded_simulated_modis_proxy_fcls",
        expanded,
        coarse_shape,
        coarse_transform,
        MODIS_CRS,
        ["snow", "vegetation", "rock", "soil", "quartz_proxy"],
    )
    conditions = []
    for sensor, s, back in [("AVIRIS", snow, library), ("MODIS7", snow_modis, library_modis.T)]:
        for index in (0, int(np.flatnonzero(radii == 200)[0]), len(radii) - 1):
            a = np.column_stack((s[index], back))
            conditions.append(
                {
                    "sensor": sensor,
                    "radius_um": radii[index],
                    "rank": np.linalg.matrix_rank(a),
                    "physical_components": a.shape[1],
                    "condition": np.linalg.cond(a),
                }
            )
    pd.DataFrame(conditions).to_csv(path / "tables/expanded_endmember_condition.csv", index=False)
    # 扰动七波段光谱，检验各端元比例对噪声的敏感性。
    eligible = np.flatnonzero(full & np.isfinite(combined["rms"]))
    stable_points = np.random.default_rng(config["seed"]).choice(
        eligible, min(100, len(eligible)), replace=False
    )
    repeated = []
    rng = np.random.default_rng(config["seed"])
    for repeat in range(20):
        perturbed = coarse_broad[stable_points] + rng.normal(0, 0.005, (len(stable_points), 7))
        fit = unmix_pixels(perturbed, snow_modis, radii, library_modis.T, config, solver="fcls")
        repeated.append(fit["normalized"])
    repeated = np.asarray(repeated)
    variability = np.nanstd(repeated, axis=0)
    variability_mean = np.nanmean(variability, axis=0) if variability.size else np.full(5, np.nan)
    # 将 MOD09GA 观测结果与面积聚合后的 AVIRIS 估计值配对。
    modis = loadmat(ROOT / "data/cache/modis_fields.mat")
    r0, r1, c0, c1 = tile_window
    raw = modis["band_raw"][r0:r1, c0:c1]
    qc = modis["qc"][r0:r1, c0:c1]
    state = np.repeat(np.repeat(modis["state"], 2, 0), 2, 1)[r0:r1, c0:c1]
    attributes = loadmat(ROOT / "data/cache/modis_attributes.mat")
    qc_fill = int(attributes["qc_fill"].item())
    scale = float(attributes["scale"].item())
    assert scale == 0.0001 and int(attributes["refl_fill"].item()) == -28672
    actual_qa = quality_mask(raw, qc, state, qc_fill=qc_fill).ravel()
    actual = raw.reshape(-1, 7).astype(float) * scale
    common = actual_qa & full
    actual[~common] = np.nan
    radii_real, snow_av25, library_av25, _ = sensor_library(
        w[band_mask], fwhm[band_mask], 25, config
    )
    snow25 = snow_av25 @ kept_response
    backgrounds25 = (library_av25[:, :2].T @ kept_response).T
    real = unmix_pixels(actual, snow25, radii_real, backgrounds25, config, solver="fcls")
    save_solution(
        path,
        "real_modis_fcls",
        real,
        coarse_shape,
        coarse_transform,
        MODIS_CRS,
        ["snow", "vegetation", "bare"],
    )
    paired = (
        common
        & (reference_coverage >= minimum)
        & np.isfinite(real["normalized"][:, 0])
        & np.isfinite(reference)
    )
    reference_for_plot = reference.copy()
    reference_for_plot[~paired] = np.nan
    map_panel(
        path,
        "actual_vs_simulated_modis_snow",
        [reference_for_plot, combined["normalized"][:, 0], real["normalized"][:, 0]],
        ["聚合AVIRIS参照（共同有效区）", "模拟MODIS", "真实MODIS / 同日"],
        coarse_shape,
        (0, 1),
        "Blues",
    )
    fig, ax = plt.subplots(figsize=(5.8, 5.5))
    ax.scatter(reference[paired], real["normalized"][paired, 0], s=25, alpha=0.7, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="black", lw=1)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="聚合 AVIRIS 雪覆盖比例",
        ylabel="真实 MODIS 雪覆盖比例",
        title=f"独立传感器一致性对比；n={paired.sum()}",
    )
    save_figure(fig, path, "real_modis_aviris_scatter")
    solar = np.repeat(np.repeat(modis["solar_zenith_raw"], 2, 0), 2, 1)[r0:r1, c0:c1].ravel() * 0.01
    view = np.repeat(np.repeat(modis["sensor_zenith_raw"], 2, 0), 2, 1)[r0:r1, c0:c1].ravel() * 0.01
    coarse_table = pd.DataFrame(
        {
            "tile_row": np.repeat(np.arange(r0, r1), coarse_shape[1]),
            "tile_column": np.tile(np.arange(c0, c1), coarse_shape[0]),
            "aviris_valid_area_fraction": coverage,
            "aviris_reference_valid_area_fraction": reference_coverage,
            "spectral_reference_valid_area_fraction": broad_ref_coverage,
            "grain_valid_snow_area_fraction": grain_snow_support,
            "modis_strict_qa": actual_qa,
            "paired_valid": paired,
            "aviris_mean_snow": reference,
            "spectral_then_spatial_snow": broad_reference[:, 0],
            "spatial_only_snow": spatial_only["normalized"][:, 0],
            "simulated_modis_snow": combined["normalized"][:, 0],
            "real_modis_snow": real["normalized"][:, 0],
            "aviris_snow_weighted_radius_um": reference_radius,
            "simulated_radius_um": combined["radius_um"],
            "real_radius_um": real["radius_um"],
            "solar_zenith_deg": solar,
            "sensor_zenith_deg": view,
            "real_rms": real["rms"],
        }
    )
    coarse_table.to_csv(path / "tables/coarse_pixel_comparison.csv", index=False)
    for band in range(7):
        coarse_table[f"real_band{band + 1}_reflectance"] = raw.reshape(-1, 7)[:, band] * 0.0001
    coarse_table.to_csv(path / "tables/coarse_pixel_spectra.csv", index=False)
    ndvi = (cube[..., np.argmin(abs(w - 0.86))] - cube[..., np.argmin(abs(w - 0.65))]) / (
        cube[..., np.argmin(abs(w - 0.86))] + cube[..., np.argmin(abs(w - 0.65))] + 1e-10
    )
    ndsi = (cube[..., np.argmin(abs(w - 0.55))] - cube[..., np.argmin(abs(w - 1.64))]) / (
        cube[..., np.argmin(abs(w - 0.55))] + cube[..., np.argmin(abs(w - 1.64))] + 1e-10
    )
    metrics = {
        "aviris_roi_shape": list(shape),
        "aviris_valid_pixels": int(valid.sum()),
        "aviris_valid_fraction": float(valid.mean()),
        "usable_aviris_bands": int(band_mask.sum()),
        "roi_center_lon_lat": config["image"]["roi_center_lon_lat"],
        "roi_selection": config["image"].get(
            "roi_selection_reason", "coverage and strict QA before fitting"
        ),
        "roi_label": config["image"].get("roi_label", "Sierra"),
        "modis_actual_attributes": {
            "qc_fill": qc_fill,
            "scale_factor": scale,
            "reflectance_fill": int(attributes["refl_fill"].item()),
            "valid_range": attributes["valid_range"].ravel().tolist(),
        },
        "snow_ndsi_gt0p4_pixels": int((valid & (ndsi > 0.4)).sum()),
        "vegetation_ndvi_gt0p3_pixels": int((valid & (ndvi > 0.3)).sum()),
        "bare_index_candidate_pixels": int((valid & (ndvi < 0.2) & (ndsi < 0.4)).sum()),
        "index_candidates_are_not_ground_truth": True,
        "models": {
            label: summarize_result(result)
            for label, result in [
                ("fixed_original", fixed_result),
                ("multi_original", original),
                ("multi_fcls", constrained),
                ("expanded_proxy_original", extended_original),
            ]
        },
        "coarse_shape": list(coarse_shape),
        "coarse_full_coverage_pixels": int(full.sum()),
        "modis_strict_qa_pixels": int(actual_qa.sum()),
        "matched_pixels": int(paired.sum()),
        "meets_minimum_matched_pixels": bool(
            paired.sum() >= config["image"]["minimum_matched_pixels"]
        ),
        "modis_rsr_coverage": rsr_coverage.tolist(),
        "spatial_only_agreement": paired_metrics(
            reference[reference_full], spatial_only["normalized"][reference_full, 0]
        ),
        "combined_degradation_agreement": paired_metrics(
            reference[reference_full], combined["normalized"][reference_full, 0]
        ),
        "spectral_degradation_agreement": paired_metrics(
            constrained["normalized"][:, 0], fine_broad["normalized"][:, 0]
        ),
        "noncommutation_same_bands": paired_metrics(
            broad_reference[broad_reference_full, 0],
            combined["normalized"][broad_reference_full, 0],
        ),
        "real_modis_agreement": paired_metrics(reference[paired], real["normalized"][paired, 0]),
        "real_vs_simulated_modis_agreement": paired_metrics(
            combined["normalized"][paired, 0], real["normalized"][paired, 0]
        ),
        "spatial_block_bootstrap": bootstrap_blocks(
            reference, real["normalized"][:, 0], paired, coarse_shape, config["seed"]
        ),
        "modis_solar_zenith_median_deg": float(np.median(solar[paired])) if paired.any() else None,
        "modis_sensor_zenith_median_deg": float(np.median(view[paired])) if paired.any() else None,
        "grain_radius_has_independent_ground_validation": False,
        "expanded_components_noise_std_mean": {
            name: (float(value) if np.isfinite(value) else None)
            for name, value in zip(
                ["snow", "vegetation", "rock", "soil", "quartz_proxy"], variability_mean
            )
        },
        "true_gravel_spectrum_obtained": False,
        "footprint_projection": "four-corner polygon intersection in equal-area MODIS sinusoidal CRS",
    }
    # 代表像元的残差曲线与 RMS 均由原始系数重建光谱计算。
    selected = []
    for name, component in [("snow", 0), ("vegetation", 1), ("rock", 2), ("soil", 3)]:
        scores = constrained["normalized"][:, component].copy()
        scores[~np.isfinite(scores)] = -np.inf
        pixel = int(np.argmax(scores))
        selected.append((name, pixel))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    residual_rows = []
    for ax, (name, pixel) in zip(axes.ravel(), selected):
        index = constrained["radius_index"][pixel]
        coefficients = constrained["fractions"][pixel]
        fitted = coefficients[0] * snow[index] + background @ coefficients[1:]
        residual = data[pixel] - fitted
        # 强水吸收波段设为 NaN，使残差曲线在缺测区间断开。
        full_residual = np.full(len(w), np.nan)
        full_residual[band_mask] = residual * 100
        ax.plot(w, full_residual, color="#9e4a2e")
        ax.axhline(0, color="black", lw=0.6)
        ax.set(title=f"{name}代表像元：{pixel}", xlabel="波长 / μm", ylabel="E_c / pp")
        for wavelength, observed, modeled, error in zip(
            w[band_mask], data[pixel], fitted, residual
        ):
            residual_rows.append(
                {
                    "representative": name,
                    "pixel_index": pixel,
                    "wavelength_um": wavelength,
                    "observed": observed,
                    "modeled": modeled,
                    "E_c": error,
                }
            )
    save_figure(fig, path, "aviris_pixel_residuals")
    pd.DataFrame(residual_rows).to_csv(
        path / "tables/representative_pixel_residuals.csv", index=False
    )
    write_json(
        path / "spatial_metadata.json",
        {
            "aviris_crs": crs.to_wkt(),
            "aviris_transform": list(transform)[:6],
            "modis_crs": MODIS_CRS.to_wkt(),
            "modis_transform": list(coarse_transform)[:6],
            "tile_window_row_col": tile_window,
            "coarse_nominal_500m_actual_spacing_m": coarse_transform.a,
        },
    )
    (path / "README.md").write_text(
        "# AVIRIS、模拟MODIS与真实MODIS\n\nAVIRIS 14.3 m数据面积平均至30 m后处理；使用现代182波段掩膜，历史175波段设置见方法差异。库端元为针叶、固体花岗岩、土壤与quartz粉末；quartz_proxy为石英粉末材料代理。\n\n原文约束和FCLS均保存原始与阴影归一化系数。比例图色标0–1仅用于可读展示，原始越界值保留在GeoTIFF/NPZ及诊断统计。每波段残差由原始系数重建。\n\nMODIS模拟为一致的RSR加权反射率卷积＋四角投影像元相交面积聚合，大气、BRDF与PSF影响列为后续敏感性因素。粗像元至少95%被有效AVIRIS覆盖后参与对比。真实MOD09GA取当天第一观测层（_1），保留雪QA标记、剔除云/阴影/卷云/邻云和坏波段。观测时间使用产品日期，观测指针和角度一并保存。\n\n聚合AVIRIS提供模型一致性参照，雪面积加权粒径用于尺度比较。独立精度评价需要同期地面测量。\n",
        encoding="utf-8",
    )
    finish_run(path, metrics)
    return path
