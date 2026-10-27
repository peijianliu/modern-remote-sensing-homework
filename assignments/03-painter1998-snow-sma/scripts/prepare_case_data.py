"""从公开遥感输入准备教学数据和统一背景的受控补实验。

完整重建需 --source-root 指向已准备的 AVIRIS、MOD09GA 与模型缓存。
该工作区还需保留已归档的 v2 尺度及 WorldView CSV；默认教学使用仓库冻结文件。
Notebook 默认使用 data/demo，运行时不下载原始影像。
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.case_analysis import (
    absorption_area,
    binary_agreement,
    fixed_radius_scan,
    select_quantile_indices,
)
from snowmix.spectra import gaussian_band_matrix


def digest(path: Path) -> str:
    """返回文件 SHA-256。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, value: dict) -> None:
    """以 UTF-8 保存有限值 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fit_summary(fractions, normalized, rms, condition) -> dict:
    """汇总共同观测集合上的系数、归一化比例及光谱误差。"""
    good = np.isfinite(rms)
    snow = normalized[:, 0]
    finite = good & np.isfinite(snow)
    return {
        "n": int(good.sum()),
        "normalized_valid_n": int(finite.sum()),
        "raw_snow_mean": float(np.nanmean(fractions[good, 0])),
        "normalized_snow_mean": float(np.nanmean(snow[finite])),
        "rms_mean": float(np.nanmean(rms[good])),
        "negative_snow_fraction": float(np.mean(snow[finite] < 0)),
        "over_one_snow_fraction": float(np.mean(snow[finite] > 1)),
        "condition_median": float(np.nanmedian(condition[good])),
    }


def prepare(source_root: Path, output: Path, verification: Path | None = None) -> None:
    """保存新版本结果；拒绝覆盖已有教学文件。"""
    import snowmix.image as image

    started = time.perf_counter()
    if output.exists() and any(path.name != "README.md" for path in output.iterdir()):
        raise FileExistsError("Demo output exists; select a new directory")
    output.mkdir(parents=True, exist_ok=True)
    image.ROOT = source_root
    config = json.loads((source_root / "configs/baseline.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (source_root / "data/cache/aviris_metadata.json").read_text(encoding="utf-8")
    )
    cube, valid, bands, transform, shape, crs, bounds = image.prepare_roi(config, metadata)
    w = np.asarray(metadata["wavelength_um"])
    fwhm = np.asarray(metadata["fwhm_um"])
    optical = loadmat(source_root / "data/cache/forward_models.mat")
    all_radii = optical["radii_um"].ravel()
    keep = (all_radii >= 30) & (all_radii <= 1100)
    radii = all_radii[keep]
    response = gaussian_band_matrix(optical["wavelength_um"].ravel(), w[bands], fwhm[bands])
    ai = int(np.flatnonzero(optical["angles_deg"].ravel() == 22)[0])
    snow = optical["snicar_albedo"][keep, :, ai, 0] @ response
    flat = np.flatnonzero(valid)
    observation = cube[..., bands].reshape(-1, int(bands.sum()))[flat].astype(float)

    def band(target):
        return cube[..., int(np.argmin(abs(w - target)))]

    green, nir, red, swir = band(0.55), band(0.86), band(0.65), band(1.64)
    ndsi = (green - swir) / (green + swir + 1e-10)
    ndvi = (nir - red) / (nir + red + 1e-10)
    veg = np.flatnonzero(valid & (ndvi > 0.5) & (ndsi < 0))
    bare = np.flatnonzero(valid & (abs(ndvi) < 0.1) & (ndsi < 0))
    if not len(veg) or len(bare) < 2:
        raise ValueError("Prespecified background selection has insufficient candidates")
    # 选样只使用预定指数与亮度，不依赖新增模型的拟合效果。
    vi = int(veg[np.lexsort((veg, -ndvi.ravel()[veg]))[0]])
    targets = np.quantile(green.ravel()[bare], [0.9, 0.1])
    selected_background = [vi]
    for target in targets:
        order = np.lexsort((bare, abs(green.ravel()[bare] - target)))
        selected_background.append(
            next(int(bare[i]) for i in order if int(bare[i]) not in selected_background)
        )
    background = cube[..., bands].reshape(-1, int(bands.sum()))[selected_background].T.astype(float)
    background_names = ["vegetation_example", "bright_bare_example", "dark_bare_example"]
    print(
        f"CASE input: valid={len(flat)}, bands={bands.sum()}, background pixels={selected_background}",
        flush=True,
    )
    with threadpool_limits(limits=1):
        scan = fixed_radius_scan(observation, snow, radii, background, include_spectra=False)
        physical = image.unmix_pixels(
            observation, snow, radii, background, config, progress="physical MGS"
        )
        constrained = image.unmix_pixels(
            observation, snow, radii, background, config, solver="fcls", progress="physical FCLS"
        )
    candidate = (
        valid & (ndsi > 0.7) & (abs(ndvi) < 0.15) & (green >= 0.65) & (green <= 1.1) & (nir > 0.11)
    )
    candidates = np.flatnonzero(candidate)
    full_observation = cube[..., bands].reshape(-1, int(bands.sum()))
    areas = absorption_area(w[bands], full_observation[candidates])
    selected = select_quantile_indices(areas, candidates)
    image_snow = full_observation[selected].astype(float)
    selected_area = absorption_area(w[bands], image_snow)
    with threadpool_limits(limits=1):
        empirical = image.unmix_pixels(
            observation,
            image_snow,
            np.arange(1, 6),
            background,
            config,
            progress="five image spectra MGS",
        )

    def full(array):
        result = np.full(np.prod(shape), np.nan, dtype=np.float32)
        result[flat] = array
        return result.reshape(shape)

    scan_rows = [
        {
            "radius_um": float(radius),
            **fit_summary(
                scan["fractions"][i], scan["normalized"][i], scan["rms"][i], scan["condition"][i]
            ),
        }
        for i, radius in enumerate(scan["radii_um"])
    ]
    pd.DataFrame(scan_rows).to_csv(output / "fixed_radius_summary.csv", index=False)
    comparison = [
        {
            "method": name,
            **fit_summary(
                result["fractions"], result["normalized"], result["rms"], result["condition"]
            ),
        }
        for name, result in [
            ("physical_library_MGS", physical),
            ("five_image_spectra_MGS", empirical),
            ("physical_library_FCLS", constrained),
        ]
    ]
    pd.DataFrame(comparison).to_csv(output / "image_endmember_summary.csv", index=False)
    endmembers = pd.DataFrame(
        {
            "candidate_id": np.arange(1, 6),
            "quantile": [0.05, 0.25, 0.5, 0.75, 0.95],
            "row": selected // shape[1],
            "column": selected % shape[1],
            "absorption_area_um": selected_area,
        }
    )
    endmembers.to_csv(output / "image_endmembers.csv", index=False)
    # 代表光谱为候选中位雪谱及三个预设背景；它们用于方法示例。
    representatives = [int(selected[2]), *selected_background]
    representative_array = full_observation[representatives].astype(float)
    representative_scan = fixed_radius_scan(representative_array, snow, radii, background)
    np.savez_compressed(output / "representative_scan.npz", **representative_scan)
    rgb = np.clip(
        np.nan_to_num(np.stack([band(v) for v in (0.65, 0.55, 0.47)], axis=-1)) / 0.65, 0, 1
    )
    rgb[~valid] = 0.9
    # 分母从原始系数直接求和，保存 float64。
    denominator = np.full((len(scan["radii_um"]), *shape), np.nan, dtype=np.float64)
    denominator.reshape(len(scan["radii_um"]), -1)[:, flat] = 1 - scan["shade"]
    np.savez_compressed(
        output / "aviris_case.npz",
        wavelength_um=w[bands],
        radii_um=radii,
        snow_spectra=snow,
        backgrounds=background,
        background_names=np.array(background_names),
        background_row_col=np.column_stack(
            (
                np.asarray(selected_background) // shape[1],
                np.asarray(selected_background) % shape[1],
            )
        ),
        representative_spectra=representative_array,
        representative_row_col=np.column_stack(
            (np.asarray(representatives) // shape[1], np.asarray(representatives) % shape[1])
        ),
        rgb=rgb.astype(np.float32),
        valid_mask=valid,
        fixed_radii_um=scan["radii_um"],
        fixed_snow_fraction=np.stack([full(item[:, 0]) for item in scan["normalized"]]),
        fixed_raw_snow=np.stack([full(item[:, 0]) for item in scan["fractions"]]),
        fixed_rms=np.stack([full(item) for item in scan["rms"]]),
        fixed_denominator=denominator,
        physical_snow_fraction=full(physical["normalized"][:, 0]),
        physical_radius_um=full(physical["radius_um"]),
        physical_rms=full(physical["rms"]),
        image_snow_fraction=full(empirical["normalized"][:, 0]),
        image_rms=full(empirical["rms"]),
        image_snow_spectra=image_snow,
        image_absorption_area_um=selected_area,
        image_endmember_row_col=np.column_stack((selected // shape[1], selected % shape[1])),
        fcls_snow_fraction=full(constrained["normalized"][:, 0]),
    )
    spatial = {
        "aviris_crs": crs.to_wkt(),
        "aviris_transform": list(transform)[:6],
        "aviris_shape": list(shape),
        "aviris_solar_zenith_setting_deg": 22,
        "modis_solar_zenith_setting_deg": 25,
        "background_classes": background_names,
        "representative_classes": ["snow_example", *background_names],
        "background_selection": {
            "vegetation": "NDVI>0.5 and NDSI<0; largest NDVI, row-major tie",
            "bare": "|NDVI|<0.1 and NDSI<0; green 90th/10th quantile nearest distinct pixel",
        },
        "background_row_col": np.column_stack(
            (
                np.asarray(selected_background) // shape[1],
                np.asarray(selected_background) % shape[1],
            )
        ).tolist(),
    }
    agreement_rows, units = prepare_modis_units(
        source_root,
        metadata,
        w,
        bands,
        fwhm,
        radii,
        background,
        config,
        valid,
        transform,
        shape,
        crs,
        constrained,
        spatial,
    )
    units.to_csv(output / "modis_1km_units.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(output / "product_consistency.csv", index=False)
    dump(output / "spatial_metadata.json", spatial)
    finish_inputs(source_root, output)
    summary = {
        "input_shape": [*shape, int(bands.sum())],
        "input_valid_pixels": len(flat),
        "candidate_snow_pixels": int(candidate.sum()),
        "background_classes": background_names,
        "background_selection": spatial["background_selection"],
        "background_row_col": spatial["background_row_col"],
        "fixed_radius_scan": scan_rows,
        "image_endmember_comparison": comparison,
        "image_endmembers": endmembers.to_dict(orient="records"),
        "product_consistency": agreement_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(output / "case_summary.json", summary)
    finalize_case_statistics(output)
    if verification:
        verification.mkdir(parents=True, exist_ok=True)
        dump(
            verification / "case_summary.json",
            json.loads((output / "case_summary.json").read_text(encoding="utf-8")),
        )
    finish_manifest(source_root, output, config)
    print(
        json.dumps(
            {
                "valid_pixels": len(flat),
                "candidate_snow_pixels": len(candidates),
                "one_km_common": units.groupby("roi").common_valid.sum().to_dict(),
                "bytes": sum(path.stat().st_size for path in output.iterdir()),
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def prepare_modis_units(
    source_root,
    metadata,
    w,
    bands,
    fwhm,
    radii,
    background,
    config,
    valid,
    transform,
    shape,
    crs,
    constrained,
    spatial,
):
    """将两套模型按实际相交面积聚合到原始 1 km 状态单元。"""
    from affine import Affine
    import snowmix.image as image
    from snowmix.geo import MODIS_CRS, MODIS_PIXEL_SIZE, area_weights, aggregate, modis_roi_grid
    from snowmix.modis import quality_mask
    from snowmix.spectra import modis_responses, response_matrix

    modis = loadmat(source_root / "data/cache/modis_fields.mat")
    attributes = loadmat(source_root / "data/cache/modis_attributes.mat")
    scale = float(attributes["scale"].item())
    qc_fill = int(attributes["qc_fill"].item())
    rw, rs = modis_responses(source_root / "data/raw/Terra_RSR_in-band.xlsx")
    unique, indices = np.unique(w, return_index=True)
    operator_unique, _ = response_matrix(unique, rw, rs, bands[indices])
    operator = np.zeros((len(w), 7))
    operator[indices] = operator_unique
    if not np.isfinite(operator).all():
        raise ValueError("MODIS responses lack AVIRIS wavelength support")
    background7 = background.T @ operator[bands]
    optics25 = loadmat(source_root / "data/cache/forward_models_25deg.mat")
    source_radii = optics25["radii_um"].ravel()
    keep = np.isin(source_radii, radii)
    rsr25 = gaussian_band_matrix(optics25["wavelength_um"].ravel(), w[bands], fwhm[bands])
    snow25 = optics25["snicar_albedo"][keep, :, 0, 0] @ rsr25 @ operator[bands]
    all_units = []
    spatial["modis_crs"] = MODIS_CRS.to_wkt()
    spatial["one_km_windows"] = []
    for label, config_name in [("Mammoth", "baseline.json"), ("south_low_snow", "clear_roi.json")]:
        roi_config = json.loads((source_root / "configs" / config_name).read_text(encoding="utf-8"))
        if config_name == "baseline.json":
            roi_valid, roi_transform, roi_shape, roi_crs, result = (
                valid,
                transform,
                shape,
                crs,
                constrained,
            )
            roi_bounds = (
                transform.c,
                transform.f - shape[0] * 30,
                transform.c + shape[1] * 30,
                transform.f,
            )
        else:
            roi_cube, roi_valid, _, roi_transform, roi_shape, roi_crs, roi_bounds = (
                image.prepare_roi(roi_config, metadata)
            )
            values = roi_cube[..., bands].reshape(-1, int(bands.sum())).astype(float)
            values[~roi_valid.ravel()] = np.nan
            optics22 = loadmat(source_root / "data/cache/forward_models.mat")
            angle = int(np.flatnonzero(optics22["angles_deg"].ravel() == 22)[0])
            snow22 = optics22["snicar_albedo"][keep, :, angle, 0] @ gaussian_band_matrix(
                optics22["wavelength_um"].ravel(), w[bands], fwhm[bands]
            )
            with threadpool_limits(limits=1):
                result = image.unmix_pixels(
                    values,
                    snow22,
                    radii,
                    background,
                    roi_config,
                    solver="fcls",
                    progress="south FCLS",
                )
        grid500, grid_shape, tile_window = modis_roi_grid(roi_bounds, roi_crs)
        r0, r1, c0, c1 = tile_window
        # _1km 原生索引确定单元，不把其复制成四个独立参照。
        sr0, sr1, sc0, sc1 = r0 // 2, (r1 + 1) // 2, c0 // 2, (c1 + 1) // 2
        tile_left = grid500.c - c0 * MODIS_PIXEL_SIZE
        tile_top = grid500.f + r0 * MODIS_PIXEL_SIZE
        grid1k = Affine(
            MODIS_PIXEL_SIZE * 2,
            0,
            tile_left + sc0 * MODIS_PIXEL_SIZE * 2,
            0,
            -MODIS_PIXEL_SIZE * 2,
            tile_top - sr0 * MODIS_PIXEL_SIZE * 2,
        )
        shape1k = (sr1 - sr0, sc1 - sc0)
        weights_fine = area_weights(roi_transform, roi_shape, roi_crs, grid1k, shape1k, MODIS_CRS)
        fine_snow = (
            result["normalized"][:, 0]
            if len(result["normalized"]) == np.prod(roi_shape)
            else np.full(np.prod(roi_shape), np.nan)
        )
        if len(result["normalized"]) != np.prod(roi_shape):
            fine_snow[np.flatnonzero(roi_valid)] = result["normalized"][:, 0]
        fine_snow = fine_snow.copy()
        fine_snow[~roi_valid.ravel()] = np.nan
        fine_mean, fine_coverage = aggregate(weights_fine, fine_snow)
        raw = modis["band_raw"][r0:r1, c0:c1]
        qc = modis["qc"][r0:r1, c0:c1]
        state500 = np.repeat(np.repeat(modis["state"], 2, 0), 2, 1)[r0:r1, c0:c1]
        qa = quality_mask(raw, qc, state500, qc_fill=qc_fill).ravel()
        weights500 = area_weights(roi_transform, roi_shape, roi_crs, grid500, grid_shape, MODIS_CRS)
        _, fine_input_coverage = aggregate(weights500, np.zeros(np.prod(roi_shape)), roi_valid)
        observation = raw.reshape(-1, 7).astype(float) * scale
        observation[~qa | (fine_input_coverage < 0.95)] = np.nan
        with threadpool_limits(limits=1):
            real = image.unmix_pixels(
                observation, snow25, radii, background7.T, roi_config, solver="fcls"
            )
        weights_modis = area_weights(grid500, grid_shape, MODIS_CRS, grid1k, shape1k, MODIS_CRS)
        modis_mean, modis_coverage = aggregate(weights_modis, real["normalized"][:, 0])
        state = modis["state"][sr0:sr1, sc0:sc1].ravel()
        flag12 = ((state >> 12) & 1).astype(float)
        flag15 = ((state >> 15) & 1).astype(float)
        flag12[state == 65535] = np.nan
        flag15[state == 65535] = np.nan
        common = (fine_coverage >= 0.95) & (modis_coverage >= 0.95) & (state != 65535)
        units = pd.DataFrame(
            {
                "roi": label,
                "tile_row_1km": np.repeat(np.arange(sr0, sr1), shape1k[1]),
                "tile_column_1km": np.tile(np.arange(sc0, sc1), shape1k[0]),
                "state_1km": state,
                "flag12": flag12,
                "flag15": flag15,
                "aviris_fsca": fine_mean[:, 0],
                "aviris_valid_area_fraction": fine_coverage,
                "modis_fsca": modis_mean[:, 0],
                "modis_valid_area_fraction": modis_coverage,
                "common_valid": common,
            }
        )
        assert not units.duplicated(["tile_row_1km", "tile_column_1km"]).any()
        all_units.append(units)
        spatial["one_km_windows"].append(
            {
                "roi": label,
                "transform": list(grid1k)[:6],
                "shape": list(shape1k),
                "tile_window": [sr0, sr1, sc0, sc1],
            }
        )
        print(f"CASE 1 km {label}: common={common.sum()}", flush=True)
    units = pd.concat(all_units, ignore_index=True)
    rows = []
    for label, group in units.groupby("roi", sort=False):
        common = group.common_valid.to_numpy(dtype=bool)
        for name in ["aviris", "modis"]:
            for bit in [12, 15]:
                for threshold in [0.1, 0.5, 0.9]:
                    rows.append(
                        {
                            "roi": label,
                            "model": name,
                            "bit": bit,
                            **binary_agreement(
                                group[f"flag{bit}"].to_numpy()[common],
                                group[f"{name}_fsca"].to_numpy()[common],
                                threshold,
                            ),
                        }
                    )
    return rows, units


def finish_inputs(source_root: Path, output: Path) -> None:
    """复制公开计算缓存、USGS 石英谱和已归档的派生统计。"""
    copies = {
        "data/raw/spectra/quartz_hs32.5854.asc": "quartz_hs32.5854.asc",
        "data/cache/aviris_metadata.json": "aviris_metadata.json",
    }
    corrected = (
        source_root / "experiments/painter1998/2026-10-01/v2/15_review_reanalysis_r002/tables"
    )
    for name in [
        "05_07_image_r001_corrected_coverage.csv",
        "05_07_image_r002_corrected_coverage.csv",
        "same_dictionary_control_corrected.csv",
    ]:
        copies[(corrected / name).relative_to(source_root).as_posix()] = name
    worldview = (
        source_root / "experiments/painter1998/2026-10-01/v2/13_worldview_validation_r004/tables"
    )
    for path in worldview.glob("*_sample_pixels.csv"):
        copies[path.relative_to(source_root).as_posix()] = path.name
    copies[(worldview / "stratified_metrics.csv").relative_to(source_root).as_posix()] = (
        "stratified_metrics.csv"
    )
    for source, destination in copies.items():
        if destination.endswith("_sample_pixels.csv"):
            payload = (source_root / source).read_bytes()
            (output / (destination + ".gz")).write_bytes(
                gzip.compress(payload, compresslevel=6, mtime=0)
            )
        else:
            shutil.copy2(source_root / source, output / destination)
    # 教学入口使用现算 WW80，完整 MATLAB MAT 仅属于可选的完整研究流程。
    template = Path(__file__).resolve().parents[1] / "data/demo"
    for name in (
        "scattering_inputs.npz",
        "scattering_inputs.source.json",
        "solar_illumination.npz",
        "forward_reference.npz",
        "forward_reference.source.json",
        "sample_support.csv",
        "lab_conditions.csv",
    ):
        shutil.copy2(template / name, output / name)


def finish_manifest(source_root: Path, output: Path, config: dict) -> None:
    """记录公开来源、版本与教学文件哈希。"""
    sources = [
        {
            "id": "aviris",
            "url": "https://popo.jpl.nasa.gov/avcl/y13_data/f130503t01p00r14_refl.tar.gz",
            "product": "AVIRIS-Classic L2 2013-05-03 f130503t01p00r14",
            "access": "public NASA/JPL data",
            "sha256": digest(source_root / "data/raw/f130503t01p00r14_refl.tar.gz"),
        },
        {
            "id": "modis",
            "url": "https://cmr.earthdata.nasa.gov/search/concepts/G2261932874-LPCLOUD.html",
            "product": "MOD09GA.A2013123.h08v05.061.2021229143331",
            "access": "public NASA data",
            "sha256": digest(
                source_root / "data/raw/MOD09GA.A2013123.h08v05.061.2021229143331.hdf"
            ),
        },
        {
            "id": "snow_model",
            "url": "https://github.com/mflanner/SNICARv3",
            "license": "MIT code",
            "optical_properties_version": "202105",
            "contents": "calculated spectra; original optical database excluded",
        },
        {
            "id": "usgs_quartz",
            "url": "https://pubs.usgs.gov/of/2003/ofr-03-395/DESCRIPT/M/quartz_hs32.html",
            "access": "public USGS spectral library",
        },
        {
            "id": "worldview",
            "url": "https://doi.org/10.5281/zenodo.4031446",
            "license": "CC BY 4.0",
            "creators": ["Timbo Stillinger", "Ned Bair"],
        },
        {
            "id": "legacy_statistics",
            "contents": "corrected scale and WorldView model statistics; legacy_library_output",
            "note": "Historical derived outputs; not combined with the new AVIRIS image-background experiments. Original JHU spectra are not distributed.",
        },
    ]
    template = Path(__file__).resolve().parents[1] / "data/demo/manifest.json"
    for source in json.loads(template.read_text(encoding="utf-8"))["sources"]:
        if source["id"] in {"snicar_optical_inputs", "lab_snow_conditions"}:
            sources.append(source)
    files = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        record = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "source_id": "this_experiment",
        }
        if (
            path.name.endswith(("_sample_pixels.csv", "_sample_pixels.csv.gz"))
            or path.name == "stratified_metrics.csv"
        ):
            source = Path(
                "experiments/painter1998/2026-10-01/v2/13_worldview_validation_r004/tables"
            ) / path.name.removesuffix(".gz")
            if path.suffix == ".gz":
                payload = gzip.decompress(path.read_bytes())
                record.update(
                    uncompressed_bytes=len(payload),
                    uncompressed_sha256=hashlib.sha256(payload).hexdigest(),
                    gzip_mtime=0,
                )
            record.update(
                source_id="legacy_worldview",
                source_path=source.as_posix(),
                source_sha256=digest(source_root / source),
            )
        elif (
            path.name.endswith("_corrected_coverage.csv")
            or path.name == "same_dictionary_control_corrected.csv"
        ):
            source = (
                Path("experiments/painter1998/2026-10-01/v2/15_review_reanalysis_r002/tables")
                / path.name
            )
            record.update(
                source_id="legacy_scale",
                source_path=source.as_posix(),
                source_sha256=digest(source_root / source),
            )
        elif path.name in {"quartz_hs32.5854.asc", "aviris_metadata.json"}:
            record["source_id"] = {
                "quartz_hs32.5854.asc": "usgs_quartz",
                "aviris_metadata.json": "aviris",
            }[path.name]
        elif path.name.startswith(("scattering_inputs",)):
            record["source_id"] = "snicar_optical_inputs"
        elif path.name.startswith(("forward_reference", "solar_illumination")):
            record["source_id"] = "snow_model"
        elif path.name == "lab_conditions.csv":
            record["source_id"] = "lab_snow_conditions"
        files.append(record)
    dump(
        output / "manifest.json",
        {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sources": sources,
            "experiment": {
                "project": "painter1998",
                "run_date": "2026-10-24",
                "version": "v3/group_case_r001",
            },
            "base_configuration": config,
            "code_sha256": {
                name: digest(Path(__file__).resolve().parents[1] / name)
                for name in [
                    "scripts/prepare_case_data.py",
                    "snowmix/case_analysis.py",
                    "snowmix/linear.py",
                    "snowmix/image.py",
                    "snowmix/geo.py",
                ]
            },
            "files": files,
        },
    )


def finalize_case_statistics(output: Path) -> None:
    """从冻结图层计算共同归一化域统计，保留逐模型可用域诊断。"""
    with np.load(output / "aviris_case.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    common = np.isfinite(arrays["fixed_snow_fraction"]).all(axis=0)
    model_common = (
        np.isfinite(arrays["physical_snow_fraction"])
        & np.isfinite(arrays["image_snow_fraction"])
        & np.isfinite(arrays["fcls_snow_fraction"])
    )
    arrays["common_normalization_mask"] = common
    arrays["model_comparison_common_normalization_mask"] = model_common
    np.savez_compressed(output / "aviris_case.npz", **arrays)
    table = pd.read_csv(output / "fixed_radius_summary.csv")
    for i in range(len(table)):
        snow = arrays["fixed_snow_fraction"][i][common]
        table.loc[i, "common_normalized_n"] = int(common.sum())
        table.loc[i, "common_normalized_snow_mean"] = np.mean(snow, dtype=np.float64)
        table.loc[i, "common_raw_snow_mean"] = np.mean(
            arrays["fixed_raw_snow"][i][common], dtype=np.float64
        )
        table.loc[i, "common_rms_mean"] = np.mean(arrays["fixed_rms"][i][common], dtype=np.float64)
        table.loc[i, "common_negative_snow_fraction"] = np.mean(snow < 0)
        table.loc[i, "common_over_one_snow_fraction"] = np.mean(snow > 1)
    table["common_normalized_n"] = table.common_normalized_n.astype(int)
    table.to_csv(output / "fixed_radius_summary.csv", index=False)
    comparison = pd.read_csv(output / "image_endmember_summary.csv")
    for i, field in enumerate(
        ["physical_snow_fraction", "image_snow_fraction", "fcls_snow_fraction"]
    ):
        comparison.loc[i, "common_normalized_n"] = int(model_common.sum())
        comparison.loc[i, "common_normalized_snow_mean"] = np.mean(
            arrays[field][model_common], dtype=np.float64
        )
    comparison["common_normalized_n"] = comparison.common_normalized_n.astype(int)
    comparison.to_csv(output / "image_endmember_summary.csv", index=False)
    summary = json.loads((output / "case_summary.json").read_text(encoding="utf-8"))
    summary.update(
        fixed_normalization_common_n=int(common.sum()),
        model_normalization_common_n=int(model_common.sum()),
        fixed_radius_scan=table.to_dict(orient="records"),
        image_endmember_comparison=comparison.to_dict(orient="records"),
        experiment={
            "project": "painter1998",
            "run_date": "2026-10-24",
            "version": "v3/group_case_r001",
        },
    )
    dump(output / "case_summary.json", summary)
    from snowmix import teaching

    diagnostic, bootstrap = teaching.fixed_radius_statistics(arrays)
    diagnostic.to_csv(output / "fixed_radius_uncertainty.csv", index=False)
    dump(output / "fixed_radius_bootstrap.json", bootstrap)
    units = pd.read_csv(output / "modis_1km_units.csv")
    teaching.coverage_sensitivity(units).to_csv(output / "coverage_sensitivity.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/demo"
    )
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    prepare(args.source_root.resolve(), args.output.resolve(), args.verification)
