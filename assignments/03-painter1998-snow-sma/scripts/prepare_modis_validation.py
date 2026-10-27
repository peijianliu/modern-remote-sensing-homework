"""将公开WorldView派生30 m覆盖按相交面积配到同日MOD09GA，重算独立验证。

原始商业影像不参与。HDF字段先由prepare_modis_hdf.py读取；默认教学只需本脚本
产出的七波段小表和模型输入，Notebook会重新求解。
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

from affine import Affine
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from scipy.io import loadmat

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from snowmix import teaching as t
from snowmix.geo import MODIS_CRS, MODIS_PIXEL_SIZE, modis_roi_grid, area_weights, aggregate
from snowmix.modis import quality_mask, subset_quality_mask
from snowmix.modis_validation import evaluate_pixels, evaluate_flags
from snowmix.optics import ww80_semi_infinite
from snowmix.spectra import modis_responses, response_matrix, read_usgs_ascii

REFERENCE_NAMES = {
    "20160213": "16FEB13190431_landsat_30m_fsca.tif",
    "20170521": "17MAY21191849_landsat_30m_fsca.tif",
}


def digest(path: Path) -> str:
    """计算原始输入或派生文件的SHA-256。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_gzip(table: pd.DataFrame, path: Path) -> None:
    """使用固定gzip时间戳，重复写出相同表得到相同字节。"""
    with path.open("xb") as stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=9
        ) as compressed:
            compressed.write(table.to_csv(index=False).encode("utf-8"))


def prepare(source: Path, field_root: Path, output: Path) -> dict:
    """从两套公开完整输入生成像元表、原生状态表、光谱库和评价结果。"""
    if output.exists():
        raise FileExistsError("Use a new independent-validation output directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    optics = t.load_optical_inputs()
    rw, rs = modis_responses(source / "data/raw/Terra_RSR_in-band.xlsx")
    response, coverage = response_matrix(optics["wavelength_um"], rw, rs)
    if not np.isfinite(response).all():
        raise ValueError("Snow grid lacks MODIS response support")
    radii = optics["radii_um"]
    radius_mask = (radii >= 30) & (radii <= 1100)
    backgrounds = []
    for name in ("lodgepole_pine.asc", "quartz_hs32.5854.asc"):
        wave, reflectance, _ = read_usgs_ascii(t.demo_directory() / name)
        weights, support = response_matrix(wave, rw, rs)
        if not np.isfinite(weights).all():
            raise ValueError("Public background lacks MODIS support")
        backgrounds.append(reflectance @ weights)
    pixel_tables, unit_tables, snows, angles, records = [], [], [], [], []
    for scene, reference_name in REFERENCE_NAMES.items():
        reference_path = source / "data/raw/public_validation/zenodo_4031446" / reference_name
        fields = (
            field_root
            / ("modis_2016" if scene.startswith("2016") else "modis_2017")
            / "modis_fields.mat"
        )
        origin = json.loads(fields.with_suffix(".source.json").read_text(encoding="utf-8"))
        day_code = dt.datetime.strptime(scene, "%Y%m%d").strftime("%Y%j")
        if not origin["native_id_verified"] or not origin["product"].startswith(
            f"MOD09GA.A{day_code}.h08v05.061."
        ):
            raise ValueError("MODIS field extraction has no verified native product identity")
        data = loadmat(fields)
        attributes = loadmat(fields.parent / "modis_attributes.mat")
        scale = float(attributes["scale"].item())
        if scale != 0.0001:
            raise ValueError("Unexpected native MOD09GA reflectance scale")
        with rasterio.open(reference_path) as dataset:
            raw_reference = dataset.read(1)
            valid = (raw_reference >= 0) & (raw_reference <= 1000)
            if not valid.any():
                raise ValueError("Reference contains no valid snow-cover fractions")
            reference = np.where(valid, raw_reference / 1000.0, np.nan)
            grid, shape, window = modis_roi_grid(dataset.bounds, dataset.crs)
            # 只为有效参考像元构建多边形，CSR仍保留原始像元索引。
            print(
                f"{scene}: reference {valid.sum():,} pixels; constructing intersections", flush=True
            )
            weights = area_weights(
                dataset.transform,
                dataset.shape,
                dataset.crs,
                grid,
                shape,
                MODIS_CRS,
                source_valid=valid,
            )
            reference500, coverage500 = aggregate(weights, reference.ravel())
            reference500 = reference500[:, 0]
            rr, cc = np.indices(shape)
            x = grid.c + (cc.ravel() + 0.5) * grid.a
            y = grid.f + (rr.ravel() + 0.5) * grid.e
            utm_x, utm_y = Transformer.from_crs(MODIS_CRS, dataset.crs, always_xy=True).transform(
                x, y
            )
            ref_column, ref_row = (~dataset.transform) * (utm_x, utm_y)
            positions = np.column_stack((np.floor(ref_row), np.floor(ref_column))).astype(int)
        r0, r1, c0, c1 = window
        raw = data["band_raw"][r0:r1, c0:c1]
        qc = data["qc"][r0:r1, c0:c1]
        state = np.repeat(np.repeat(data["state"], 2, 0), 2, 1)[r0:r1, c0:c1]
        sun = np.repeat(np.repeat(data["solar_zenith_raw"], 2, 0), 2, 1)[r0:r1, c0:c1].ravel()
        eligible = (coverage500 >= 0.95) & (sun > 0) & (sun < 9000)
        if not eligible.any():
            raise ValueError("No supported reference cells with a valid solar zenith")
        zenith = float(np.median(sun[eligible])) * 0.01
        angles.append(zenith)
        albedo = ww80_semi_infinite(
            optics["ss_alb"][radius_mask, :, 0],
            optics["asm_prm"][radius_mask, :, 0],
            np.cos(np.deg2rad(np.float64(zenith))),
        )
        snows.append(albedo @ response)
        qa7 = quality_mask(raw, qc, state, qc_fill=int(attributes["qc_fill"].item())).ravel()
        qa5 = subset_quality_mask(raw, qc, state).ravel()
        keep = coverage500 > 0
        table = pd.DataFrame(
            {
                "scene": scene,
                "tile_row_500m": rr.ravel() + r0,
                "tile_column_500m": cc.ravel() + c0,
                "reference_fsca": reference500,
                "reference_coverage": coverage500,
                "reference_row": positions[:, 0],
                "reference_column": positions[:, 1],
                "qa_strict7": qa7.astype(int),
                "qa_selected5": qa5.astype(int),
                "QC_500m": qc.ravel(),
                "state_1km": state.ravel(),
                "solar_zenith_deg": sun * 0.01,
            }
        )
        reflectance = raw.reshape(-1, 7).astype(float) * scale
        for band in range(7):
            table[f"B{band + 1}"] = reflectance[:, band]
        pixel_tables.append(table[keep])
        # 原生1 km单元由四个原生500 m单元组成，按参考支持面积再次加权。
        sr0, sr1, sc0, sc1 = r0 // 2, (r1 + 1) // 2, c0 // 2, (c1 + 1) // 2
        tile_left = grid.c - c0 * MODIS_PIXEL_SIZE
        tile_top = grid.f + r0 * MODIS_PIXEL_SIZE
        grid1k = Affine(
            2 * MODIS_PIXEL_SIZE,
            0,
            tile_left + sc0 * 2 * MODIS_PIXEL_SIZE,
            0,
            -2 * MODIS_PIXEL_SIZE,
            tile_top - sr0 * 2 * MODIS_PIXEL_SIZE,
        )
        shape1k = (sr1 - sr0, sc1 - sc0)
        weights1k = area_weights(grid, shape, MODIS_CRS, grid1k, shape1k, MODIS_CRS)
        cover1k = np.asarray(weights1k @ coverage500).ravel()
        total = np.asarray(weights1k @ (np.nan_to_num(reference500) * coverage500)).ravel()
        reference1k = np.divide(total, cover1k, out=np.full(len(total), np.nan), where=cover1k > 0)
        native = data["state"][sr0:sr1, sc0:sc1].ravel().astype(np.uint16)
        clear = (
            (native != 65535)
            & ((native & 3) == 0)
            & (((native >> 2) & 1) == 0)
            & (((native >> 8) & 3) == 0)
        )
        clear &= (((native >> 10) & 1) == 0) & (((native >> 13) & 1) == 0)
        row, col = np.indices(shape1k)
        units = pd.DataFrame(
            {
                "scene": scene,
                "tile_row_1km": row.ravel() + sr0,
                "tile_column_1km": col.ravel() + sc0,
                "reference_fsca": reference1k,
                "reference_coverage": cover1k,
                "state_1km": native,
                "state_clear_no_adjacent": clear.astype(int),
                "flag12": ((native >> 12) & 1).astype(int),
                "flag15": ((native >> 15) & 1).astype(int),
            }
        )
        unit_tables.append(units[cover1k > 0])
        records.append(
            {
                "scene": scene,
                "reference": {
                    "name": reference_name,
                    "sha256": digest(reference_path),
                    "source": "https://doi.org/10.5281/zenodo.4031446",
                    "creators": ["Timbo Stillinger", "Ned Bair"],
                    "license": "CC BY 4.0",
                    "modifications": "Explicit fill mask; exact footprint-area aggregation from public 30 m derived reference",
                },
                "modis": origin,
                "scale": scale,
                "solar_zenith_median_deg": zenith,
                "solar_zenith_range_deg": [
                    float(sun[eligible].min() * 0.01),
                    float(sun[eligible].max() * 0.01),
                ],
                "reference_grid": {
                    "transform": list(grid)[:6],
                    "shape": list(shape),
                    "crs": MODIS_CRS.to_wkt(),
                },
                "supported_500m_n": int((coverage500 >= 0.95).sum()),
                "strict7_n": int(np.sum((coverage500 >= 0.95) & qa7)),
                "selected5_n": int(np.sum((coverage500 >= 0.95) & qa5)),
            }
        )
        print(
            json.dumps(
                records[-1] | {"reference_grid": "native MODIS sinusoidal"}, ensure_ascii=False
            ),
            flush=True,
        )
        del weights, data
    pixels = pd.concat(pixel_tables, ignore_index=True)
    units = pd.concat(unit_tables, ignore_index=True)
    library = {
        "scene_ids": np.array(list(REFERENCE_NAMES)),
        "radii_um": radii[radius_mask],
        "snow": np.stack(snows),
        "backgrounds": np.array(backgrounds).T,
        "solar_zenith_deg": np.array(angles),
    }
    np.savez_compressed(output / "modis_external_library.npz", **library)
    save_gzip(pixels, output / "modis_reference_pixels.csv.gz")
    units.to_csv(output / "modis_reference_1km.csv", index=False)
    metrics, paired, support, predictions = evaluate_pixels(pixels, library)
    flags = evaluate_flags(units)
    for name, table in (
        ("modis_external_metrics", metrics),
        ("modis_external_paired_deltas", paired),
        ("modis_external_support", support),
        ("modis_external_flags", flags),
    ):
        table.to_csv(output / (name + ".csv"), index=False)
    np.savez_compressed(
        output / "modis_external_predictions.npz",
        **{
            scene + "__" + key: value
            for scene, d in predictions.items()
            for key, value in d.items()
        },
    )
    provenance = {
        "sources": records,
        "snow_model": "NumPy WW80, pure spherical semi-infinite snow; Warren 1984",
        "snow_inputs": "Official SNICAR 202105 scattering properties; not new Mie calculations",
        "backgrounds": ["USGS lodgepole pine", "USGS Quartz HS32"],
        "snow_radii_um": [30, 1100, 10],
        "solar_angle_policy": "Median native MOD09GA solar zenith over reference-supported cells, without snow-label selection",
        "quality_tiers": {
            "strict7": "All seven bands highest quality; MODLAND ideal; clear and not adjacent to cloud",
            "selected5": "Bands 1/2/4/6/7 highest quality; MODLAND 0 or 1; same clear/no-adjacent state",
        },
        "comparison": "2016-02-13 and 2017-05-21 same-day independent visible snow-cover reference",
        "area_threshold": 0.95,
        "fraction_thresholds": [0.1, 0.5, 0.9],
        "units": "Reflectance and fractions dimensionless; error tables in pp",
        "seconds": time.perf_counter() - started,
    }
    (output / "modis_external.source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {"support": support.to_dict("records"), "seconds": provenance["seconds"]}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "stage": "independent_modis",
        "status": "complete",
        "output_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(support.to_string(index=False), flush=True)
    print(flags.query("reference_threshold==.1").to_string(index=False), flush=True)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--field-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root.resolve(), args.field_root.resolve(), args.output.resolve())
