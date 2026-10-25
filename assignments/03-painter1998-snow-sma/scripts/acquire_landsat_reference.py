"""按 WorldView 参考范围读取 Landsat C2 L2 的 COG 窗口。

下载结果保存原始 DN 值。公开数据服务的临时签名仅在请求时使用。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "raw" / "public_validation" / "zenodo_4031446"
OUTPUT = ROOT / "data" / "cache" / "public_validation"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
PAIRS = {
    "LC08_L2SP_043033_20160213_02_T1": "16FEB13190431_landsat_30m_fsca.tif",
    "LC08_L2SP_043032_20170522_02_T1": "17MAY21191849_landsat_30m_fsca.tif",
}
BANDS = {
    "blue": "SR_B2",
    "green": "SR_B3",
    "red": "SR_B4",
    "nir08": "SR_B5",
    "swir16": "SR_B6",
    "swir22": "SR_B7",
    "qa_pixel": "QA_PIXEL",
    "qa_radsat": "QA_RADSAT",
    "qa_aerosol": "SR_QA_AEROSOL",
}


def public_json(url: str, params: dict | None = None) -> dict:
    """读取公开 JSON 数据；请求失败时仅报告状态码。"""
    response = requests.get(url, params=params, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Public data service status {response.status_code}")
    return response.json()


def sha256(path: Path) -> str:
    """计算文件的 SHA-256 摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire_scene(scene: str, reference_name: str) -> None:
    """按 Landsat 原生网格保存覆盖 WorldView 足迹的窗口。"""
    folder = OUTPUT / scene
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "source_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in previous["files"]:
            if sha256(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("Existing Landsat artifact does not match its manifest")
        print(scene, "already verified")
        return
    item = public_json(f"{STAC}/collections/landsat-c2-l2/items/{scene}")
    metadata_path = folder / "stac_item_unsigned.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    with rasterio.open(REFERENCE / reference_name) as reference:
        reference_crs = reference.crs
        reference_bounds = reference.bounds
    files: list[dict] = []
    fixed_window = None
    fixed_transform = None
    fixed_crs = None
    for asset_name, band in BANDS.items():
        output_path = folder / f"{band}.tif"
        original_href = item["assets"][asset_name]["href"].split("?", 1)[0]
        if not output_path.exists():
            signed = public_json(SIGN, {"href": original_href})["href"]
            try:
                with rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_HTTP_TIMEOUT="90",
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
                ):
                    with rasterio.open(signed) as source:
                        if fixed_window is None:
                            bounds = transform_bounds(
                                reference_crs, source.crs, *reference_bounds, densify_pts=21
                            )
                            raw = from_bounds(*bounds, transform=source.transform)
                            left, top = int(np.floor(raw.col_off)), int(np.floor(raw.row_off))
                            right = int(np.ceil(raw.col_off + raw.width))
                            bottom = int(np.ceil(raw.row_off + raw.height))
                            fixed_window = Window(left, top, right - left, bottom - top)
                            fixed_window = fixed_window.intersection(
                                Window(0, 0, source.width, source.height)
                            )
                            fixed_transform = source.window_transform(fixed_window)
                            fixed_crs = source.crs
                        if (
                            source.crs != fixed_crs
                            or source.window_transform(fixed_window) != fixed_transform
                        ):
                            raise ValueError("Landsat bands do not share the expected native grid")
                        array = source.read(1, window=fixed_window)
                        profile = {
                            "driver": "COG",
                            "width": array.shape[1],
                            "height": array.shape[0],
                            "count": 1,
                            "dtype": array.dtype,
                            "crs": source.crs,
                            "transform": fixed_transform,
                            "compress": "DEFLATE",
                            "blocksize": 256,
                        }
                        if band.startswith("SR_B"):
                            profile["nodata"] = 0
                        with rasterio.open(output_path, "w", **profile) as destination:
                            destination.write(array, 1)
                            destination.set_band_description(1, band)
                            if band.startswith("SR_B"):
                                destination.scales = (0.0000275,)
                                destination.offsets = (-0.2,)
                            destination.update_tags(
                                source_item_id=scene,
                                source_asset=asset_name,
                                stored_values="raw_DN_unscaled",
                            )
            except (rasterio.errors.RasterioError, requests.RequestException) as error:
                # GDAL 异常可能含临时签名 URL，仅保留异常类型。
                raise RuntimeError(
                    f"Window read failed for {scene} {band}: {type(error).__name__}"
                ) from None
            finally:
                del signed
        with rasterio.open(output_path) as saved:
            if fixed_crs is None:
                fixed_crs, fixed_transform = saved.crs, saved.transform
            if saved.crs != fixed_crs or saved.transform != fixed_transform:
                raise ValueError("Saved bands have inconsistent grids")
            files.append(
                {
                    "path": str(output_path.relative_to(ROOT)),
                    "band": band,
                    "source_url_unsigned": original_href,
                    "sha256": sha256(output_path),
                    "size_bytes": output_path.stat().st_size,
                    "dtype": saved.dtypes[0],
                    "shape": list(saved.shape),
                    "crs": saved.crs.to_string(),
                    "transform": list(saved.transform)[:6],
                    "scale": saved.scales[0],
                    "offset": saved.offsets[0],
                    "nodata": saved.nodata,
                }
            )
        print(scene, band, output_path.stat().st_size)
    manifest = {
        "scene_id": scene,
        "collection": "landsat-c2-l2",
        "product_level": "C2_L2SP",
        "reference": str((REFERENCE / reference_name).relative_to(ROOT)),
        "reference_sha256": sha256(REFERENCE / reference_name),
        "reference_definition": "WorldView viewable fractional snow, independent of Landsat SMA",
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_utc": item["properties"]["datetime"],
        "license": "USGS Landsat public domain / unrestricted use",
        "files": files,
        "window_definition": "Native Landsat pixels covering full projected WorldView reference extent",
        "grants_persisted": False,
        "qa_not_scaled": True,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main() -> None:
    for scene, reference in PAIRS.items():
        acquire_scene(scene, reference)


if __name__ == "__main__":
    main()
