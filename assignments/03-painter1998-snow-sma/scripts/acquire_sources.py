"""下载指定公开资料，支持大文件断点续传并记录来源。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "optics": (
        "http://snow.engin.umich.edu/opticalprops/snicar_v3_optics_202105/snicar_v3_optics.tar.gz",
        "data/raw/snicar_v3_optics.tar.gz",
        502638762,
    ),
    "aviris": (
        "https://popo.jpl.nasa.gov/avcl/y13_data/f130503t01p00r14_refl.tar.gz",
        "data/raw/f130503t01p00r14_refl.tar.gz",
        2745422132,
    ),
    "snicar": (
        "https://raw.githubusercontent.com/mflanner/SNICARv3/227e9de9b100ed0671662f94d2040fd8231aab91/snicarAD_v3.m",
        "vendor/SNICARv3/snicarAD_v3.m",
        None,
    ),
    "snicar_driver": (
        "https://raw.githubusercontent.com/mflanner/SNICARv3/227e9de9b100ed0671662f94d2040fd8231aab91/snicarAD_v3_drv.m",
        "vendor/SNICARv3/snicarAD_v3_drv.m",
        None,
    ),
    "snicar_license": (
        "https://raw.githubusercontent.com/mflanner/SNICARv3/227e9de9b100ed0671662f94d2040fd8231aab91/LICENSE",
        "vendor/SNICARv3/LICENSE",
        None,
    ),
    "twostream": (
        "https://raw.githubusercontent.com/edwardbair/SPIRES/bcca68ebad36a34bf67f8bc88b7198eda9fc608f/SolarLongwaveRadiation/twostream.m",
        "vendor/SPIRES/twostream.m",
        None,
    ),
    "check_sizes": (
        "https://raw.githubusercontent.com/edwardbair/SPIRES/bcca68ebad36a34bf67f8bc88b7198eda9fc608f/General/checkSizes.m",
        "vendor/SPIRES/checkSizes.m",
        None,
    ),
    "spires_license": (
        "https://raw.githubusercontent.com/edwardbair/SPIRES/bcca68ebad36a34bf67f8bc88b7198eda9fc608f/LICENSE",
        "vendor/SPIRES/LICENSE",
        None,
    ),
    "aviris_readme": (
        "https://aviris.jpl.nasa.gov/dataportal/20170911_AV_Download.readme",
        "data/raw/AVIRIS_download.readme",
        None,
    ),
    "quartz": (
        "https://pubs.usgs.gov/of/2003/ofr-03-395/ASCII/M/quartz_hs32.5854.asc",
        "data/raw/spectra/quartz_hs32.5854.asc",
        None,
    ),
    "granite": (
        "https://speclib.jpl.nasa.gov/ecospeclibdata/rock.igneous.felsic.solid.all.granite_h2.jhu.becknic.spectrum.txt",
        "data/raw/spectra/granite_h2.txt",
        None,
    ),
    "soil": (
        "https://speclib.jpl.nasa.gov/ecospeclibdata/soil.inceptisol.cryumbrept.none.all.87p3855.jhu.becknic.spectrum.txt",
        "data/raw/spectra/cryumbrept_87p3855.txt",
        None,
    ),
    "pine": (
        "https://pubs.usgs.gov/of/2003/ofr-03-395/ASCII/V/lodgepole-pine_lp-needles-1.11574.asc",
        "data/raw/spectra/lodgepole_pine.asc",
        None,
    ),
    "modis_rsr": (
        "https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/Terra_RSR_in-band.xlsx",
        "data/raw/Terra_RSR_in-band.xlsx",
        None,
    ),
}
EXCLUDED_SOURCES = {"granite", "soil"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(name: str) -> dict:
    if name in EXCLUDED_SOURCES:
        raise PermissionError(
            "Historical raw spectrum excluded by PUBLIC_DATA_POLICY.md; use the public USGS or image-derived inputs"
        )
    url, relative, expected = SOURCES[name]
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if path.exists():
        if expected is not None and path.stat().st_size != expected:
            raise ValueError(f"Existing source has unexpected size; preserved: {path}")
    else:
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            if expected is not None and offset == expected:
                partial.rename(path)
                break
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with requests.get(url, stream=True, timeout=(20, 90), headers=headers) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        raise RuntimeError("Server did not honor resume; partial file preserved")
                    if offset and not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    ):
                        raise RuntimeError("Wrong Content-Range; partial file preserved")
                    last_report = time.monotonic()
                    with partial.open("ab") as handle:
                        for chunk in response.iter_content(2**20):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last_report > 20:
                                print(
                                    f"{name}: {offset / 1e6:.1f} MB"
                                    + (f" / {expected / 1e6:.1f}" if expected else ""),
                                    flush=True,
                                )
                                last_report = time.monotonic()
                if expected is not None and partial.stat().st_size != expected:
                    raise RuntimeError(f"Incomplete {name}: {partial.stat().st_size}/{expected}")
                partial.rename(path)
                break
            except requests.RequestException as exc:
                print(f"{name}: retry {attempt + 1}: {type(exc).__name__}", flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 20))
    result = {
        "name": name,
        "url": url,
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest = ROOT / "data" / "sources" / f"{name}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"{name}: complete {result['bytes']} bytes SHA256={result['sha256']}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    approved = [name for name in SOURCES if name not in EXCLUDED_SOURCES]
    parser.add_argument("names", nargs="*", choices=approved, default=approved)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(download, args.names):
            pass
