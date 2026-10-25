"""下载指定版本的 Zenodo 积雪光谱数据。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "public_validation"
RECORDS = {
    "4031446": {
        "valKey.csv",
        "15JAN07193448_landsat_30m_fsca.tif",
        "15MAR05191121_landsat_30m_fsca.tif",
        "15DEC27191740_landsat_30m_fsca.tif",
        "16FEB13190431_landsat_30m_fsca.tif",
        "17APR03193802_landsat_30m_fsca.tif",
        "17MAY21191849_landsat_30m_fsca.tif",
        "17JUN27190804_landsat_30m_fsca.tif",
        "17DEC11190506_landsat_30m_fsca.tif",
        "19MAR18191853_landsat_30m_fsca.tif",
        "16FEB13190431_modis_463.3127m_fsca.tif",
        "17MAY21191849_modis_463.3127m_fsca.tif",
    },
    "3580825": {"reflectance_spectra_field_2010_2011.zip"},
    "3580078": {
        "Darklab_snow_reflectance_avg.csv",
        "Darklab_snow_reflectance_std.csv",
        "Darklab_snow_pit_measurements.csv",
        "Darklab_snow_SSA.zip",
        "Darklab_snow_samplewise_reflectance_avg.csv",
        "Darklab_snow_samplewise_reflectance_std.csv",
        "Darklab_pine_and_spruce_reflectance_all.csv",
        "Darklab_pine_and_spruce_reflectance_all_std.csv",
    },
}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for record, names in RECORDS.items():
        folder = DATA / ("zenodo_" + record)
        folder.mkdir(exist_ok=True)
        response = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
        response.raise_for_status()
        metadata = response.json()
        metadata_path = folder / "zenodo_record.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        entries = []
        for entry in metadata["files"]:
            if entry["key"] not in names:
                continue
            target = folder / entry["key"]
            url = entry["links"]["self"]
            if not target.exists():
                blob = requests.get(url, timeout=90)
                blob.raise_for_status()
                if len(blob.content) != entry["size"]:
                    raise ValueError(f"Wrong byte count: {target}")
                md5 = hashlib.md5(blob.content).hexdigest()
                if "md5:" + md5 != entry["checksum"]:
                    raise ValueError(f"Checksum mismatch: {target}")
                with target.open("xb") as stream:
                    stream.write(blob.content)
            content = target.read_bytes()
            actual_md5 = "md5:" + hashlib.md5(content).hexdigest()
            if actual_md5 != entry["checksum"]:
                raise ValueError(f"Existing file does not match source: {target}")
            entries.append(
                {
                    "path": str(target.relative_to(ROOT)),
                    "url": url,
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "source_md5": entry["checksum"],
                }
            )
            print(entry["key"], len(content))
        manifest_path = folder / "source_manifest.json"
        manifest = {
            "record_id": record,
            "doi": metadata.get("doi"),
            "title": metadata["metadata"]["title"],
            "license": metadata["metadata"].get("license"),
            "acquired_utc": datetime.now(timezone.utc).isoformat(),
            "anonymous_get": True,
            "files": entries,
        }
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    acquire_oli_response()


def acquire_oli_response() -> None:
    """保存 USGS 官方 OLI 光谱响应文件及来源信息。"""
    folder = DATA / "USGS_L8_OLI"
    folder.mkdir(exist_ok=True)
    base = "https://landsat.usgs.gov/landsat/spectral_viewer/"
    names = {"L8_OLI_RSR.xlsx": base + "bands/L8_OLI_RSR.xlsx"}
    for band in range(1, 8):
        name = f"Landsat8OLI{band}.json"
        names[name] = base + "c3-master/htdocs/data/bands/" + name
    entries = []
    for name, url in names.items():
        target = folder / name
        if not target.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if name.endswith(".json"):
                response.json()
            elif not response.content.startswith(b"PK"):
                raise ValueError("USGS workbook response is not XLSX")
            with target.open("xb") as stream:
                stream.write(response.content)
        entries.append(
            {
                "path": str(target.relative_to(ROOT)),
                "url": url,
                "size_bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    manifest_path = folder / "source_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(
                {
                    "source": "USGS Spectral Characteristics Viewer",
                    "source_page": "https://landsat.usgs.gov/spectral-characteristics-viewer",
                    "acquired_utc": datetime.now(timezone.utc).isoformat(),
                    "license": "USGS public-domain data; acknowledge USGS/NASA OLI sensor characterization",
                    "json_wavelength_unit": "um",
                    "response_dimensionless": True,
                    "files": entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
