"""解压指定 AVIRIS L2 数据，检查路径并保存影像元数据。"""

import json
from pathlib import Path
import tarfile
import rasterio
from spectral.io import envi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / "data/cache/aviris"
destination.mkdir(parents=True, exist_ok=True)
metadata_path = ROOT / "data/cache/aviris_metadata.json"
if not list(destination.rglob("*corr.hdr")):
    with tarfile.open(ROOT / "data/raw/f130503t01p00r14_refl.tar.gz", "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member")
            if target.exists():
                if target.stat().st_size != member.size:
                    raise ValueError("Existing extracted source differs; preserved")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as handle:
                while chunk := source.read(2**22):
                    handle.write(chunk)
            print(f"Extracted {target.name}: {member.size} bytes", flush=True)
headers = list(destination.rglob("*corr.hdr"))
if len(headers) != 1:
    raise ValueError(f"Expected one reflectance header: {headers}")
header = headers[0]
binary = header.with_suffix("")
metadata = envi.read_envi_header(str(header))
unit = metadata.get("wavelength units", "")
wavelength = np.array(metadata["wavelength"], dtype=float)
inferred = not bool(unit)
factor = 0.001 if "nano" in unit.lower() or (not unit and np.median(wavelength) > 100) else 1
wavelength *= factor
fwhm = (
    np.array(metadata.get("fwhm", [10 if factor == 0.001 else 0.01] * len(wavelength)), dtype=float)
    * factor
)
with rasterio.open(binary) as image:
    record = {
        "header_path": str(header.relative_to(ROOT)),
        "binary_path": str(binary.relative_to(ROOT)),
        "wavelength_um": wavelength.tolist(),
        "fwhm_um": fwhm.tolist(),
        "wavelength_unit_source": unit,
        "unit_inferred": inferred,
        "shape": [image.height, image.width, image.count],
        "crs": image.crs.to_wkt(),
        "transform": list(image.transform)[:6],
        "bounds": list(image.bounds),
        "nodata": image.nodata,
        "scale_factor": 10000,
        "scale_source": "JPL AVIRIS Download README L2 reflectance units",
        "header_metadata": metadata,
    }
metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k not in ["wavelength_um", "fwhm_um", "header_metadata", "crs"]
        },
        ensure_ascii=False,
    ),
    flush=True,
)
