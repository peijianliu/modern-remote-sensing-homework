"""核验实验产物的数值约束、栅格属性和输入文件哈希。"""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import rasterio
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT, write_json

base = ROOT / "experiments/painter1998/2026-10-01/v1"
results = {"runs": [], "errors": []}
for manifest_path in sorted(base.glob("*/run_manifest.json")):
    run = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        results["errors"].append(f"Incomplete run: {run.name}")
        continue
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    raster_count = 0
    for raster in (run / "rasters").glob("*.tif"):
        with rasterio.open(raster) as dataset:
            assert dataset.crs is not None and dataset.count > 0
            assert all(dataset.descriptions)
            array = dataset.read()
            if "fcls_spectral_fractions" in raster.name or (
                manifest["stage"] == "08_qa_sensitivity"
                and raster.name.endswith("_spectral_fractions.tif")
            ):
                good = np.isfinite(array).all(0)
                assert np.min(array[:, good]) >= -1e-6
                np.testing.assert_allclose(array[:, good].sum(0), 1, atol=2e-6)
            if "_grain_rms_condition" in raster.name:
                rms = array[1]
                assert np.nanmin(rms) >= 0
            if "_qa.tif" in raster.name:
                assert dataset.dtypes == ("uint16",)
        raster_count += 1
    if "synthetic_shape" in metrics:
        assert metrics["synthetic_shape"] == [146, 101, 224]
        assert metrics["matched_200um_max_rms"] < 1e-8
        assert metrics["minimum_max_rms"] < 1e-8
    if "coarse_shape" in metrics:
        table = pd.read_csv(run / "tables/coarse_pixel_comparison.csv")
        assert len(table) == np.prod(metrics["coarse_shape"])
        assert int(table.paired_valid.sum()) == metrics["matched_pixels"]
        assert table.aviris_valid_area_fraction.max() <= 1 + 1e-6
    results["runs"].append(
        {
            "run": run.name,
            "status": "complete",
            "rasters_checked": raster_count,
            "png_count": len(list((run / "figures").glob("*.png"))),
        }
    )
forward = loadmat(ROOT / "data/cache/forward_models.mat")
assert np.isfinite(forward["snicar_albedo"]).all()
assert forward["depth_differences"][:, 3].max() < 1e-4
results["physical_arrays_checked"] = True
for name, path in [
    ("original_pdf", ROOT / "research/papers/Painter_1998_grain_size_AVIRIS.pdf"),
    ("MODIS", ROOT / "data/raw/MOD09GA.A2013123.h08v05.061.2021229143331.hdf"),
]:
    # 论文以 DOI 引用；如保存本地副本，则额外记录文件哈希。
    if name == "original_pdf" and not path.is_file():
        continue
    with path.open("rb") as stream:
        results[name + "_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
write_json(ROOT / "experiments/artifact_verification.json", results)
print(json.dumps(results, ensure_ascii=False, indent=2))
if results["errors"]:
    raise SystemExit(1)
