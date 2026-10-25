"""检查选定 v2 实验输出及其来源元数据。"""

from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT, paired_metrics, write_json

BASE = ROOT / "experiments/painter1998/2026-10-01/v2"
STAGES = [
    "10_publication_figures",
    "11_empirical_spectra",
    "12_aviris_band_sensitivity",
    "13_worldview_validation",
    "14_empirical_mixtures",
    "15_review_reanalysis",
    "16_conservative_maps",
]


def latest_complete(stage: str) -> Path:
    """选择最近完成的版本，保留其他运行记录。"""
    runs = [
        path
        for path in sorted(BASE.glob(stage + "_r*"))
        if json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))["status"]
        == "complete"
    ]
    if not runs:
        raise ValueError(f"No completed run: {stage}")
    return runs[-1]


def digest(path: Path) -> str:
    """以分块读取计算文件哈希。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run() -> Path:
    results = {
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_runs": {},
        "source_files_checked": 0,
        "rasters_checked": 0,
        "pngs_checked": 0,
        "source_snapshots_checked": 0,
        "historical_noncomplete_runs": [],
        "errors": [],
    }
    # 检查本地来源清单及文件哈希。
    for folder in [ROOT / "data/raw/public_validation", ROOT / "data/cache/public_validation"]:
        for path in folder.glob("*/source_manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for entry in manifest["files"]:
                assert digest(ROOT / entry["path"]) == entry["sha256"], entry["path"]
                results["source_files_checked"] += 1
    for manifest_path in BASE.glob("*/run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            results["historical_noncomplete_runs"].append(
                {
                    "run": manifest_path.parent.name,
                    "status": manifest["status"],
                    "failure": manifest.get("failure"),
                }
            )
    for stage in STAGES:
        path = latest_complete(stage)
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        results["selected_runs"][stage] = str(path.relative_to(ROOT))
        if (path / "source_snapshot.zip").exists():
            with zipfile.ZipFile(path / "source_snapshot.zip") as archive:
                for relative, expected in manifest["code_sha256"].items():
                    assert (
                        hashlib.sha256(archive.read(relative.replace("\\", "/"))).hexdigest()
                        == expected
                    )
            results["source_snapshots_checked"] += 1
        if stage == "10_publication_figures":
            assert json.loads((path / "verification.json").read_text(encoding="utf-8"))[
                "all_passed"
            ]
            for source, expected in manifest["source_sha256"].items():
                assert digest(ROOT / source) == expected
        for figure in (path / "figures").glob("*.png"):
            with Image.open(figure) as image:
                assert min(image.size) > 900
                assert min(image.info.get("dpi", (0, 0))) >= 299, figure
            assert figure.with_suffix(".svg").exists()
            results["pngs_checked"] += 1
        for raster in (path / "rasters").glob("*.tif"):
            with rasterio.open(raster) as dataset:
                assert dataset.crs is not None and all(dataset.descriptions)
                array = dataset.read()
                if "fcls_spectral_fractions" in raster.name:
                    good = np.isfinite(array).all(axis=0)
                    assert np.min(array[:, good]) >= -1e-6
                    np.testing.assert_allclose(array[:, good].sum(axis=0), 1, atol=2e-6)
            results["rasters_checked"] += 1
    worldview = latest_complete("13_worldview_validation")
    summary = json.loads((worldview / "metrics.json").read_text(encoding="utf-8"))
    for scene in summary["scenes"]:
        table = pd.read_csv(worldview / "tables" / (scene["id"] + "_sample_pixels.csv"))
        assert len(table) == scene["sampled_pixels"] == 12000
        assert (
            int(table.common_four_methods_normalization_valid.sum())
            == scene["common_four_methods_normalized_pixels"]
        )
        arrays = np.load(worldview / "arrays" / (scene["id"] + "_sample_solutions.npz"))
        for record in scene["methods"]:
            keep = table.common_four_methods_normalization_valid.to_numpy(copy=True)
            if record["stratum"] != "all":
                keep &= table.stratum.to_numpy() == record["stratum"]
            estimate = table[record["method"] + "_snow"].to_numpy()
            actual = paired_metrics(table.reference_fsca.to_numpy()[keep], estimate[keep])
            expected = record["common_methods_cover"]
            assert actual["n"] == expected["n"]
            if actual["n"]:
                np.testing.assert_allclose(
                    [actual["rmse"], actual["bias"]],
                    [expected["rmse"], expected["bias"]],
                    atol=1e-7,
                )
        for method in [
            "fixed200_six_bands",
            "multi_six_bands",
            "fixed200_four_bands",
            "multi_four_bands",
        ]:
            f = arrays[method + "_fractions"]
            shade = arrays[method + "_shade"]
            assert np.nanmin(f) >= -1e-7
            np.testing.assert_allclose(f.sum(axis=1) + shade, 1, atol=2e-6)
            np.testing.assert_allclose(
                np.sqrt(np.mean(arrays[method + "_residual"] ** 2, axis=1)),
                arrays[method + "_rms"],
                atol=1e-7,
            )
            normal = arrays[method + "_normalized"]
            good = np.isfinite(normal).all(axis=1)
            np.testing.assert_allclose(normal[good].sum(axis=1), 1, atol=2e-6)
            assert np.isnan(normal[1 - shade <= 0.05]).all()
    mixtures = latest_complete("14_empirical_mixtures")
    arrays = np.load(mixtures / "arrays/empirical_mixture_inputs_and_residuals.npz")
    for method in ["fixed200_original", "multi_original", "multi_fcls"]:
        np.testing.assert_allclose(
            arrays["controlled_observation"] - arrays[method + "_modeled_zero_noise"],
            arrays[method + "_Ec_zero_noise"],
            atol=1e-12,
        )
    metrics = json.loads((mixtures / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["base_cases"] == 440 and metrics["result_rows"] == 79200
    paper = ROOT / "research/papers/Painter_1998_grain_size_AVIRIS.pdf"
    if paper.exists():
        results["reference_paper_sha256"] = digest(paper)
    results["worldview_metrics_and_SMA_invariants"] = "passed"
    results["empirical_mixture_residual_identity"] = "passed"
    number = 1
    while (ROOT / f"reports/verification_2026-10-02_v2_r{number:03d}.json").exists():
        number += 1
    output = ROOT / f"reports/verification_2026-10-02_v2_r{number:03d}.json"
    write_json(output, results)
    write_json(
        ROOT / "experiments/supplement_verification_latest.json",
        {"verification": str(output.relative_to(ROOT))},
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    run()
