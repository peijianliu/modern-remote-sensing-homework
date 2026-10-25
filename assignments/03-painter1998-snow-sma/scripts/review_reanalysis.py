"""从已保存的 v1 结果重算共同有效面积掩膜及配对统计。"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT, new_run, finish_run, paired_metrics
from snowmix.geo import aggregate, area_weights, MODIS_CRS
from affine import Affine


def run() -> Path:
    config = json.loads((ROOT / "configs/supplement_v2.json").read_text(encoding="utf-8"))
    path = new_run(config, "15_review_reanalysis")
    base = ROOT / "experiments/painter1998/2026-10-01/v1"
    results = {}
    minimum = 0.95
    for name in ("05_07_image_r001", "05_07_image_r002"):
        source = base / name
        weights = load_npz(source / "arrays/area_weights.npz")
        table = pd.read_csv(source / "tables/coarse_pixel_comparison.csv")
        fit = np.load(source / "arrays/aviris_multi_fcls.npz")
        low = np.load(source / "arrays/spectral_only_modis_fcls.npz")
        reference, ref_cov = aggregate(weights, fit["normalized"][:, 0])
        low_reference, low_cov = aggregate(weights, low["normalized"][:, 0])
        full = table.aviris_valid_area_fraction.to_numpy() >= minimum
        ref_valid = full & (ref_cov >= minimum)
        low_valid = full & (low_cov >= minimum)
        snow = fit["normalized"][:, 0]
        grain = fit["radius_um"]
        good = np.isfinite(grain) & np.isfinite(snow)
        total = weights @ np.nan_to_num(snow, nan=0)
        grain_area = weights @ np.where(good, snow, 0)
        support = np.divide(grain_area, total, out=np.full(len(total), np.nan), where=total > 0)
        numerator = weights @ np.where(good, snow * grain, 0)
        radius = np.divide(
            numerator,
            grain_area,
            out=np.full(len(total), np.nan),
            where=ref_valid & (support >= minimum) & (reference[:, 0] >= 0.1) & (grain_area > 0),
        )
        table["reference_valid_area_fraction"] = ref_cov
        table["spectral_reference_valid_area_fraction"] = low_cov
        table["reference_coarse_mask_corrected"] = ref_valid
        table["grain_valid_snow_area_fraction"] = support
        table["reference_radius_corrected_um"] = radius
        table.to_csv(path / f"tables/{name}_corrected_coverage.csv", index=False)
        results[name] = {
            "spatial_only_before": paired_metrics(
                reference[full, 0], table.loc[full, "spatial_only_snow"]
            ),
            "spatial_only_corrected": paired_metrics(
                reference[ref_valid, 0], table.loc[ref_valid, "spatial_only_snow"]
            ),
            "combined_before": paired_metrics(
                reference[full, 0], table.loc[full, "simulated_modis_snow"]
            ),
            "combined_corrected": paired_metrics(
                reference[ref_valid, 0], table.loc[ref_valid, "simulated_modis_snow"]
            ),
            "noncommutation_corrected": paired_metrics(
                low_reference[low_valid, 0], table.loc[low_valid, "simulated_modis_snow"]
            ),
            "reference_coverage_rejected_n": int((full & ~ref_valid).sum()),
            "strict_actual_paired_reference_coverage_passes": bool(
                (ref_cov[table.paired_valid] >= minimum).all()
            ),
            "old_grain_reference_n": int(np.isfinite(table.aviris_snow_weighted_radius_um).sum()),
            "corrected_grain_reference_n": int(np.isfinite(radius).sum()),
        }
    source = base / "09_resolution_control_r001"
    spatial = json.loads(
        (base / "05_07_image_r001/spatial_metadata.json").read_text(encoding="utf-8")
    )
    shape = (267, 267)
    table = pd.read_csv(source / "tables/controlled_resolution.csv")
    coarse_shape = tuple(
        json.loads((base / "05_07_image_r001/metrics.json").read_text(encoding="utf-8"))[
            "coarse_shape"
        ]
    )
    weights = area_weights(
        Affine(*spatial["aviris_transform"]),
        shape,
        spatial["aviris_crs"],
        Affine(*spatial["modis_transform"]),
        coarse_shape,
        MODIS_CRS,
    )
    high = np.load(source / "arrays/fine_high_same4.npz")["normalized"][:, 0]
    low = np.load(source / "arrays/fine_modis_same4.npz")["normalized"][:, 0]
    reference, ref_cov = aggregate(weights, high)
    low_reference, low_cov = aggregate(weights, low)
    full = table.coverage.to_numpy() >= minimum
    ref_valid, low_valid = full & (ref_cov >= minimum), full & (low_cov >= minimum)
    table["reference_valid_area_fraction"] = ref_cov
    table["spectral_reference_valid_area_fraction"] = low_cov
    table.to_csv(path / "tables/same_dictionary_control_corrected.csv", index=False)
    results["09_resolution_control_r001"] = {
        "spatial_only_corrected": paired_metrics(
            reference[ref_valid, 0], table.loc[ref_valid, "spatial_only_snow"]
        ),
        "combined_corrected": paired_metrics(
            reference[ref_valid, 0], table.loc[ref_valid, "combined_snow"]
        ),
        "same_bands_noncommutation_corrected": paired_metrics(
            low_reference[low_valid, 0], table.loc[low_valid, "combined_snow"]
        ),
        "rejected_reference_coverage_n": int((full & ~ref_valid).sum()),
    }
    results["reconstructed_from_v1_without_refitting"] = True
    (path / "README.md").write_text(
        "# 共同有效面积条件与统计复算\n\n读取原v1保存的解混结果和像元相交权重，重新限制输入与阴影归一化参照都至少95%有效面积；粒径加权另要求95%已估计雪面积具有粒径值。统计直接由保存结果复算。严格真实MODIS配对原本已经应用参照有效面积条件，逐像元配对结果保持一致。\n\n输入参照为AVIRIS模型。旧/新统计同表保留，仅少量边缘或暗区参照像元改变。\n",
        encoding="utf-8",
    )
    finish_run(path, results)
    return path


if __name__ == "__main__":
    run()
