"""保持物理端元字典一致，分别评价光谱与空间分辨率退化。"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT, new_run, finish_run, configure_plots, paired_metrics
from snowmix.image import prepare_roi, sensor_library, unmix_pixels, save_solution, map_panel
from snowmix.geo import modis_roi_grid, MODIS_CRS, area_weights, aggregate
from snowmix.spectra import modis_responses, response_matrix

c = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
p = new_run(c, "09_resolution_control")
configure_plots()
m = json.loads((ROOT / "data/cache/aviris_metadata.json").read_text(encoding="utf-8"))
cube, valid, mask, t, shape, crs, bounds = prepare_roi(c, m)
w = np.array(m["wavelength_um"])
fw = np.array(m["fwhm_um"])
r, s, b, records = sensor_library(w[mask], fw[mask], 22, c)
b = b[:, :3]
high = cube[..., mask].reshape(-1, mask.sum()).astype(float)
high[~valid.ravel()] = np.nan
rw, rs = modis_responses(ROOT / "data/raw/Terra_RSR_in-band.xlsx")
unique, idx = np.unique(w, return_index=True)
ru, coverage = response_matrix(unique, rw, rs, mask[idx])
response = np.zeros((len(w), 7))
response[idx] = ru
low = np.nan_to_num(cube.reshape(-1, len(w))) @ response
low[~valid.ravel()] = np.nan
sl = s @ response[mask]
bl = (b.T @ response[mask]).T
fine_high = unmix_pixels(high, s, r, b, c, solver="fcls")
fine_low = unmix_pixels(low, sl, r, bl, c, solver="fcls")
ct, cs, _ = modis_roi_grid(bounds, crs)
weights = area_weights(t, shape, crs, ct, cs, MODIS_CRS)
coarse_high, cov = aggregate(weights, high, valid)
coarse_low, _ = aggregate(weights, low, valid)
full = cov >= 0.95
coarse_high[~full] = np.nan
coarse_low[~full] = np.nan
spatial = unmix_pixels(coarse_high, s, r, b, c, solver="fcls")
combined = unmix_pixels(coarse_low, sl, r, bl, c, solver="fcls")
reference, ref_cov = aggregate(weights, fine_high["normalized"][:, 0])
reference = reference[:, 0]
low_reference, low_ref_cov = aggregate(weights, fine_low["normalized"][:, 0])
low_reference = low_reference[:, 0]
reference_full = full & (ref_cov >= 0.95)
low_reference_full = full & (low_ref_cov >= 0.95)
classes = ["snow", "vegetation", "rock", "soil"]
for label, fit, sh, tr, projection in [
    ("fine_high_same4", fine_high, shape, t, crs),
    ("fine_modis_same4", fine_low, shape, t, crs),
    ("spatial_only_same4", spatial, cs, ct, MODIS_CRS),
    ("combined_modis_same4", combined, cs, ct, MODIS_CRS),
]:
    save_solution(p, label, fit, sh, tr, projection, classes)
map_panel(
    p,
    "same_dictionary_resolution_control",
    [reference, spatial["normalized"][:, 0], combined["normalized"][:, 0]],
    ["AVIRIS先解混后聚合", "仅空间退化", "光谱＋空间退化"],
    cs,
    (0, 1),
    "Blues",
)
pd.DataFrame(
    {
        "coverage": cov,
        "reference_coverage": ref_cov,
        "fine_modis_reference_coverage": low_ref_cov,
        "reference_snow": reference,
        "fine_modis_aggregated_snow": low_reference,
        "spatial_only_snow": spatial["normalized"][:, 0],
        "combined_snow": combined["normalized"][:, 0],
    }
).to_csv(p / "tables/controlled_resolution.csv", index=False)
metrics = {
    "dictionary": ["snow", "vegetation", "rock", "soil", "zero_shade"],
    "dictionary_identical_in_all_branches": True,
    "spectral_only": paired_metrics(fine_high["normalized"][:, 0], fine_low["normalized"][:, 0]),
    "spatial_only": paired_metrics(
        reference[reference_full], spatial["normalized"][reference_full, 0]
    ),
    "combined": paired_metrics(
        reference[reference_full], combined["normalized"][reference_full, 0]
    ),
    "same_bands_noncommutation": paired_metrics(
        low_reference[low_reference_full], combined["normalized"][low_reference_full, 0]
    ),
    "spectral_endmember_response_operator_identical_to_data": True,
    "reference_is_model_estimate_not_ground_truth": True,
}
(p / "README.md").write_text(
    "# 固定字典的光谱/空间退化对照\n\n所有分支均采用雪＋植被＋岩石＋土壤＋零阴影、相同粒径库和FCLS。端元与影像使用同一线性RSR算子。分别改变波段或空间聚合，固定字典用于分离光谱与空间分辨率影响，参照为AVIRIS模型结果。\n",
    encoding="utf-8",
)
finish_run(p, metrics)
