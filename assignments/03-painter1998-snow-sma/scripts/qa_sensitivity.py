"""在 Mammoth 研究区比较 QA 与波段设置，保留严格筛选基线。"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import (
    ROOT,
    new_run,
    finish_run,
    configure_plots,
    paired_metrics,
    save_figure,
)
from snowmix.image import prepare_roi, sensor_library, unmix_pixels, save_solution, map_panel
from snowmix.geo import modis_roi_grid, MODIS_CRS, area_weights, aggregate
from snowmix.spectra import modis_responses, response_matrix
from snowmix.modis import quality_mask, subset_quality_mask

config = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
path = new_run(config, "08_qa_sensitivity")
configure_plots()
meta = json.loads((ROOT / "data/cache/aviris_metadata.json").read_text(encoding="utf-8"))
cube, valid, mask, t, shape, crs, bounds = prepare_roi(config, meta)
w = np.array(meta["wavelength_um"])
fw = np.array(meta["fwhm_um"])
ct, cs, win = modis_roi_grid(bounds, crs)
weights = area_weights(t, shape, crs, ct, cs, MODIS_CRS)
rw, rs = modis_responses(ROOT / "data/raw/Terra_RSR_in-band.xlsx")
unique, indices = np.unique(w, return_index=True)
ru, _ = response_matrix(unique, rw, rs, mask[indices])
response = np.zeros((len(w), 7))
response[indices] = ru
radii, snow, library, _ = sensor_library(w[mask], fw[mask], 22)
data = cube[..., mask].reshape(-1, mask.sum()).astype(float)
data[~valid.ravel()] = np.nan
aviris = unmix_pixels(data, snow, radii, library[:, :3], config, solver="fcls")
reference, rcov = aggregate(weights, aviris["normalized"][:, 0])
reference = reference[:, 0]
broad = np.nan_to_num(cube.reshape(-1, len(w))) @ response
broad[~valid.ravel()] = np.nan
simulated, cov = aggregate(weights, broad, valid)
full = (cov >= 0.95) & (rcov >= 0.95)
d = loadmat(ROOT / "data/cache/modis_fields.mat")
r0, r1, c0, c1 = win
raw = d["band_raw"][r0:r1, c0:c1]
qc = d["qc"][r0:r1, c0:c1]
state = np.repeat(np.repeat(d["state"], 2, 0), 2, 1)[r0:r1, c0:c1]
radii25, s25, l25, _ = sensor_library(w[mask], fw[mask], 25)
wm = response[mask]
snow7 = snow @ wm
back7 = (library[:, :2].T @ wm).T
real_snow = s25 @ wm
real_back = (l25[:, :2].T @ wm).T
tiers = [
    ("strict7", list(range(7)), quality_mask(raw, qc, state)),
    ("strict5", [0, 1, 3, 5, 6], subset_quality_mask(raw, qc, state)),
    (
        "adjacent5_sensitivity",
        [0, 1, 3, 5, 6],
        subset_quality_mask(raw, qc, state, allow_adjacent=True),
    ),
]
rows = []
solutions = []
records = []
for name, bands, qmask in tiers:
    use = qmask.ravel() & full
    actual = raw.reshape(-1, 7).astype(float)[:, bands] * 0.0001
    actual[~use] = np.nan
    sim = simulated[:, bands].copy()
    sim[~use] = np.nan
    model = unmix_pixels(
        actual, real_snow[:, bands], radii25, real_back[bands], config, solver="fcls"
    )
    simulated_fit = unmix_pixels(sim, snow7[:, bands], radii, back7[bands], config, solver="fcls")
    paired = use & np.isfinite(model["normalized"][:, 0]) & np.isfinite(reference)
    record = {
        "tier": name,
        "band_ids": [b + 1 for b in bands],
        "n": int(paired.sum()),
        "aviris_reference_snow_ge_0p1": int((paired & (reference >= 0.1)).sum()),
        "real_snow_ge_0p1": int((paired & (model["normalized"][:, 0] >= 0.1)).sum()),
        "agreement": paired_metrics(reference[paired], model["normalized"][paired, 0]),
        "simulated_agreement": paired_metrics(
            reference[paired], simulated_fit["normalized"][paired, 0]
        ),
        "adjacent_cloud_retained": name == "adjacent5_sensitivity",
    }
    records.append(record)
    solutions.append(model["normalized"][:, 0])
    save_solution(path, name, model, cs, ct, MODIS_CRS, ["snow", "vegetation", "bare"])
    for i in np.flatnonzero(paired):
        rows.append(
            {
                "tier": name,
                "pixel": int(i),
                "reference_snow": reference[i],
                "real_snow": model["normalized"][i, 0],
                "simulated_snow": simulated_fit["normalized"][i, 0],
                "cloud_adjacent_flag": int((state.ravel()[i] >> 13) & 1),
            }
        )
pd.DataFrame(rows).to_csv(path / "tables/qa_tier_comparison.csv", index=False)
map_panel(
    path,
    "qa_sensitivity_snow",
    solutions,
    ["严格7波段", "严格5波段", "5波段 / 保留邻云对照"],
    cs,
    (0, 1),
    "Blues",
)
fig, ax = plt.subplots(figsize=(6, 5.5))
table = pd.DataFrame(rows)
for name, group in table.groupby("tier"):
    ax.scatter(
        group.reference_snow, group.real_snow, s=15, alpha=0.55, label=f"{name} n={len(group)}"
    )
ax.plot([0, 1], [0, 1], color="black", lw=1)
ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="AVIRIS雪覆盖参照",
    ylabel="真实MODIS雪覆盖",
    title="QA与波段策略敏感性；邻云对照单列",
)
ax.legend(fontsize=8)
save_figure(fig, path, "qa_sensitivity_scatter")
(path / "README.md").write_text(
    "# QA/波段敏感性\n\n严格7波段基线保持不变。严格5波段使用1、2、4、6、7，固定舍弃band5及blue band3，按保留波段质量位筛选并允许MODLAND less-than-ideal但逐带为best。本文件出现的band5代码8指dead/interpolated、band3代码13指校正越界，分别对应插值与校正质量状态。邻云5波段对照只放开邻云bit13，仍剔除cloud-state非clear、cloud-shadow、cirrus、internal-cloud。\n\n保留邻云的结果有潜在云污染，作为单独的邻云敏感性对照，按各自有效像元统计。\n",
    encoding="utf-8",
)
finish_run(path, {"tiers": records, "primary_baseline_unchanged": True})
