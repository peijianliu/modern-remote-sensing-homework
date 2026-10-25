"""以公开实验室和野外实测光谱评价预定积雪混合模型。"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import (
    ROOT,
    new_run,
    finish_run,
    configure_plots,
    save_figure,
    paired_metrics,
    write_json,
)
from snowmix.experiments import sensor_grid
from snowmix.linear import select_snow
from snowmix.spectra import gaussian_band_matrix, read_usgs_ascii, read_ecostress
from snowmix.validation import interval_mask, read_lab_spectra, read_ssa_archive, read_field_archive
import matplotlib.pyplot as plt


def observed_sensor_spectra(
    w: np.ndarray, y: np.ndarray, centers: np.ndarray, fwhm: np.ndarray
) -> np.ndarray:
    """使用与模型端元相同的高斯仪器响应卷积实测光谱。"""
    return y @ gaussian_band_matrix(w, centers, fwhm)


def snow_library(
    cache: str, centers: np.ndarray, fwhm: np.ndarray, angle: float, optical_option: int
) -> tuple[np.ndarray, np.ndarray]:
    """读取与指定角度及冰光学选项匹配的正演缓存。"""
    data = loadmat(ROOT / "data/cache" / cache)
    index = np.flatnonzero(np.isclose(data["angles_deg"].ravel(), angle))
    if len(index) != 1:
        raise ValueError("Requested model zenith angle is absent")
    optical_index = np.flatnonzero(data["ice_options"].ravel() == optical_option)
    if len(optical_index) != 1:
        raise ValueError("Requested ice optical option is absent")
    weights = gaussian_band_matrix(data["wavelength_um"].ravel(), centers, fwhm)
    return data["radii_um"].ravel(), data["snicar_albedo"][
        :, :, index[0], int(optical_index[0])
    ] @ weights


def run() -> Path:
    config = json.loads((ROOT / "configs/supplement_v2.json").read_text(encoding="utf-8"))
    cfg = config["empirical"]
    path = new_run(config, "11_empirical_spectra")
    configure_plots()
    plt.rcParams["savefig.dpi"] = 300
    baseline = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
    centers, widths, _ = sensor_grid(baseline)
    keep = interval_mask(centers, cfg["usable_intervals_um"])
    cw, fw = centers[keep], widths[keep]
    background_spectra, background_metadata = [], []
    # 实测谱、雪谱与背景谱采用相同仪器响应。
    for filename, reader in [
        ("lodgepole_pine.asc", read_usgs_ascii),
        ("granite_h2.txt", read_ecostress),
        ("cryumbrept_87p3855.txt", read_ecostress),
    ]:
        bw, br, record = reader(ROOT / "data/raw/spectra" / filename)
        background_spectra.append(br @ gaussian_band_matrix(bw, cw, fw))
        background_metadata.append({"file": filename, "metadata": record})
    background = np.column_stack(background_spectra)
    lab = ROOT / "data/raw/public_validation/zenodo_3580078"
    meta, w, spectra = read_lab_spectra(lab / "Darklab_snow_reflectance_avg.csv")
    ssa = read_ssa_archive(lab / "Darklab_snow_SSA.zip", cfg["ice_density_kg_m3"])
    meta = meta.merge(ssa, on="name", how="left", validate="one_to_one")
    # M_rimed14 在雪况均值表中标为干雪，在逐样本元数据中标为湿雪。
    meta["dry_unambiguous"] = (
        (meta.snow_type == 1) & (meta.litter == 1) & (meta["name"] != "M_rimed14")
    )
    y = observed_sensor_spectra(w, spectra, cw, fw)
    radii, snow = snow_library(
        "forward_models_55deg.mat", cw, fw, cfg["lab_zenith_deg"], cfg["ice_option"]
    )
    fixed = np.flatnonzero(radii == 200)
    modes = {
        "fixed200_original": (snow[fixed], radii[fixed], "original"),
        "multi_original": (snow, radii, "original"),
        "multi_fcls": (snow, radii, "fcls"),
    }
    rows, spectra_rows, profiles = [], [], []
    fits = {}
    with threadpool_limits(limits=1):
        for label, (dictionary, candidate_radii, solver) in modes.items():
            fit = select_snow(y, dictionary, candidate_radii, background, solver=solver)
            fits[label] = fit
            for i, record in meta.iterrows():
                rows.append(
                    {
                        **record.to_dict(),
                        "method": label,
                        "snow_fraction": fit.normalized[i, 0],
                        "radius_um": fit.radius_um[i],
                        "rms": fit.rms[i],
                        "qa": int(fit.qa[i]),
                        "shade": fit.shade[i],
                        "radius_library_min_um": radii[0],
                        "radius_library_max_um": radii[-1],
                    }
                )
                for wave, obs, modeled, residual in zip(cw, y[i], fit.modeled[i], fit.residual[i]):
                    spectra_rows.append(
                        {
                            "name": record["name"],
                            "method": label,
                            "wavelength_um": wave,
                            "observed": obs,
                            "modeled": modeled,
                            "E_c": residual,
                        }
                    )
        # 同时拟合纯雪谱形与独立灰度增益。
        shape_keep = interval_mask(cw, [cfg["nir_shape_interval_um"]])
        for i, record in meta.iterrows():
            observation = y[i, shape_keep]
            library = snow[:, shape_keep]
            gain = library @ observation / np.sum(library * library, axis=1)
            rms = np.sqrt(np.mean((observation[None, :] - gain[:, None] * library) ** 2, axis=1))
            selected = int(np.argmin(rms))
            near = radii[rms <= rms[selected] + 0.005]  # 光谱 RMS 诊断容差
            meta.loc[i, "nir_shape_radius_um"] = radii[selected]
            meta.loc[i, "nir_shape_gain"] = gain[selected]
            meta.loc[i, "nir_shape_rms"] = rms[selected]
            meta.loc[i, "near_minimum_radius_low_um"] = near.min()
            meta.loc[i, "near_minimum_radius_high_um"] = near.max()
            for radius, cost in zip(radii, rms):
                profiles.append(
                    {"name": record["name"], "radius_um": radius, "nir_shape_rms": cost}
                )
    table = pd.DataFrame(rows)
    table.to_csv(path / "tables/lab_model_results.csv", index=False)
    pd.DataFrame(spectra_rows).to_csv(path / "tables/lab_spectra_and_Ec.csv", index=False)
    meta.to_csv(path / "tables/lab_condition_ssa_and_shape.csv", index=False)
    pd.DataFrame(profiles).to_csv(path / "tables/lab_candidate_rms_profiles.csv", index=False)
    metrics = {
        "lab_condition_count": len(meta),
        "lab_independent_ssa_conditions": len(ssa),
        "lab_replications_are_not_independent_conditions": True,
        "ssa_is_IceCube_model_based_measurement": True,
        "DHR_vs_nadir_BRF_mismatch_retained": True,
        "band_count": len(cw),
        "SSA_formula": "radius_um=3e6/(917*SSA_m2_kg)",
        "lab_methods": {},
    }
    for label, group in table.groupby("method", sort=False):
        dry = group.dry_unambiguous
        paired = dry & group.surface_reference_radius_um.notna()
        metrics["lab_methods"][label] = {
            "clean_dry_cover_vs_known_full_target": paired_metrics(
                np.ones(dry.sum()), group.loc[dry, "snow_fraction"]
            ),
            "dry_condition_SSA_radius_discrepancy": paired_metrics(
                group.loc[paired, "surface_reference_radius_um"], group.loc[paired, "radius_um"]
            ),
            "spectral_rms_mean": float(group.rms.mean()),
            "qa_overflow_count": int(((group.qa & 64) != 0).sum()),
        }
    metrics["metadata_disagreement_excluded_from_dry_primary"] = ["M_rimed14"]
    dry_ssa = meta.dry_unambiguous & meta.surface_reference_radius_um.notna()
    metrics["lab_nir_shape_vs_SSA"] = paired_metrics(
        meta.loc[dry_ssa, "surface_reference_radius_um"], meta.loc[dry_ssa, "nir_shape_radius_um"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for label in ("fixed200_original", "multi_original", "multi_fcls"):
        group = table[table.method == label]
        axes[0].plot(np.arange(len(group)), group.snow_fraction, "o-", label=label, ms=4)
    axes[0].axhline(1, color="black", lw=0.9)
    axes[0].set(
        xticks=range(len(meta)),
        xticklabels=meta["name"],
        ylabel="阴影归一化雪比例",
        title="真实实验室雪样：模型覆盖偏差",
    )
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].legend(fontsize=8)
    plot_number = 0
    for condition in meta.itertuples():
        if np.isfinite(condition.surface_reference_radius_um):
            plot_number += 1
            axes[1].scatter(
                condition.surface_reference_radius_um,
                condition.nir_shape_radius_um,
                marker="o" if condition.snow_type == 1 else "s",
                color="#0072B2" if condition.snow_type == 1 else "#D55E00",
            )
            axes[1].annotate(
                str(plot_number),
                (condition.surface_reference_radius_um, condition.nir_shape_radius_um),
                fontsize=8,
                xytext=(5, 7 if plot_number % 2 else -12),
                textcoords="offset points",
            )
    axes[1].plot([50, 1500], [50, 1500], "k--", lw=1)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlim=(50, 1500),
        ylim=(50, 1500),
        xlabel="近表层 SSA 半径代理 / μm（log轴）",
        ylabel="近红外谱形拟合半径 / μm（log轴）",
        title="条件级参照：干雪○；湿雪□",
    )
    from matplotlib.ticker import ScalarFormatter

    for axis in (axes[1].xaxis, axes[1].yaxis):
        axis.set_ticks([50, 100, 200, 500, 1000, 1500])
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(plt.NullFormatter())
    fig.subplots_adjust(bottom=0.3, wspace=0.35)
    save_figure(fig, path, "independent_lab_cover_and_ssa")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    names = ["D_dendrites15", "D_sun13", "W_pure13", "W_litter13"]
    for ax, name in zip(axes.ravel(), names):
        i = int(np.flatnonzero(meta["name"].to_numpy() == name)[0])
        observed = y[i].copy()
        observed[np.r_[False, np.diff(cw) > 0.05]] = np.nan
        ax.plot(cw, observed, label="独立实测", color="black", lw=1.6)
        for label in ("fixed200_original", "multi_fcls"):
            shown = fits[label].modeled[i].copy()
            # 在剔除波段处断线。
            shown[np.r_[False, np.diff(cw) > 0.05]] = np.nan
            ax.plot(cw, shown, label=label, lw=1.1)
        ax.set(title=name, xlabel="波长 / μm", ylabel="反射因子 / 1")
        ax.legend(fontsize=7)
    save_figure(fig, path, "independent_lab_observed_and_fitted")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for record in meta.loc[dry_ssa].itertuples():
        profile = pd.DataFrame(profiles)
        subset = profile[profile.name == record.name]
        ax.plot(subset.radius_um, subset.nir_shape_rms * 100, label=record.name)
    ax.set(
        xlabel="候选半径 / μm",
        ylabel="0.8–1.3 μm RMS / 反射率百分点",
        title="候选 RMS 剖面与近最优半径范围",
    )
    ax.legend()
    save_figure(fig, path, "independent_lab_candidate_rms")
    # 野外数据来自外部观测，入射角采用配置中的假设值。
    field = (
        ROOT / "data/raw/public_validation/zenodo_3580825/reflectance_spectra_field_2010_2011.zip"
    )
    sites, field_w, field_spectra = read_field_archive(field)
    fy = observed_sensor_spectra(field_w, field_spectra, cw, fw)
    fr, fs = snow_library("forward_models.mat", cw, fw, cfg["field_assumed_zenith_deg"], 1)
    with threadpool_limits(limits=1):
        field_fit = select_snow(fy, fs, fr, background, solver="fcls")
    sites["estimated_snow_fraction"] = field_fit.normalized[:, 0]
    sites["estimated_radius_um"] = field_fit.radius_um
    sites["rms"] = field_fit.rms
    sites["qa"] = field_fit.qa
    sites["reference_snow_presence"] = sites.TargetType.isin([1, 2, 3])
    sites["predicted_snow_presence"] = (
        sites.estimated_snow_fraction >= cfg["snow_presence_threshold"]
    )
    sites["condition_group"] = (
        sites.Date.astype(str) + "_" + sites.Lat.astype(str) + "_" + sites.Lon.astype(str)
    )
    sites.to_csv(path / "tables/field_target_diagnostics.csv", index=False)
    confusion = pd.crosstab(sites.reference_snow_presence, sites.predicted_snow_presence).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    metrics["field"] = {
        "spectrum_count": len(sites),
        "date_location_groups": int(sites.condition_group.nunique()),
        "snow_targets": int(sites.reference_snow_presence.sum()),
        "non_snow_targets": int((~sites.reference_snow_presence).sum()),
        "confusion_rows_true_no_snow_snow_columns_pred_no_snow_snow": confusion.to_numpy().tolist(),
        "geometric_GrainSize_mm_not_converted_to_optical_radius": True,
        "assumed_solar_zenith_deg": cfg["field_assumed_zenith_deg"],
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for target, group in sites.groupby("TargetType"):
        ax.scatter(
            group.index,
            group.estimated_snow_fraction,
            label=f"TargetType {int(target)}; n={len(group)}",
            s=24,
            alpha=0.8,
        )
    ax.axhline(0.5, color="black", lw=0.8, ls="--", label="预先设定的存在阈值0.5")
    ax.set(xlabel="原始样本顺序", ylabel="估计雪比例", title="独立野外目标：雪、阴影雪和少量裸地")
    ax.legend(fontsize=8)
    save_figure(fig, path, "independent_field_presence_diagnostics")
    captions = [
        "# 独立实测光谱补充实验",
        "",
        "输入为作者公开、CC-BY 4.0 的实验室和野外数据；模型、波段窗口和阈值预先固定。使用10种条件均值，不将158个同条件重复雪样算作158种独立条件。",
        "",
        "![图1](figures/independent_lab_cover_and_ssa.png)",
        "",
        "图1：左图检验实验室全雪目标是否被误分到背景；附落叶和类型冲突M_rimed14条件不用于纯净干雪主统计。右图使用log轴，保留原数值以辨读聚集的干雪点；IceCube剖面最高记录点SSA按 r=3/(ρice SSA) 转为表面积/体积等效半径，作为条件级近表层代理。SSA只有7种条件，其中4种干雪；SSA用于条件级近表层对照。",
        "",
        "![图2](figures/independent_lab_observed_and_fitted.png)",
        "",
        "图2：实测光谱与固定粒径/多粒径拟合。模型为方向—半球反射率，实测为天底方向反射因子，观测几何不完全相同。拟合采用预先冻结的谱库，覆盖误差另行评价。",
        "",
        "![图3](figures/independent_lab_candidate_rms.png)",
        "",
        "图3：四种干雪的逐候选粒径RMS。仅0.8–1.3 μm并允许灰度增益；灰度增益用于谱形匹配。表中最优值+0.005反射率内的半径范围作为谱形拟合的诊断容差。",
        "",
        "![图4](figures/independent_field_presence_diagnostics.png)",
        "",
        "图4：87个野外光谱按作者TargetType对照；1=雪，2/3=树干/枝阴影雪，4=融雪后裸地，5=地衣，7=苔藓。非雪目标共5个，存在性结果按本样本集解释。人工毫米粒径与SnowPatch保留原测量定义，19 cm为仪器视场。",
        "",
        "来源：[实验室](https://doi.org/10.5281/zenodo.3580078)、[野外](https://doi.org/10.5281/zenodo.3580825)、[SSA定义](https://doi.org/10.1098/rsta.2018.0161)。输入许可、URL和SHA见对应data/raw/public_validation下manifest。IceCube也是光学模型反演，属于独立仪器参照，非完全无模型的微结构真值。",
    ]
    numbered = meta[meta.surface_reference_radius_um.notna()]
    captions.insert(
        7,
        "右图编号："
        + "；".join(f"{i}={name}" for i, name in enumerate(numbered["name"], 1))
        + "。",
    )
    (path / "README.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    write_json(
        path / "source_links.json",
        {
            "lab": "https://doi.org/10.5281/zenodo.3580078",
            "field": "https://doi.org/10.5281/zenodo.3580825",
            "background_metadata": background_metadata,
        },
    )
    finish_run(path, metrics)
    return path


if __name__ == "__main__":
    run()
