"""使用外部实测光谱的控制混合检验积雪光谱混合分析。

积雪和背景光谱取自独立观测，混合系数按线性模型设定。
评价对象为控制案例的已知混合系数。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import (
    ROOT,
    configure_plots,
    finish_run,
    new_run,
    paired_metrics,
    save_figure,
    write_json,
)
from snowmix.experiments import sensor_grid
from snowmix.linear import select_snow
from snowmix.spectra import (
    gaussian_band_matrix,
    load_terrestrial_library,
    read_ecostress,
    read_usgs_ascii,
)
from snowmix.validation import interval_mask, read_field_archive, read_lab_spectra, read_ssa_archive
from scripts.empirical_validation import observed_sensor_spectra, snow_library
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "fixed200_original": "固定200 μm / 原文约束",
    "multi_original": "多粒径 / 原文约束",
    "multi_fcls": "多粒径 / FCLS",
}
METHOD_COLORS = {
    "fixed200_original": "#D55E00",
    "multi_original": "#0072B2",
    "multi_fcls": "#009E73",
}
TARGET_LABELS = {4: "裸地", 5: "地衣", 7: "苔藓"}


def frozen_backgrounds(centers: np.ndarray, widths: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """以同一 AVIRIS 高斯响应卷积预定非雪端元光谱。"""
    directory = ROOT / "data/raw/spectra"
    _, metadata = load_terrestrial_library(centers, directory)
    columns = []
    for entry in metadata[:3]:
        path = directory / entry["file"]
        reader = read_usgs_ascii if path.suffix == ".asc" else read_ecostress
        wavelength, spectrum, _ = reader(path)
        columns.append(spectrum @ gaussian_band_matrix(wavelength, centers, widths))
    return np.column_stack(columns), metadata[:3]


def build_cases(
    lab_meta: pd.DataFrame, lab_sensor: np.ndarray, bare_meta: pd.DataFrame, bare_sensor: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """生成已知归一化雪比例和零光谱灰阴影系数的混合案例。"""
    rows, spectra = [], []
    for i, snow in lab_meta.iterrows():
        for j, bare in bare_meta.iterrows():
            group = f"{snow['name']}__{bare.ID}"
            for fraction in np.linspace(0, 1, 11):
                for shade in (0.0, 0.3):
                    spectrum = (1 - shade) * (
                        fraction * lab_sensor[i] + (1 - fraction) * bare_sensor[j]
                    )
                    spectra.append(spectrum)
                    rows.append(
                        {
                            "case_id": len(rows),
                            "condition_group": group,
                            "snow_condition": snow["name"],
                            "non_snow_target": bare.ID,
                            "non_snow_type": int(bare.TargetType),
                            "non_snow_date": int(bare.Date),
                            "non_snow_lat": float(bare.Lat),
                            "non_snow_lon": float(bare.Lon),
                            "non_snow_cloudiness": float(bare.cloudiness),
                            "true_snow_fraction": float(fraction),
                            "true_gray_shade": shade,
                            "true_raw_snow_coefficient": float(fraction * (1 - shade)),
                            "near_surface_ssa_m2_kg": float(snow.surface_ssa_m2_kg),
                            "near_surface_ssa_radius_proxy_um": float(
                                snow.surface_reference_radius_um
                            ),
                            "ssa_calibration": snow.ssa_calibration,
                            "lab_zenith_deg": 90 - float(snow.light_elevation_angle),
                            "field_geometry_unchanged": True,
                        }
                    )
    return pd.DataFrame(rows), np.asarray(spectra)


def group_statistics(frame: pd.DataFrame) -> dict:
    """汇总单组指标，并单独统计粒径估计缺失的案例。"""
    positive = frame.true_snow_fraction >= 0.1 - 1e-10
    cover = paired_metrics(
        frame.loc[positive, "true_snow_fraction"], frame.loc[positive, "estimated_snow_fraction"]
    )
    radius = paired_metrics(
        frame.loc[positive, "near_surface_ssa_radius_proxy_um"],
        frame.loc[positive, "estimated_radius_um"],
    )
    detected = frame.estimated_snow_fraction >= 0.1 - 1e-10
    valid_cover = frame.estimated_snow_fraction.notna()
    return {
        "cases": len(frame),
        "positive_cases": int(positive.sum()),
        "cover_n": cover["n"],
        "cover_bias": cover["bias"],
        "cover_mae": cover["mae"],
        "cover_rmse": cover["rmse"],
        "cover_missing_count": int((positive & ~valid_cover).sum()),
        "radius_n": radius["n"],
        "radius_bias_um": radius["bias"],
        "radius_mae_um": radius["mae"],
        "radius_rmse_um": radius["rmse"],
        "radius_missing_count": int((positive & frame.estimated_radius_um.isna()).sum()),
        "false_negative_count": int((positive & (~detected | ~valid_cover)).sum()),
        "false_positive_count": int((~positive & detected & valid_cover).sum()),
        "spectral_rms_mean": float(frame.rms.mean()),
        "original_overflow_count": int(((frame.qa.to_numpy(np.uint16) & 64) != 0).sum()),
        "radius_boundary_count": int(((frame.qa.to_numpy(np.uint16) & 16) != 0).sum()),
    }


def aggregate_group_statistics(groups: pd.DataFrame) -> pd.DataFrame:
    """在每次噪声重复中，对各雪况与背景组合等权汇总。"""
    records = []
    for (method, sigma, repeat), group in groups.groupby(
        ["method", "noise_std", "repeat"], sort=False
    ):
        records.append(
            {
                "method": method,
                "noise_std": sigma,
                "repeat": int(repeat),
                "groups": len(group),
                "cover_bias": float(group.cover_bias.mean()),
                "cover_mae": float(group.cover_mae.mean()),
                "cover_rmse": float(np.sqrt(np.mean(group.cover_rmse**2))),
                "mean_group_cover_rmse": float(group.cover_rmse.mean()),
                "false_negatives": int(group.false_negative_count.sum()),
                "false_positives": int(group.false_positive_count.sum()),
                "radius_missing_count": int(group.radius_missing_count.sum()),
                "radius_bias_um": float(group.radius_bias_um.mean()),
                "radius_rmse_um": float(np.sqrt(np.mean(group.radius_rmse_um**2))),
                "spectral_rms_mean": float(group.spectral_rms_mean.mean()),
            }
        )
    return pd.DataFrame(records)


def run() -> Path:
    config = json.loads((ROOT / "configs/supplement_v2.json").read_text(encoding="utf-8"))
    cfg = config["empirical"]
    config["empirical_mixtures"] = {
        "snow_conditions": ["D_sun13", "D_shadow13", "D_decomposed14", "D_dendrites15"],
        "snow_fractions": np.linspace(0, 1, 11).tolist(),
        "gray_shades": [0, 0.3],
        "noise_std": [0, 0.005, 0.01],
        "noise_repeats": 20,
        "zero_noise_actual_fits": 1,
        "shared_noise_across_methods": True,
        "minimum_snow_presence": 0.1,
        "fixed_snow_radius_um": 200,
        "analysis_unit": "snow_condition x non_snow_target",
        "fitting_backgrounds": "Frozen USGS pine and ECOSTRESS granite/soil; Gaussian-convolved",
        "generation_backgrounds_not_fitted": True,
        "mixture_is_assumed_linear_not_observed_patch_cover": True,
        "SSA_is_condition_level_near_surface_proxy": True,
        "geometries": "55deg lamp/nadir laboratory snow + original field illumination; retained mismatch",
    }
    path = new_run(config, "14_empirical_mixtures")
    configure_plots()
    plt.rcParams["savefig.dpi"] = 300
    started = time.perf_counter()
    baseline = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
    centers, widths, grid_source = sensor_grid(baseline)
    keep = interval_mask(centers, cfg["usable_intervals_um"])
    cw, fw = centers[keep], widths[keep]
    backgrounds, background_metadata = frozen_backgrounds(cw, fw)
    lab_path = ROOT / "data/raw/public_validation/zenodo_3580078"
    lab_meta, lab_w, lab_spectra = read_lab_spectra(lab_path / "Darklab_snow_reflectance_avg.csv")
    ssa = read_ssa_archive(lab_path / "Darklab_snow_SSA.zip", cfg["ice_density_kg_m3"])
    lab_meta = lab_meta.merge(ssa, on="name", validate="one_to_one", how="left")
    names = config["empirical_mixtures"]["snow_conditions"]
    snow_keep = lab_meta["name"].isin(names).to_numpy()
    lab_meta = lab_meta.loc[snow_keep].reset_index(drop=True)
    lab_sensor = observed_sensor_spectra(lab_w, lab_spectra[snow_keep], cw, fw)
    field_path = (
        ROOT / "data/raw/public_validation/zenodo_3580825/reflectance_spectra_field_2010_2011.zip"
    )
    field_meta, field_w, field_spectra = read_field_archive(field_path)
    bare_keep = ~field_meta.TargetType.isin([1, 2, 3]).to_numpy()
    bare_meta = field_meta.loc[bare_keep].reset_index(drop=True)
    bare_sensor = observed_sensor_spectra(field_w, field_spectra[bare_keep], cw, fw)
    if len(lab_meta) != 4 or len(bare_meta) != 5 or len(cw) != 150:
        raise ValueError("Unexpected independent-spectrum experiment schema")
    cases, y0 = build_cases(lab_meta, lab_sensor, bare_meta, bare_sensor)
    if len(cases) != 440 or not np.isfinite(y0).all():
        raise ValueError("Controlled cases contain invalid spectra")
    radii, snow_library_spectra = snow_library(
        "forward_models_55deg.mat", cw, fw, 55, cfg["ice_option"]
    )
    fixed = np.flatnonzero(radii == 200)
    modes = {
        "fixed200_original": (snow_library_spectra[fixed], radii[fixed], "original"),
        "multi_original": (snow_library_spectra, radii, "original"),
        "multi_fcls": (snow_library_spectra, radii, "fcls"),
    }
    rng = np.random.default_rng(config["seed"])
    observations = [y0]
    observation_metadata = [cases.assign(noise_std=0.0, repeat=0)]
    for sigma in (0.005, 0.01):
        for repeat in range(20):
            observations.append(y0 + rng.normal(0, sigma, y0.shape))
            observation_metadata.append(cases.assign(noise_std=sigma, repeat=repeat))
    observation = np.concatenate(observations)
    labels = pd.concat(observation_metadata, ignore_index=True)
    results, arrays = (
        [],
        {
            "wavelength_um": cw,
            "fwhm_um": fw,
            "controlled_observation": y0,
            "measured_snow": lab_sensor,
            "measured_background": bare_sensor,
            "fitting_background": backgrounds,
            "snow_library": snow_library_spectra,
            "radii_um": radii,
        },
    )
    print(
        f"BASE CASES: {len(cases)}; bands {len(cw)}; noise observations {len(labels)}", flush=True
    )
    with threadpool_limits(limits=1):
        for method, (library, candidate_radii, solver) in modes.items():
            fit = select_snow(
                observation,
                library,
                candidate_radii,
                backgrounds,
                solver=solver,
                minimum_snow=0.1,
                minimum_denominator=0.05,
            )
            frame = labels.copy()
            frame["method"] = method
            frame["estimated_snow_fraction"] = fit.normalized[:, 0]
            frame["estimated_raw_snow_coefficient"] = fit.fractions[:, 0]
            frame["estimated_shade"] = fit.shade
            frame["estimated_radius_um"] = fit.radius_um
            frame["rms"] = fit.rms
            frame["qa"] = fit.qa
            frame["matrix_condition"] = fit.condition
            frame["cover_error"] = frame.estimated_snow_fraction - frame.true_snow_fraction
            # 零噪声重复的输入一致，拟合一次后保存为 20 个重复记录。
            zero = frame.loc[frame.noise_std == 0]
            frame = pd.concat(
                [frame] + [zero.assign(repeat=r) for r in range(1, 20)], ignore_index=True
            )
            results.append(frame)
            arrays[method + "_modeled_zero_noise"] = fit.modeled[: len(cases)]
            arrays[method + "_Ec_zero_noise"] = fit.residual[: len(cases)]
            arrays[method + "_fractions_zero_noise"] = fit.fractions[: len(cases)]
            print(method, "finished", round(time.perf_counter() - started, 2), "s", flush=True)
    table = pd.concat(results, ignore_index=True)
    table.to_csv(path / "tables/controlled_mixture_results.csv", index=False)
    cases.to_csv(path / "tables/controlled_cases.csv", index=False)
    groups = []
    keys = ["method", "noise_std", "repeat", "snow_condition", "non_snow_target"]
    for labels_value, frame in table.groupby(keys, sort=False):
        groups.append({**dict(zip(keys, labels_value)), **group_statistics(frame)})
    grouped = pd.DataFrame(groups)
    grouped.to_csv(path / "tables/group_metrics.csv", index=False)
    summary = aggregate_group_statistics(grouped)
    summary.to_csv(path / "tables/noise_repeat_metrics.csv", index=False)
    intervals = []
    for (method, sigma), frame in summary.groupby(["method", "noise_std"], sort=False):
        row = {"method": method, "noise_std": sigma, "repeats": len(frame), "groups_per_repeat": 20}
        for field in (
            "cover_bias",
            "cover_mae",
            "cover_rmse",
            "radius_bias_um",
            "radius_rmse_um",
            "radius_missing_count",
            "false_negatives",
            "false_positives",
        ):
            row[field + "_mean"] = float(frame[field].mean())
            row[field + "_q025"] = float(frame[field].quantile(0.025))
            row[field + "_q975"] = float(frame[field].quantile(0.975))
        intervals.append(row)
    interval_table = pd.DataFrame(intervals)
    interval_table.to_csv(path / "tables/noise_intervals.csv", index=False)
    np.savez_compressed(path / "arrays/empirical_mixture_inputs_and_residuals.npz", **arrays)
    zero = table[(table.noise_std == 0) & (table.repeat == 0)].copy()
    positive_zero = zero[zero.true_snow_fraction >= 0.1 - 1e-10]
    # 图 1：各固定条件的中位数与离散范围。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for shade, ax in zip((0, 0.3), axes):
        for method, label in METHOD_LABELS.items():
            sample = positive_zero[
                (positive_zero.method == method) & (positive_zero.true_gray_shade == shade)
            ]
            distribution = sample.groupby("true_snow_fraction").estimated_snow_fraction
            middle = distribution.median()
            low = distribution.quantile(0.25)
            high = distribution.quantile(0.75)
            ax.plot(middle.index, middle, "o-", color=METHOD_COLORS[method], label=label, ms=4)
            ax.fill_between(middle.index, low, high, color=METHOD_COLORS[method], alpha=0.12)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="已知控制面积比例")
        ax.set(
            xlabel="控制雪面积比例 / 1",
            ylabel="估计阴影归一化雪比例 / 1",
            title=f"灰阴影系数 {shade:.1f}",
        )
        ax.legend(fontsize=8)
    fig.suptitle("真实端元谱组成的线性混合：中位数与20组合的四分位范围")
    save_figure(fig, path, "empirical_mixture_cover_curves")
    # 图 2：分别展示 20 个雪况与背景组合。
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    group_zero = grouped[(grouped.noise_std == 0) & (grouped.repeat == 0)]
    maximum = float(group_zero.cover_rmse.max() * 100)
    for method, ax in zip(METHOD_LABELS, axes):
        matrix = group_zero[group_zero.method == method].pivot(
            index="snow_condition", columns="non_snow_target", values="cover_rmse"
        )
        matrix = matrix.reindex(index=lab_meta["name"], columns=bare_meta.ID)
        shown = ax.imshow(
            matrix.to_numpy() * 100, vmin=0, vmax=maximum, cmap="magma", aspect="auto"
        )
        labels = [
            f"{row.ID}\n{TARGET_LABELS[int(row.TargetType)]}" for row in bare_meta.itertuples()
        ]
        ax.set(
            xticks=range(5),
            xticklabels=labels,
            yticks=range(4),
            yticklabels=lab_meta["name"],
            title=METHOD_LABELS[method],
        )
        ax.tick_params(axis="x", rotation=65, labelsize=8)
        ax.grid(False)
        for r in range(4):
            for c in range(5):
                value = matrix.iloc[r, c] * 100
                ax.text(
                    c,
                    r,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value < maximum * 0.55 else "black",
                )
    fig.subplots_adjust(bottom=0.29, right=0.90, wspace=0.15)
    fig.colorbar(shown, cax=fig.add_axes([0.92, 0.29, 0.015, 0.56]), label="覆盖 RMSE / 百分点")
    fig.suptitle("零噪声外部光谱失配：每格含10雪比例 × 2灰阴影")
    save_figure(fig, path, "empirical_mixture_group_errors")
    # 图 3：残差与覆盖误差、固定输入的噪声区间及 SSA 代理半径。
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for method, label in METHOD_LABELS.items():
        sample = positive_zero[positive_zero.method == method]
        axes[0].scatter(
            sample.rms * 100,
            np.abs(sample.cover_error) * 100,
            color=METHOD_COLORS[method],
            s=15,
            alpha=0.45,
            label=label,
        )
        noise = interval_table[interval_table.method == method].sort_values("noise_std")
        # 偏态重复的分位区间可能不包含均值，图中以中位数为中心；均值保存在 CSV。
        repeat_group = summary[summary.method == method].groupby("noise_std").cover_rmse
        center = repeat_group.median().reindex(noise.noise_std).to_numpy() * 100
        interval = np.vstack(
            (
                center - noise.cover_rmse_q025.to_numpy() * 100,
                noise.cover_rmse_q975.to_numpy() * 100 - center,
            )
        )
        axes[1].errorbar(
            noise.noise_std,
            center,
            yerr=interval,
            marker="o",
            capsize=4,
            color=METHOD_COLORS[method],
            label=label,
        )
        condition = sample.groupby("snow_condition").apply(
            lambda x: np.nanmean(x.estimated_radius_um - x.near_surface_ssa_radius_proxy_um)
        )
        axes[2].plot(
            range(4),
            condition.reindex(lab_meta["name"]),
            "o-",
            color=METHOD_COLORS[method],
            label=label,
        )
    axes[0].set(
        xlabel="光谱 RMS / 反射率百分点",
        ylabel="绝对覆盖偏差 / 百分点",
        title="低残差是否意味着覆盖准确？",
    )
    axes[1].set(
        xlabel="反射率噪声标准差 / 1", ylabel="覆盖 RMSE / 百分点", title="固定20组合的噪声重复区间"
    )
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set(
        xticks=range(4),
        xticklabels=lab_meta["name"],
        ylabel="候选半径 − SSA proxy / μm",
        title="组级近表层SSA差异（仅有效估计）",
    )
    axes[2].tick_params(axis="x", rotation=55, labelsize=8)
    for ax in axes:
        ax.legend(fontsize=7)
    fig.subplots_adjust(bottom=0.25, wspace=0.32)
    save_figure(fig, path, "empirical_mixture_residual_noise_and_ssa")
    source_files = [
        lab_path / "Darklab_snow_reflectance_avg.csv",
        lab_path / "Darklab_snow_SSA.zip",
        field_path,
        ROOT / "data/cache/aviris_metadata.json",
        ROOT / "data/cache/forward_models_55deg.mat",
    ]
    source_files += [ROOT / "data/raw/spectra" / entry["file"] for entry in background_metadata]
    provenance = [
        {
            "path": str(source.relative_to(ROOT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size_bytes": source.stat().st_size,
        }
        for source in source_files
    ]
    write_json(
        path / "source_links.json",
        {
            "files": provenance,
            "lab": "https://doi.org/10.5281/zenodo.3580078",
            "field": "https://doi.org/10.5281/zenodo.3580825",
            "background_metadata": background_metadata,
            "sensor_grid": grid_source,
            "fitting_background_response": "Gaussian integration; differs from center interpolation in earlier primary runs",
        },
    )
    captions = [
        "# 独立实测端元的控制混合补实验",
        "",
        "4种干雪条件均值与5个非雪目标实测谱构成440个控制案例。反射谱来自独立测量；面积与零谱灰阴影系数由线性混合假设设定。拟合使用冻结SNICAR和USGS/JPL库，真值非雪材料来自独立测量。",
        "",
        "![图1](figures/empirical_mixture_cover_curves.png)",
        "",
        "图1：横轴是控制混合中的雪面积比例，纵轴是阴影归一化估计。虚线是无偏结果；曲线是20个雪况/背景组合的中位数，浅色带为四分位范围，属于样本间描述而非置信区间。只画雪比例≥0.1；图中超界结果不截断。",
        "",
        "![图2](figures/empirical_mixture_group_errors.png)",
        "",
        "图2：每格一个雪况与一个非雪目标组合，数值为零噪声下10个正雪比例、2个灰阴影的覆盖RMSE。三方法共用色标，单位为覆盖百分点。5个非雪目标包括3条裸地、1条地衣和1条苔藓，其中相近目标有相关性；按20个雪况/背景组合汇总控制案例。",
        "",
        "![图3](figures/empirical_mixture_residual_noise_and_ssa.png)",
        "",
        "图3：左图逐控制案例展示光谱残差与覆盖误差，可发现低RMS却覆盖错误的模型。中图为同一固定20组合上每种非零噪声20次重复的中位数及2.5–97.5%分位，三方法共享噪声；零噪声只实际拟合一次并复用20次，区间为零宽。区间描述固定输入的噪声重复。右图为仅有效候选半径与雪况级IceCube最高记录点SSA等效半径proxy之差；SSA采用条件级近表层参照，缺失半径计数另存CSV，零雪不参与粒径评价。",
        "",
        "入射55°实验室方向雪谱与不同日期野外非雪谱按假设混合，未统一其真实BRDF/照明；SNICAR的方向—半球量与天底方向观测也有差别。它们是预先保留的外部光谱失配压力，不解释为独立空间像元的绝对精度。",
        "",
        "150个保守波段按configs/supplement_v2.json固定窗口，使用2013AVIRIS实际中心/FWHM的Gaussian响应近似；所有生成谱与模型端元用相同响应。此前合成实验的terrestrial loader采用中心插值，本补充混合流程使用一致卷积；v1实景AVIRIS背景已经做Gaussian卷积，不涉及更换背景算子。",
        "",
        "统计入口tables/group_metrics.csv和noise_intervals.csv；输入及零噪声Ec保存在arrays；SHA和来源见source_links.json，完整配置及源码见run_manifest.json/source_snapshot.zip。",
    ]
    (path / "README.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    metrics = {
        "base_cases": len(cases),
        "condition_target_groups": 20,
        "snow_conditions": 4,
        "non_snow_targets": 5,
        "band_count": len(cw),
        "reported_repeats": 20,
        "actual_zero_noise_repeats": 1,
        "noise_stds": [0, 0.005, 0.01],
        "result_rows": len(table),
        "positive_case_count_per_repeat": 400,
        "physical_input_geometries_differ": True,
        "measured_non_snow_fitted": False,
        "noise_intervals": intervals,
        "elapsed_seconds": time.perf_counter() - started,
        "primary_statistics": "Equal-weight group means; known snow fraction >= 0.1; missing radius separately counted",
    }
    finish_run(path, metrics)
    return path


if __name__ == "__main__":
    run()
