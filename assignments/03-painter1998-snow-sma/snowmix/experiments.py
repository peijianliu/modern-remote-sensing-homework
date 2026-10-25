"""雪光谱正演与已知端元比例的混合实验。"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
from .artifacts import (
    ROOT,
    configure_plots,
    new_run,
    finish_run,
    save_figure,
    write_json,
    paired_metrics,
)
from .linear import select_snow
from .cache_contract import validate_forward_settings
from .spectra import (
    read_usgs_ascii,
    gaussian_band_matrix,
    load_terrestrial_library,
)


def forward_data():
    """读取雪光谱正演缓存，将波长、半径和入射角坐标转为一维数组。"""
    data = loadmat(ROOT / "data/cache/forward_models.mat")
    for key in ("wavelength_um", "radii_um", "angles_deg"):
        data[key] = data[key].ravel()
    return data


def sensor_grid(config):
    """读取配置指定的合成实验波段中心与半高全宽。"""
    # 高斯光谱响应采用 2013 年 AVIRIS 波段中心与半高全宽。
    metadata = ROOT / "data/cache/aviris_metadata.json"
    mode = config["synthetic"].get("sensor_grid_mode", "aviris2013")
    if mode == "aviris2013":
        data = json.loads(metadata.read_text(encoding="utf-8"))
        return (
            np.array(data["wavelength_um"]),
            np.array(data["fwhm_um"]),
            "2013 AVIRIS centers/FWHM; Gaussian RSR approximation",
        )
    if mode != "nominal224":
        raise ValueError(f"Unknown synthetic sensor grid: {mode}")
    return (
        np.linspace(0.4, 2.5, 224),
        np.full(224, 0.01),
        "nominal evenly spaced 224 bands, 0.4-2.5 um; Gaussian FWHM=10 nm; modern nominal-band approximation",
    )


def synthetic_inputs(config):
    """选取配置中的正演参数，生成同一波段网格上的雪、石英及对照光谱。"""
    data = forward_data()
    effective = validate_forward_settings(data, config)
    wavelengths, fwhm, provenance = sensor_grid(config)
    response = gaussian_band_matrix(data["wavelength_um"], wavelengths, fwhm)
    ai = np.flatnonzero(data["angles_deg"] == config["synthetic"]["angle_deg_assumed"])[0]
    radii = data["radii_um"]
    cfg = config["synthetic"]
    if cfg["radius_step_um"] <= 0 or cfg["radius_min_um"] > cfg["radius_max_um"]:
        raise ValueError("Snow radius range and step must be positive and ordered")
    requested = np.arange(
        cfg["radius_min_um"],
        cfg["radius_max_um"] + cfg["radius_step_um"] / 2,
        cfg["radius_step_um"],
    )
    if not np.isin(requested, radii).all():
        raise ValueError(
            "Requested snow radii are absent from frozen optics cache; no extrapolation is allowed"
        )
    subset = np.isin(radii, requested)
    optical_indices = np.flatnonzero(data["ice_options"].ravel() == config["forward"]["ice_ri"])
    if len(optical_indices) != 1:
        raise ValueError("Requested ice optical option is absent")
    oi = int(optical_indices[0])
    snow = data["ww_albedo"][subset, :, ai, oi] @ response
    snicar = data["snicar_albedo"][subset, :, ai, oi] @ response
    qw, qr, qm = read_usgs_ascii(ROOT / "data/raw/spectra/quartz_hs32.5854.asc")
    # 高分辨率石英库谱采用与雪光谱相同的高斯响应积分。
    quartz_response = gaussian_band_matrix(qw, wavelengths, fwhm)
    quartz = qr @ quartz_response
    return (
        wavelengths,
        radii[subset],
        snow,
        quartz,
        snicar,
        {
            "sensor": provenance,
            "quartz": qm,
            "solar_angle_assumption_deg": float(data["angles_deg"][ai]),
            "forward": "WW80 semi-infinite delta-Eddington with SNICAR spherical lognormal properties",
            "ice_option": int(config["forward"]["ice_ri"]),
            "actual_radii_um": radii[subset].tolist(),
            "effective_cache_settings": effective,
        },
    )


def run_physics(config):
    """绘制雪粒半径与光谱反射率、太阳谱加权反照率的关系，并比较两种求解器。"""
    path = new_run(config, "01_physics")
    configure_plots()
    data = forward_data()
    validate_forward_settings(data, config)
    rows = []
    wavelength = data["wavelength_um"]
    for angle in (0, 60):
        index = np.flatnonzero(data["angles_deg"] == angle)[0]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for radius in (50, 200, 500, 1000):
            ri = np.flatnonzero(data["radii_um"] == radius)[0]
            spectrum = data["snicar_albedo"][ri, :, index, 0]
            ax.plot(wavelength, spectrum, label=f"r = {radius} μm", linewidth=1.8)
            for w, a in zip(wavelength, spectrum):
                rows.append(
                    {"angle_deg": angle, "radius_um": radius, "wavelength_um": w, "albedo": a}
                )
        ax.set(
            xlim=(0.3, 2.5),
            ylim=(0, 1),
            xlabel="波长 / μm",
            ylabel="方向—半球光谱反射率",
            title=f"纯净球形厚雪层；直接入射 {angle}°；SNICAR / Warren 1984",
        )
        ax.legend(ncol=2)
        save_figure(fig, path, f"snow_spectra_{angle}deg")
    pd.DataFrame(rows).to_csv(path / "tables/spectral_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    broadband_rows = []
    for ai, angle in enumerate(data["angles_deg"]):
        if angle not in (0, 60):
            continue
        for oi, label in ((0, "Warren 1984"), (1, "Warren–Brandt 2008")):
            alpha = data["broadband_snicar"][:, ai, oi]
            ax.plot(
                data["radii_um"],
                alpha,
                label=f"{angle}° / {label}",
                linestyle="-" if oi == 0 else "--",
            )
            for r, a in zip(data["radii_um"], alpha):
                broadband_rows.append(
                    {"radius_um": r, "angle_deg": angle, "ice_option": label, "albedo": a}
                )
    ax.axvline(100, color="#5ba5ac", alpha=0.6, linewidth=1)
    ax.axvline(1000, color="#b8703d", alpha=0.6, linewidth=1)
    ax.text(100, 0.68, "新雪代表\n100 μm", ha="center", va="bottom")
    ax.text(1000, 0.87, "老雪代表\n1000 μm", ha="center", va="bottom")
    ax.set(
        xscale="log",
        xlim=(30, 1500),
        ylim=(0.65, 0.94),
        xlabel="光学等效雪粒半径 / μm",
        ylabel="太阳谱加权反照率（0.2–5 μm）",
        title="纯净厚雪层；中纬度冬季入射谱",
    )
    ax.legend(loc="lower left", fontsize=9)
    save_figure(fig, path, "grain_radius_broadband_albedo")
    pd.DataFrame(broadband_rows).to_csv(path / "tables/broadband_albedo.csv", index=False)
    difference = data["snicar_albedo"] - data["ww_albedo"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ai = np.flatnonzero(data["angles_deg"] == 60)[0]
    for radius in (50, 200, 500, 1000):
        ri = np.flatnonzero(data["radii_um"] == radius)[0]
        ax.plot(wavelength, difference[ri, :, ai, 0] * 100, label=f"{radius} μm")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xlim=(0.3, 2.5),
        xlabel="波长 / μm",
        ylabel="SNICAR − WW80 / 反射率百分点",
        title="求解器交叉检查：相同球形光学属性，60°",
    )
    ax.legend()
    save_figure(fig, path, "forward_model_difference")
    metrics = {
        "radius_count": len(data["radii_um"]),
        "angles_deg": data["angles_deg"],
        "snicar_min": float(data["snicar_albedo"].min()),
        "snicar_max": float(data["snicar_albedo"].max()),
        "max_depth_100m_vs_200m_difference": float(data["depth_differences"][:, 3].max()),
        "max_solver_difference": float(np.max(np.abs(difference))),
        "max_ice_constants_difference": float(
            np.max(np.abs(data["snicar_albedo"][:, :, :, 0] - data["snicar_albedo"][:, :, :, 1]))
        ),
        "albedo_60deg_100um": float(data["broadband_snicar"][7, ai, 0]),
        "albedo_60deg_1000um": float(data["broadband_snicar"][97, ai, 0]),
    }
    (path / "README.md").write_text(
        "# 基础物理实验\n\n使用官方 SNICAR-ADv3 的冻结源码，唯一源码改动为工作副本的光学库路径。球形、无杂质、100 m 雪深、200 kg/m³、零底层反照率、直接入射、中纬度冬季太阳谱；新/老雪仅为半径代表标记。\n\nWW80 与 SNICAR 使用相同更新后的光学属性，用于求解器交叉检查。标准库半径范围30–1500 μm；历史光学常数差异见方法说明。\n",
        encoding="utf-8",
    )
    finish_run(path, metrics)
    return path


def run_paper(config):
    """生成雪与石英的已知混合谱，比较固定粒径与多粒径端元并输出 E_c。"""
    path = new_run(config, "02_03_paper")
    configure_plots()
    w, radii, snow, quartz, snicar, source = synthetic_inputs(config)
    step = config["synthetic"]["fraction_step"]
    if step <= 0:
        raise ValueError("fraction_step must be positive")
    intervals = round(1 / step)
    if not np.isclose(intervals * step, 1):
        raise ValueError("fraction_step must divide [0,1] exactly")
    f = np.linspace(0, 1, intervals + 1)
    truth_snow = np.repeat(f[None, :], len(radii), axis=0).ravel()
    truth_radius = np.repeat(radii[:, None], len(f), axis=1).ravel()
    y = (
        snow[:, None, :] * f[None, :, None] + quartz[None, None, :] * (1 - f[None, :, None])
    ).reshape(-1, len(w))
    fixed_radius = config["synthetic"]["fixed_radius_um"]
    candidates = np.flatnonzero(radii == fixed_radius)
    if len(candidates) != 1:
        raise ValueError("Fixed snow radius must be included in the candidate cache grid")
    fixed = int(candidates[0])
    with threadpool_limits(limits=1):
        single = select_snow(y, snow[fixed : fixed + 1], [fixed_radius], quartz)
        multiple = select_snow(y, snow, radii, quartz)
    grid = single.rms.reshape(len(radii), len(f)) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(f, radii, grid, shading="auto", cmap="magma")
    ax.axhline(fixed_radius, color="white", lw=1, linestyle="--")
    ax.set(
        xlabel="真实雪覆盖比例",
        ylabel="真实雪粒半径 / μm",
        title=f"固定 {fixed_radius:g} μm 雪端元：光谱 RMS 误差面",
    )
    fig.colorbar(im, ax=ax, label="光谱 RMS / 反射率百分点")
    save_figure(fig, path, "fig3a_rms_heatmap")
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(projection="3d")
    xx, yy = np.meshgrid(f, radii)
    ax.plot_surface(xx, yy, grid, cmap="magma", linewidth=0, antialiased=True, rstride=3, cstride=3)
    ax.set(
        xlabel="真实雪比例",
        ylabel="半径 / μm",
        zlabel="RMS / pp",
        title="Fig. 3a 流程复现（替代输入）",
    )
    ax.view_init(25, -125)
    save_figure(fig, path, "fig3a_rms_surface")
    pure_single = single.normalized.reshape(len(radii), len(f), 2)[:, -1, 0]
    pure_multiple = multiple.normalized.reshape(len(radii), len(f), 2)[:, -1, 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(radii, pure_single, label=f"固定 {fixed_radius:g} μm 雪端元", lw=2)
    ax.plot(radii, pure_multiple, label="逐像元 MIN RMS", lw=1.5, linestyle="--")
    ax.axhline(1, color="black", lw=0.8, label="真实 100% 雪")
    ax.axvline(fixed_radius, color="gray", linestyle=":")
    ax.set(
        xlabel="真实半径 / μm",
        ylabel="阴影归一化雪比例",
        title="Fig. 3b：纯雪被错误粒径端元解混的覆盖偏差",
    )
    ax.legend()
    save_figure(fig, path, "fig3b_pure_snow_fraction")
    ri = int(np.argmin(abs(radii - 480)))
    si = int(np.argmin(abs(radii - 150)))
    diagnostic = select_snow(snow[ri : ri + 1], snow[si : si + 1], [radii[si]], quartz)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(w, snow[ri], label=f"真实 {radii[ri]:g} μm 纯雪")
    axes[0].plot(
        w, diagnostic.modeled[0], label=f"{radii[si]:g} μm 雪 + quartz + shade 拟合", linestyle="--"
    )
    axes[0].set(ylabel="反射率", title="残差诊断（合成类似场景；原文像元约478 μm）")
    axes[0].legend()
    axes[1].plot(w, diagnostic.residual[0] * 100, color="#b45131")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="波长 / μm", ylabel="observed − modeled / pp")
    for marker in (1.03, 1.26):
        axes[1].axvline(marker, color="gray", lw=0.7, linestyle=":")
    save_figure(fig, path, "residual_spectrum_480_vs_150um")
    table = pd.DataFrame(
        {
            "true_radius_um": truth_radius,
            "true_snow_fraction": truth_snow,
            "fixed_snow_fraction": single.normalized[:, 0],
            "fixed_raw_snow": single.fractions[:, 0],
            "fixed_raw_quartz": single.fractions[:, 1],
            "fixed_shade": single.shade,
            "fixed_rms": single.rms,
            "minimum_snow_fraction": multiple.normalized[:, 0],
            "minimum_radius_um": multiple.radius_um,
            "minimum_rms": multiple.rms,
        }
    )
    table.to_csv(path / "tables/fig3_grid.csv", index=False)
    pd.DataFrame(
        {
            "wavelength_um": w,
            "observed": snow[ri],
            "modeled": diagnostic.modeled[0],
            "E_c": diagnostic.residual[0],
        }
    ).to_csv(path / "tables/residual_480_vs_150.csv", index=False)
    np.savez_compressed(
        path / "arrays/known_truth.npz",
        radii_um=radii,
        wavelength_um=w,
        snow=snow,
        quartz=quartz,
        true_fraction=truth_snow,
        true_radius=truth_radius,
        fixed_rms=single.rms,
        minimum_rms=multiple.rms,
    )
    matching = np.isclose(truth_radius, fixed_radius)
    has_snow = truth_snow >= 0.1
    assert single.rms[matching].max() < 1e-8
    assert multiple.rms.max() < 1e-8
    assert np.all(multiple.rms <= single.rms + 1e-12)
    metrics = {
        "synthetic_shape": [len(radii), len(f), len(w)],
        "sources_and_assumptions": source,
        "fixed_fraction": paired_metrics(truth_snow, single.normalized[:, 0]),
        "minimum_fraction": paired_metrics(truth_snow, multiple.normalized[:, 0]),
        "minimum_radius_snow_ge_0p1": paired_metrics(
            truth_radius[has_snow], multiple.radius_um[has_snow]
        ),
        "fixed_radius_um": fixed_radius,
        "matched_fixed_max_rms": float(single.rms[matching].max()),
        "matched_200um_max_rms": float(single.rms[matching].max()) if fixed_radius == 200 else None,
        "minimum_max_rms": float(multiple.rms.max()),
        "pure_snow_estimate_50um": float(pure_single[0]) if radii[0] == 50 else None,
        "pure_snow_estimate_1500um": float(pure_single[-1]) if radii[-1] == 1500 else None,
        "diagnostic_truth_radius_um": float(radii[ri]),
        "diagnostic_fit_radius_um": float(radii[si]),
        "fixed_fraction_underflow_rate": float(np.mean(single.normalized[:, 0] < -1e-8)),
        "fixed_fraction_overflow_rate": float(np.mean(single.normalized[:, 0] > 1 + 1e-8)),
        "published_pure_snow_50um_approx": 0.86,
        "published_pure_snow_1500um_approx": 1.2,
        "rms_is_spectral_not_cover_error": True,
    }
    (path / "README.md").write_text(
        "# Fig. 3 与多雪端元实验\n\n先生成无阴影的雪＋quartz已知真值，再允许零光谱阴影参与拟合。原文模式采用无约束最小二乘，保留越界系数。E_c=observed−modeled，使用原始光谱系数重建。\n\n"
        + source["sensor"]
        + "；模拟入射角0°是本文执行假设。quartz为USGS2003 HS32.4B，WW80公式的光学属性来自现代SNICAR。同模型合成数据用于数值实现的内部验证。\n",
        encoding="utf-8",
    )
    finish_run(path, metrics)
    return path


def run_stability(config):
    """比较噪声、系数约束、端元遗漏和端元共线性对解混结果的影响。"""
    path = new_run(config, "04_stability")
    configure_plots()
    w, radii, snow, quartz, snicar, source = synthetic_inputs(config)
    if not np.isin([50, 200, 500, 1000, 1500], radii).all():
        raise ValueError(
            "The stability stage requires its declared 50/200/500/1000/1500 um control cases"
        )
    cases = []
    for radius in (50, 200, 500, 1000, 1500):
        ri = np.flatnonzero(radii == radius)[0]
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            for shade in (0, 0.3, 0.7):
                cases.append((ri, radius, fraction, shade))
    cases = np.asarray(cases)
    truth = (1 - cases[:, 3, None]) * (
        cases[:, 2, None] * snow[cases[:, 0].astype(int)] + (1 - cases[:, 2, None]) * quartz
    )
    rows = []
    fixed = np.flatnonzero(radii == 200)[0]
    with threadpool_limits(limits=1):
        for sigma in config["synthetic"]["noise_std"]:
            for repeat in range(config["synthetic"]["repeats"]):
                # 各方法共享同一组噪声，实现配对比较。
                rng = np.random.default_rng(config["seed"] + repeat + int(sigma * 10000))
                y = truth + rng.normal(0, sigma, truth.shape)
                models = {
                    "fixed_original": (snow[fixed : fixed + 1], np.array([200]), "original"),
                    "multi_original": (snow, radii, "original"),
                    "multi_fcls": (snow, radii, "fcls"),
                }
                for method, (s, rs, solver) in models.items():
                    result = select_snow(y, s, rs, quartz, solver=solver)
                    fraction_metrics = paired_metrics(cases[:, 2], result.normalized[:, 0])
                    valid = cases[:, 2] >= 0.1
                    radius_metrics = paired_metrics(cases[valid, 1], result.radius_um[valid])
                    rows.append(
                        {
                            "noise_std": sigma,
                            "repeat": repeat,
                            "method": method,
                            "cover_bias": fraction_metrics["bias"],
                            "cover_mae": fraction_metrics["mae"],
                            "cover_rmse": fraction_metrics["rmse"],
                            "radius_bias_um": radius_metrics["bias"],
                            "radius_mae_um": radius_metrics["mae"],
                            "radius_rmse_um": radius_metrics["rmse"],
                            "radius_valid_count": radius_metrics["n"],
                            "cover_valid_count": fraction_metrics["n"],
                            "spectral_rms_mean": float(np.nanmean(result.rms)),
                            "overflow_rate": float(
                                np.mean(
                                    (result.normalized[:, 0] < -1e-8)
                                    | (result.normalized[:, 0] > 1 + 1e-8)
                                )
                            ),
                            "denominator_invalid_rate": float(np.mean((result.qa & 4) != 0)),
                        }
                    )
            print(f"stability: sigma={sigma} complete", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(path / "tables/monte_carlo_repeats.csv", index=False)
    summaries = []
    for (sigma, method), group in table.groupby(["noise_std", "method"]):
        record = {"noise_std": sigma, "method": method}
        for metric in (
            "cover_rmse",
            "cover_bias",
            "radius_rmse_um",
            "spectral_rms_mean",
            "overflow_rate",
        ):
            record[metric + "_mean"] = float(group[metric].mean())
            record[metric + "_p025"] = float(group[metric].quantile(0.025))
            record[metric + "_p975"] = float(group[metric].quantile(0.975))
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(path / "tables/monte_carlo_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for method, group in table.groupby("method"):
        means = group.groupby("noise_std").mean(numeric_only=True)
        for ax, metric, label in zip(
            axes, ("cover_rmse", "radius_rmse_um"), ("雪覆盖 RMSE", "半径 RMSE / μm")
        ):
            ax.plot(means.index, means[metric], marker="o", label=method)
            ax.set(xlabel="反射率噪声标准差", ylabel=label)
    axes[0].legend(fontsize=8)
    fig.suptitle("已知真值控制实验；20次重复，粒径仅在有雪时评价")
    save_figure(fig, path, "noise_and_constraints")
    library_valid = (w >= 0.42) & (w <= 2.45)
    library, library_metadata = load_terrestrial_library(
        w[library_valid], ROOT / "data/raw/spectra"
    )
    ri = np.flatnonzero(radii == 500)[0]
    observation = 0.5 * snow[ri, library_valid] + 0.3 * library[:, 0] + 0.2 * quartz[library_valid]
    omitted = select_snow(observation, snow[:, library_valid], radii, quartz[library_valid])
    complete = select_snow(
        observation,
        snow[:, library_valid],
        radii,
        np.column_stack((quartz[library_valid], library[:, 0])),
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True)
    axes[0].plot(w[library_valid], observation, label="观测：雪0.5 + 松针0.3 + quartz0.2")
    axes[0].plot(w[library_valid], omitted.modeled[0], label="遗漏植被的拟合", linestyle="--")
    axes[0].plot(w[library_valid], complete.modeled[0], label="完整端元拟合", linestyle=":")
    axes[0].legend(fontsize=9)
    axes[0].set(ylabel="反射率", title="端元遗漏对照：相同雪粒径，不同模型完备性")
    axes[1].plot(w[library_valid], omitted.residual[0] * 100, label="遗漏植被 E_c")
    axes[1].plot(w[library_valid], complete.residual[0] * 100, label="完整模型 E_c")
    axes[1].set(xlabel="波长 / μm", ylabel="残差 / pp")
    axes[1].legend()
    save_figure(fig, path, "omitted_endmember_residual")
    # 用两条完全相同的背景谱构造秩亏对照。
    identical = select_snow(
        observation,
        snow[:, library_valid],
        radii,
        np.column_stack((quartz[library_valid], quartz[library_valid])),
    )
    near = quartz + 1e-4 * np.sin(w * 12)
    near_rows = []
    rng = np.random.default_rng(config["seed"])
    for repeat in range(20):
        yy = 0.5 * snow[ri] + 0.25 * quartz + 0.25 * near + rng.normal(0, 0.005, len(w))
        result = select_snow(yy, snow[ri : ri + 1], [500], np.column_stack((quartz, near)))
        near_rows.append(
            {
                "repeat": repeat,
                "quartz_1": result.fractions[0, 1],
                "quartz_2": result.fractions[0, 2],
                "bare_sum": result.fractions[0, 1:].sum(),
                "condition": result.condition[0],
                "rms": result.rms[0],
            }
        )
    pd.DataFrame(near_rows).to_csv(path / "tables/near_collinear_endmembers.csv", index=False)
    mismatch = select_snow(snicar, snow, radii, quartz)
    metrics = {
        "summaries": summaries,
        "sources_and_assumptions": source,
        "omission": {
            "true_snow": 0.5,
            "omitted_snow": float(omitted.normalized[0, 0]),
            "complete_snow": float(complete.normalized[0, 0]),
            "omitted_rms": float(omitted.rms[0]),
            "complete_rms": float(complete.rms[0]),
        },
        "identical_background_rejected": bool((identical.qa[0] & 2) != 0),
        "near_collinear_quartz_component_std": float(
            np.std(np.array([[x["quartz_1"], x["quartz_2"]] for x in near_rows]))
        ),
        "cross_model_pure_snow_cover": paired_metrics(
            np.ones(len(radii)), mismatch.normalized[:, 0]
        ),
        "cross_model_radius_um": paired_metrics(radii, mismatch.radius_um),
    }
    write_json(path / "endmember_metadata.json", library_metadata)
    (path / "README.md").write_text(
        "# 噪声、约束和失败原因\n\n75个控制案例：5种半径×5种雪比例×3种阴影；每噪声水平20次，方法间共享相同噪声。重复区间描述固定输入下的噪声扰动。\n\nFCLS枚举非负简单形的活跃面，阴影是剩余比例。遗漏松针的实验检验端元缺失；相同/近似quartz端元检验不可辨识性，构造谱用于端元相似性的数值对照。\n",
        encoding="utf-8",
    )
    finish_run(path, metrics)
    return path
