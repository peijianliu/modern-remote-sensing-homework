"""在已知失配与独立覆盖参照上比较拟合、约束和覆盖误差。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .diagnostics import coefficient_status, paired_rmse_difference
from .linear import select_snow


def mismatch_study(
    inputs: dict, angle60: dict, unseen_background: np.ndarray, repeats: int = 20, seed: int = 1998
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """七种预设条件下重复已知比例混合，返回逐次误差及重复试验分位。

    输入与候选库为 (radii, bands)，背景为 (bands,)；不同方法共享同一噪声。
    未知半径取相邻 10 μm 库谱的中点线性插值，明确表示网格失配。
    角度真值采用 60°，拟合库统一 0°；未入库背景使用独立 USGS 松针谱。
    20 次试验区间是噪声重复分位，不是实际影像覆盖的置信区间。
    """
    if repeats < 2:
        raise ValueError("At least two noise repetitions are required")
    snow, radii = np.asarray(inputs["snow"]), np.asarray(inputs["radii_um"])
    quartz = np.asarray(inputs["quartz"])
    unseen = np.asarray(unseen_background, float)
    if unseen.shape != quartz.shape or not np.isfinite(unseen).all():
        raise ValueError("Unseen background must match the finite sensor-band spectrum")
    rng = np.random.default_rng(seed)
    conditions = (
        "matched",
        "noise_005",
        "noise_010",
        "off_grid_radius",
        "unseen_background",
        "zenith_60_vs_0",
        "spectral_tilt",
    )
    records = []
    for condition in conditions:
        observations, truth, grain = [], [], []
        for radius in (50, 200, 500, 1000):
            index = int(np.flatnonzero(radii == radius)[0])
            spectrum = snow[index].copy()
            true_radius = radius
            if condition == "off_grid_radius":
                spectrum = (snow[index] + snow[index + 1]) / 2
                true_radius += 5
            elif condition == "zenith_60_vs_0":
                spectrum = np.asarray(angle60["snow"])[index]
            elif condition == "spectral_tilt":
                coordinate = np.linspace(-0.5, 0.5, snow.shape[1])
                spectrum *= 1 + 0.04 * coordinate
            background = unseen if condition == "unseen_background" else quartz
            for fraction in (0, 0.1, 0.3, 0.6, 0.9, 1):
                for shade in (0, 0.25):
                    observations.append(
                        (1 - shade) * (fraction * spectrum + (1 - fraction) * background)
                    )
                    truth.append(fraction)
                    grain.append(true_radius)
        truth, grain, clean = np.asarray(truth), np.asarray(grain), np.asarray(observations)
        std = 0.01 if condition == "noise_010" else (0.005 if condition == "noise_005" else 0)
        batch = np.tile(clean, (repeats, 1)) + rng.normal(
            0, std, (len(clean) * repeats, len(quartz))
        )
        fixed_index = np.flatnonzero(radii == 200)
        for method, library, grid, solver in (
            ("fixed200_MGS", snow[fixed_index], radii[fixed_index], "original"),
            ("multi_MGS", snow, radii, "original"),
            ("multi_FCLS", snow, radii, "fcls"),
        ):
            with threadpool_limits(limits=1):
                fit = select_snow(batch, library, grid, quartz, solver=solver)
            for repeat in range(repeats):
                section = slice(repeat * len(clean), (repeat + 1) * len(clean))
                estimate = fit.normalized[section, 0]
                good = np.isfinite(estimate)
                error = estimate[good] - truth[good]
                grain_keep = good & (truth >= 0.1) & np.isfinite(fit.radius_um[section])
                records.append(
                    {
                        "condition": condition,
                        "method": method,
                        "repeat": repeat,
                        "n": int(good.sum()),
                        "bias_pp": float(error.mean() * 100),
                        "rmse_pp": float(np.sqrt(np.mean(error**2)) * 100),
                        "spectral_rms_pp": float(fit.rms[section].mean() * 100),
                        "physical_feasible_fraction": float(
                            coefficient_status(fit.fractions[section], fit.shade[section])[
                                "feasible"
                            ].mean()
                        ),
                        "radius_rmse_um": float(
                            np.sqrt(
                                np.mean(
                                    (fit.radius_um[section][grain_keep] - grain[grain_keep]) ** 2
                                )
                            )
                        )
                        if grain_keep.any()
                        else np.nan,
                    }
                )
    repeated = pd.DataFrame(records)
    summary = (
        repeated.groupby(["condition", "method"], sort=False)
        .agg(
            repeats=("repeat", "size"),
            n=("n", "min"),
            bias_pp=("bias_pp", "mean"),
            rmse_pp=("rmse_pp", "mean"),
            rmse_p025_pp=("rmse_pp", lambda a: a.quantile(0.025)),
            rmse_p975_pp=("rmse_pp", lambda a: a.quantile(0.975)),
            spectral_rms_pp=("spectral_rms_pp", "mean"),
            feasible_fraction=("physical_feasible_fraction", "mean"),
            radius_rmse_um=("radius_rmse_um", "mean"),
        )
        .reset_index()
    )
    return repeated, summary


def independent_transfer(
    directory: Path, library: dict, repeats: int = 1000, seed: int = 1998
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """以公开松针、石英和角度匹配雪库重新解混两景 Landsat 教学样本。

    每行包含 B2–B7 观测、原始行列号和独立 WorldView 派生参照。参照只在
    解混结束后评价；不参与端元选择。六波段 B2–B7、四波段 B3/B5/B6/B7。
    两种波段配置、两种半径配置和两种求解器共八方法使用同一共同有效域。
    返回分层指标、配对 RMSE 差区间及预测数组。场景之间分别统计。
    """
    metrics, differences, predictions = [], [], {}
    names = np.asarray(library["scene_ids"]).tolist()
    for scene_index, scene in enumerate(names):
        path = directory / (scene + "_sample_pixels.csv.gz")
        table = pd.read_csv(path)
        observed = table[[f"observed_B{band}" for band in range(2, 8)]].to_numpy()
        truth = table.reference_fsca.to_numpy()
        positions = table[["row", "column"]].to_numpy()
        radii = np.asarray(library["radii_um"])
        snow, backgrounds = library["snow"][scene_index], library["backgrounds"]
        fitted = {}
        for bandset, selected in (("six", np.arange(6)), ("four", np.array([1, 3, 4, 5]))):
            for model, keep in (("fixed200", radii == 200), ("multi", np.ones(len(radii), bool))):
                for solver, mode in (("MGS", "original"), ("FCLS", "fcls")):
                    with threadpool_limits(limits=1):
                        result = select_snow(
                            observed[:, selected],
                            snow[keep][:, selected],
                            radii[keep],
                            backgrounds[selected],
                            solver=mode,
                        )
                    fitted[f"{model}_{bandset}_{solver}"] = result
        common = np.isfinite(truth) & np.logical_and.reduce(
            [np.isfinite(result.normalized[:, 0]) for result in fitted.values()]
        )
        predictions[scene] = {key: result.normalized[:, 0] for key, result in fitted.items()}
        predictions[scene]["common"] = common
        for stratum in ("all", "no_snow", "mixed", "full_snow"):
            use = common if stratum == "all" else common & (table.stratum.to_numpy() == stratum)
            for name, result in fitted.items():
                error = result.normalized[use, 0] - truth[use]
                metrics.append(
                    {
                        "scene": scene,
                        "method": name,
                        "stratum": stratum,
                        "n": int(use.sum()),
                        "bias_pp": float(error.mean() * 100) if len(error) else np.nan,
                        "rmse_pp": float(np.sqrt(np.mean(error**2)) * 100)
                        if len(error)
                        else np.nan,
                        "spectral_rms_pp": float(result.rms[use].mean() * 100)
                        if use.any()
                        else np.nan,
                        "feasible_fraction": float(
                            coefficient_status(result.fractions[use], result.shade[use])[
                                "feasible"
                            ].mean()
                        )
                        if use.any()
                        else np.nan,
                    }
                )
            for bandset in ("six", "four"):
                for first, second in (
                    (f"fixed200_{bandset}_MGS", f"multi_{bandset}_MGS"),
                    (f"fixed200_{bandset}_FCLS", f"multi_{bandset}_FCLS"),
                    (f"multi_{bandset}_MGS", f"multi_{bandset}_FCLS"),
                ):
                    delta = paired_rmse_difference(
                        truth[use],
                        fitted[first].normalized[use, 0],
                        fitted[second].normalized[use, 0],
                        positions[use],
                        repeats=repeats,
                        seed=seed,
                    )
                    differences.append(
                        {
                            "scene": scene,
                            "stratum": stratum,
                            "first": first,
                            "second": second,
                            **delta,
                        }
                    )
    return pd.DataFrame(metrics), pd.DataFrame(differences), predictions
