"""以共同留出波段评价预定波段窗口、光学常数和入射角设置。"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import (
    ROOT,
    new_run,
    finish_run,
    configure_plots,
    save_figure,
    write_json,
    paired_metrics,
)
from snowmix.image import prepare_roi, sensor_library
from snowmix.linear import select_snow
from snowmix.validation import interval_mask, block_bootstrap_delta
import matplotlib.pyplot as plt


def run() -> Path:
    config = json.loads((ROOT / "configs/supplement_v2.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
    cfg = config["band_sensitivity"]
    config["image"] = baseline["image"]
    config["forward"] = baseline["forward"]
    path = new_run(config, "12_aviris_band_sensitivity")
    configure_plots()
    plt.rcParams["savefig.dpi"] = 300
    metadata = json.loads((ROOT / "data/cache/aviris_metadata.json").read_text(encoding="utf-8"))
    cube, valid, original_mask, transform, shape, crs, _ = prepare_roi(baseline, metadata)
    w, fw = np.array(metadata["wavelength_um"]), np.array(metadata["fwhm_um"])
    rng = np.random.default_rng(config["seed"])
    ids = rng.choice(np.flatnonzero(valid.ravel()), cfg["pixel_sample_size"], replace=False)
    row, col = np.unravel_index(ids, shape)
    blocks = (row * cfg["spatial_blocks_per_axis"] // shape[0]) * cfg[
        "spatial_blocks_per_axis"
    ] + col * cfg["spatial_blocks_per_axis"] // shape[1]
    y = cube.reshape(-1, len(w))[ids].astype(float)
    conservative = original_mask & interval_mask(w, cfg["conservative_intervals_um"])
    holdout = np.zeros(len(w), bool)
    holdout[np.flatnonzero(conservative)[:: cfg["holdout_stride"]]] = True
    wide_train, narrow_train = original_mask & ~holdout, conservative & ~holdout
    write_json(
        path / "band_masks.json",
        {
            "wavelength_um": w,
            "original_mask": original_mask,
            "conservative_mask": conservative,
            "common_holdout": holdout,
            "wide_train": wide_train,
            "conservative_train": narrow_train,
        },
    )
    variants = [
        ("wide_182_train", wide_train, 22, 1),
        ("conservative_train", narrow_train, 22, 1),
        ("conservative_WB2008", narrow_train, 22, 2),
        ("conservative_angle0", narrow_train, 0, 1),
        ("conservative_angle60", narrow_train, 60, 1),
    ]
    results, rows, spectral_rows = {}, [], []
    with threadpool_limits(limits=1):
        for label, train, angle, ice in variants:
            local = json.loads(json.dumps(baseline))
            local["forward"]["ice_ri"] = ice
            radii, snow, bg, _ = sensor_library(w[original_mask], fw[original_mask], angle, local)
            subset = train[original_mask]
            fit = select_snow(y[:, train], snow[:, subset], radii, bg[subset, :3], solver="fcls")
            modeled = (
                fit.fractions[:, :1] * snow[fit.radius_index] + fit.fractions[:, 1:] @ bg[:, :3].T
            )
            ec = y[:, original_mask] - modeled
            heldout_rms = np.sqrt(np.mean(ec[:, holdout[original_mask]] ** 2, axis=1))
            common_rms = np.sqrt(np.mean(ec[:, conservative[original_mask]] ** 2, axis=1))
            results[label] = {
                "snow": fit.normalized[:, 0],
                "radius": fit.radius_um,
                "rms": fit.rms,
                "heldout_rms": heldout_rms,
                "common_rms": common_rms,
                "qa": fit.qa,
                "Ec": ec,
            }
            for i, pixel in enumerate(ids):
                rows.append(
                    {
                        "method": label,
                        "pixel_index": pixel,
                        "row": row[i],
                        "column": col[i],
                        "spatial_block": blocks[i],
                        "snow_fraction": fit.normalized[i, 0],
                        "radius_um": fit.radius_um[i],
                        "qa": int(fit.qa[i]),
                        "training_rms": fit.rms[i],
                        "common_rms": common_rms[i],
                        "heldout_rms": heldout_rms[i],
                    }
                )
            for j, wave in enumerate(w[original_mask]):
                spectral_rows.append(
                    {
                        "method": label,
                        "wavelength_um": wave,
                        "mean_Ec": np.mean(ec[:, j]),
                        "band_rmse": np.sqrt(np.mean(ec[:, j] ** 2)),
                        "common_holdout": bool(holdout[original_mask][j]),
                        "conservative_band": bool(conservative[original_mask][j]),
                    }
                )
            print(label, "mean heldout RMS", heldout_rms.mean(), flush=True)
    pd.DataFrame(rows).to_csv(path / "tables/paired_sample_results.csv", index=False)
    spectral = pd.DataFrame(spectral_rows)
    spectral.to_csv(path / "tables/band_residuals.csv", index=False)
    reference = results["wide_182_train"]
    metrics = {
        "sample_n": len(ids),
        "valid_roi_pixels": int(valid.sum()),
        "selection": "uniform random prefit valid pixels; fixed seed",
        "spatial_blocks": len(np.unique(blocks)),
        "original_band_n": int(original_mask.sum()),
        "conservative_band_n": int(conservative.sum()),
        "holdout_band_n": int(holdout.sum()),
        "validation_target": "prediction of shared withheld spectral bands",
        "methods": {},
    }
    for label, fit in results.items():
        metrics["methods"][label] = {
            "heldout_rms_mean": float(fit["heldout_rms"].mean()),
            "heldout_rms_median": float(np.median(fit["heldout_rms"])),
            "common_rms_mean": float(fit["common_rms"].mean()),
            "cover_difference_vs_wide": paired_metrics(reference["snow"], fit["snow"]),
            "radius_boundary_rate": float(np.mean((fit["qa"] & 16) != 0)),
            "paired_holdout_rms_delta_block_bootstrap": block_bootstrap_delta(
                blocks,
                fit["heldout_rms"] - reference["heldout_rms"],
                config["seed"],
                cfg["bootstrap_repeats"],
            ),
        }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for label in ("wide_182_train", "conservative_train", "conservative_WB2008"):
        sub = spectral[spectral.method == label]
        shown = sub.band_rmse.to_numpy() * 100
        shown[np.r_[False, np.diff(sub.wavelength_um) > 0.03]] = np.nan
        axes[0].plot(sub.wavelength_um, shown, label=label, lw=1.2)
    for low, high in ((1.3, 1.5), (1.75, 2.05)):
        axes[0].axvspan(low, high, color="gray", alpha=0.15)
    axes[0].set(
        xlabel="波长 / μm",
        ylabel="逐波段 RMSE / 反射率百分点",
        title="同一1200像元；灰区为预定保守排除窗口",
    )
    axes[0].legend(fontsize=8)
    labels = list(results)
    axes[1].boxplot(
        [results[x]["heldout_rms"] * 100 for x in labels], labels=labels, showfliers=False
    )
    axes[1].set(ylabel="公共留出波段 RMS / 反射率百分点", title="所有变体均未拟合相同30个留出波段")
    axes[1].tick_params(axis="x", rotation=65, labelsize=8)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    save_figure(fig, path, "withheld_spectral_band_sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].scatter(reference["snow"], results["conservative_train"]["snow"], s=8, alpha=0.3)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.9)
    axes[0].set(
        xlabel="宽波段模型雪比例", ylabel="保守波段模型雪比例", title="预定波段选择下的比例敏感性"
    )
    for label in ("wide_182_train", "conservative_train"):
        sample = results[label]["radius"]
        finite = np.isfinite(sample)
        axes[1].hist(
            sample[finite],
            bins=np.arange(25, 1116, 30),
            histtype="step",
            label=f"{label}; n={finite.sum()}",
        )
    axes[1].set(xlabel="估计粒径 / μm", ylabel="随机样本数", title="粒径堆积与库边界敏感性")
    axes[1].legend(fontsize=8)
    save_figure(fig, path, "cover_and_radius_band_sensitivity")
    text = [
        "# 波段、光学常数和入射角的补充诊断",
        "",
        "在解混前从有效AVIRIS像元随机抽1200个，seed=1998；保守窗口预先设为0.42–1.30、1.50–1.75、2.05–2.35 μm。每5个保守波段留出1个，五种模型都不拟合这些相同波段。误差差异按8×8空间块bootstrap；评价对象为公共留出波段的光谱预测误差。",
        "",
        "![图1](figures/withheld_spectral_band_sensitivity.png)",
        "",
        "图1：左为包括排除窗口的误差诊断，右为公共留出波段的误差分布。模型比较采用相同留出波段。0°/60°用于角度扰动，实际局地入射角需结合DEM计算。",
        "",
        "![图2](figures/cover_and_radius_band_sensitivity.png)",
        "",
        "图2：左检验模型比例对波段掩膜的敏感性；右显示低/高粒径边界是否堆积。改变掩膜后比例变化不自动等于改进。源数据为2013-05-03 AVIRIS，同一实验室背景字典和FCLS。",
        "",
        "候选模型和波段在评分前预定，所有结果按同一流程报告。所有候选的正反演算子一致，波段索引见band_masks.json。",
    ]
    (path / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    finish_run(path, metrics)
    return path


if __name__ == "__main__":
    run()
