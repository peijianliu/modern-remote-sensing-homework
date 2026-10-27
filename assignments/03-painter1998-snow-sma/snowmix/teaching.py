"""教学 Notebook 的轻量数据接口、统计与绘图。

本模块读取随案例提供的公开数据派生成果，不启动下载、MATLAB 或整景处理。
解混计算复用 linear 模块；地图读取 NumPy 数组与显式坐标元信息，无需 GIS 库。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .case_analysis import binary_agreement, fixed_radius_scan, paired_block_means
from .linear import select_snow
from .diagnostics import coefficient_status
from .optics import ww80_semi_infinite
from .spectra import gaussian_band_matrix, read_usgs_ascii

_CJK_READY = False
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")


def project_root(start: Path | None = None) -> Path:
    """根据源码或当前目录寻找案例根目录，兼容从 notebooks 子目录启动。"""
    candidates = [Path(__file__).resolve().parents[1]]
    location = Path(start or Path.cwd()).resolve()
    candidates += [location, *location.parents]
    for candidate in candidates:
        if (candidate / "data/demo").is_dir() and (candidate / "snowmix/linear.py").is_file():
            return candidate
    raise FileNotFoundError("Open the notebook from the case repository")


def demo_directory(root: Path | None = None) -> Path:
    """返回教学数据目录；缺少数据时给出可定位的相对路径。"""
    directory = (root or project_root()) / "data/demo"
    if not directory.is_dir():
        raise FileNotFoundError("Missing data/demo; use a complete repository checkout")
    return directory


def configure_plots() -> bool:
    """检测实际安装的中文字体；缺少时由 label 切换到英文绘图标签。"""
    global _CJK_READY
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK JP",
    ]
    chosen = next((font for font in preferred if font in available), None)
    _CJK_READY = chosen is not None
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen or "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "font.size": 10,
            "svg.fonttype": "none",
        }
    )
    return _CJK_READY


def label(chinese: str, english: str) -> str:
    """为图内文字选择有字形支持的语言；Notebook 正文始终使用中文。"""
    return chinese if _CJK_READY else english


def load_forward(root: Path | None = None) -> dict:
    """从散射参数现算 WW80 全网格及加权反照率；不读取 MATLAB 参考产物。

    返回 ww_albedo 的轴为 (148 radii, 480 wavelengths, 3 angles, 2 ice options)。
    solar_flux 为 SNICAR 中纬冬季直接入射太阳谱的独立输入，轴为 (angle, wavelength)。
    """
    optics = load_optical_inputs(root)
    with np.load(demo_directory(root) / "solar_illumination.npz", allow_pickle=False) as archive:
        angles = archive["angles_deg"].astype(np.float64)
        wavelength = archive["wavelength_um"]
        flux = archive["solar_flux"].astype(np.float64)
    if (
        flux.shape != (len(angles), len(wavelength))
        or not np.isfinite(flux).all()
        or np.any(flux < 0)
        or np.any(flux.sum(axis=1) <= 0)
        or not np.allclose(wavelength, optics["wavelength_um"], atol=1e-12, rtol=0)
    ):
        raise ValueError("Invalid solar illumination grid or flux")
    albedo = recompute_ww80(optics, angles)
    weights = flux.T[None, :, :, None]
    return {
        "radii_um": optics["radii_um"],
        "wavelength_um": wavelength,
        "ice_options": optics["ice_options"],
        "angles_deg": angles,
        "solar_flux": flux,
        "ww_albedo": albedo,
        "broadband_ww": np.sum(albedo * weights, axis=1) / np.sum(weights, axis=1),
    }


def load_forward_reference(root: Path | None = None) -> dict:
    """读取小型独立 MATLAB 参考：四种半径的光谱及完整半径网格的加权值。

    仅供校核 cell 使用，不能作为 load_forward 或合成混合谱的输入。
    """
    with np.load(demo_directory(root) / "forward_reference.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    expected = (4, 480, 3, 2)
    for key in ("ww_albedo", "snicar_albedo"):
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid independent reference: {key}")
    return data


def load_optical_inputs(root: Path | None = None) -> dict:
    """读取官方散射参数的教学子集；轴为半径、波长、冰光学选项。

    ss_alb 保留 float64，asm_prm 保存为 float32；计算前统一提升至 float64。
    这些参数从官方 NetCDF 提取，Mie 散射计算由上游数据库提供。
    """
    with np.load(demo_directory(root) / "scattering_inputs.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    for key, dtype in (("ss_alb", np.float64), ("asm_prm", np.float32)):
        if data[key].shape != (148, 480, 2) or data[key].dtype != dtype:
            raise ValueError(f"Invalid scattering input: {key}")
    return data


def recompute_ww80(optics: dict, angles_deg: np.ndarray) -> np.ndarray:
    """广播计算完整网格，输出轴为半径、波长、角度、冰光学选项。

    angles_deg 先转为 float64，避免 uint8 角度进入 deg2rad 的低精度路径。
    """
    mu = np.cos(np.deg2rad(np.asarray(angles_deg, dtype=np.float64)))
    return ww80_semi_infinite(
        optics["ss_alb"][:, :, None, :],
        optics["asm_prm"][:, :, None, :],
        mu[None, None, :, None],
    )


def optics_table(forward: dict) -> pd.DataFrame:
    """整理太阳谱加权反照率，保留半径、太阳天顶角和冰光学选项。"""
    rows = []
    for ai, angle in enumerate(forward["angles_deg"]):
        for oi, ice in enumerate(forward["ice_options"]):
            for ri, radius in enumerate(forward["radii_um"]):
                rows.append(
                    {
                        "radius_um": radius,
                        "solar_zenith_deg": angle,
                        "ice_option": int(ice),
                        "ww80_albedo": forward["broadband_ww"][ri, ai, oi],
                    }
                )
    return pd.DataFrame(rows)


def plot_optical_spectra(forward: dict) -> Figure:
    """在统一坐标范围下比较 0°、60° 的四种代表半径光谱。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, angle in zip(axes, (0, 60)):
        ai = int(np.flatnonzero(forward["angles_deg"] == angle)[0])
        for color, radius in zip(COLORS, (50, 200, 500, 1000)):
            ri = int(np.flatnonzero(forward["radii_um"] == radius)[0])
            ax.plot(
                forward["wavelength_um"],
                forward["ww_albedo"][ri, :, ai, 0],
                color=color,
                label=f"{radius} μm",
                lw=1.8,
            )
        ax.set(
            xlim=(0.3, 2.5),
            ylim=(0, 1.02),
            xlabel=label("波长 / μm", "Wavelength / μm"),
            title=label(f"太阳天顶角 {angle}°", f"Solar zenith {angle}°"),
        )
        ax.legend(title=label("雪粒半径", "Snow radius"), ncol=2, fontsize=8)
    axes[0].set_ylabel(label("方向—半球光谱反照率", "Directional-hemispherical albedo"))
    fig.suptitle(
        label("纯净球形厚雪层的光谱响应", "Spectral response of clean spherical, thick snow")
    )
    fig.tight_layout()
    return readable_figure(fig)


def plot_broadband_albedo(forward: dict) -> Figure:
    """比较粒径、角度和两种冰光学常数对加权反照率的影响。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharex=True)
    for color, angle in zip(COLORS, (0, 60)):
        ai = int(np.flatnonzero(forward["angles_deg"] == angle)[0])
        for oi, description in enumerate(("Warren 1984", "Warren–Brandt 2008")):
            axes[0].plot(
                forward["radii_um"],
                forward["broadband_ww"][:, ai, oi],
                color=color,
                ls="-" if oi == 0 else "--",
                label=f"{angle}° · {description}",
            )
        difference = (forward["broadband_ww"][:, ai, 1] - forward["broadband_ww"][:, ai, 0]) * 100
        axes[1].plot(forward["radii_um"], difference, color=color, label=f"{angle}°")
    axes[0].set_ylabel(label("太阳谱加权反照率", "Solar-weighted albedo"))
    axes[1].set_ylabel(label("WB08 − W84 / 反照率百分点", "WB08 − W84 / albedo pp"))
    axes[1].axhline(0, color=".4", ls=":", lw=0.8)
    for ax in axes:
        ax.set_xlabel(label("雪粒半径 / μm", "Snow radius / μm"))
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    return readable_figure(fig)


def synthetic_inputs(root: Path | None = None, angle_deg: float = 0, ice_option: int = 1) -> dict:
    """卷积到 224 个 AVIRIS 波段，构造 Fig. 3 所用 WW80 雪谱和石英谱。"""
    directory = demo_directory(root)
    forward = load_forward(root)
    metadata = json.loads((directory / "aviris_metadata.json").read_text(encoding="utf-8"))
    centers = np.asarray(metadata["wavelength_um"])
    fwhm = np.asarray(metadata["fwhm_um"])
    response = gaussian_band_matrix(forward["wavelength_um"], centers, fwhm)
    angle_index = np.flatnonzero(forward["angles_deg"] == angle_deg)
    ice_index = np.flatnonzero(forward["ice_options"] == ice_option)
    if len(angle_index) != 1 or len(ice_index) != 1:
        raise ValueError("Requested solar angle or optical option is absent from the cache")
    keep = forward["radii_um"] >= 50
    ai, oi = int(angle_index[0]), int(ice_index[0])
    source_w, source_r, _ = read_usgs_ascii(directory / "quartz_hs32.5854.asc")
    quartz = source_r @ gaussian_band_matrix(source_w, centers, fwhm)
    return {
        "wavelength_um": centers,
        "radii_um": forward["radii_um"][keep],
        "snow": forward["ww_albedo"][keep, :, ai, oi] @ response,
        "quartz": quartz,
        "solar_zenith_deg": angle_deg,
        "ice_option": ice_option,
    }


def run_fig3(inputs: dict) -> dict:
    """重新生成并解混 146×101 组混合谱，保留逐波段重建和残差。"""
    fractions = np.linspace(0, 1, 101)
    snow, radii, quartz = inputs["snow"], inputs["radii_um"], inputs["quartz"]
    observation = (
        fractions[None, :, None] * snow[:, None, :] + (1 - fractions[None, :, None]) * quartz
    ).reshape(-1, len(quartz))
    truth_fraction = np.tile(fractions, len(radii))
    truth_radius = np.repeat(radii, len(fractions))
    fixed_index = int(np.flatnonzero(radii == 200)[0])
    with threadpool_limits(limits=1):
        fixed = select_snow(observation, snow[fixed_index : fixed_index + 1], [200], quartz)
        multi = select_snow(observation, snow, radii, quartz)
    return {
        "inputs": inputs,
        "observation": observation,
        "fractions": fractions,
        "truth_fraction": truth_fraction,
        "truth_radius_um": truth_radius,
        "fixed": fixed,
        "multi": multi,
    }


def difference_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    """在共同有限值上计算 bias、MAE、RMSE；空样本返回 NaN。"""
    reference, estimate = np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(estimate)
    difference = estimate[valid] - reference[valid]
    return {
        "n": int(valid.sum()),
        "bias": float(np.mean(difference)) if len(difference) else np.nan,
        "mae": float(np.mean(np.abs(difference))) if len(difference) else np.nan,
        "rmse": float(np.sqrt(np.mean(difference**2))) if len(difference) else np.nan,
    }


def fig3_metrics(experiment: dict) -> pd.DataFrame:
    """分别报告全网格覆盖误差和纯雪端点偏差，比例误差单位为百分点。"""
    rows = []
    for method in ("fixed", "multi"):
        estimate = experiment[method].normalized[:, 0]
        metrics = difference_metrics(experiment["truth_fraction"], estimate)
        row = {
            "method": method,
            **{f"{key}_pp": value * 100 for key, value in metrics.items() if key != "n"},
            "n": metrics["n"],
        }
        for radius in (50, 1500):
            mask = (experiment["truth_radius_um"] == radius) & (experiment["truth_fraction"] == 1)
            row[f"pure_snow_{radius}um_estimate"] = float(estimate[mask][0])
            row[f"pure_snow_{radius}um_bias_pp"] = float((estimate[mask][0] - 1) * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fig3(experiment: dict) -> Figure:
    """绘制固定粒径的残差面与纯雪覆盖估计，保留大于 100% 的系数。"""
    radii = experiment["inputs"]["radii_um"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    error = experiment["fixed"].rms.reshape(len(radii), 101) * 100
    mesh = axes[0].pcolormesh(
        experiment["fractions"] * 100, radii, error, shading="nearest", cmap="viridis"
    )
    fig.colorbar(
        mesh, ax=axes[0], label=label("光谱 RMS / 反射率百分点", "Spectral RMS / reflectance pp")
    )
    axes[0].set(
        xlabel=label("真实雪比例 / %", "True snow fraction / %"),
        ylabel=label("真实雪粒半径 / μm", "True snow radius / μm"),
        title=label("固定 200 μm 雪端元的光谱误差", "Spectral error: fixed 200 μm snow"),
    )
    pure = experiment["truth_fraction"] == 1
    for key, color, description in (
        ("fixed", COLORS[0], "Fixed 200 μm"),
        ("multi", COLORS[1], "Minimum RMS"),
    ):
        axes[1].plot(
            radii, experiment[key].normalized[pure, 0] * 100, color=color, label=description
        )
    axes[1].axhline(100, color=".4", ls=":", lw=0.9)
    axes[1].set(
        xlabel=label("真实雪粒半径 / μm", "True snow radius / μm"),
        ylabel=label("去阴影雪比例比例 / %", "Shade-normalized snow fraction / %"),
        title=label("纯雪像元的覆盖估计", "Estimated cover of pure-snow pixels"),
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return readable_figure(fig)


def load_aviris_case(root: Path | None = None) -> dict:
    """读取公开派生的空间数组和代表光谱，禁止 NumPy pickle 对象。"""
    with np.load(demo_directory(root) / "aviris_case.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_coefficient_diagnostics(root: Path | None = None) -> dict:
    """读取三模型原始系数，逐像元重算去阴影比例，不分发重复的归一化数组。"""
    with np.load(
        demo_directory(root) / "coefficient_diagnostics.npz", allow_pickle=False
    ) as archive:
        data = dict(archive)
    denominator = 1 - data["shade"].astype(np.float64)
    data["normalized"] = np.full(data["fractions"].shape, np.nan, dtype=np.float64)
    np.divide(
        data["fractions"],
        denominator[..., None],
        out=data["normalized"],
        where=(denominator > 0.05)[..., None],
    )
    return data


def scan_representatives(case: dict) -> dict:
    """对相同的四条实景光谱重新计算五个固定粒径模型。"""
    with threadpool_limits(limits=1):
        return fixed_radius_scan(
            case["representative_spectra"],
            case["snow_spectra"],
            case["radii_um"],
            case["backgrounds"],
        )


def fixed_radius_statistics(case: dict, repeats: int = 1000, seed: int = 1998) -> tuple:
    """在五模型共同域上统计分母与均值，并以相同区块配对比较 1000−50 μm。

    输入为 load_aviris_case 的空间数组；比例无量纲、RMS 为反射率单位。
    固定 30 m 网格，以 ROI 左上角为原点划分 50×50 像元（1.5 km）区块。
    返回逐半径 DataFrame 和配对差值字典。区间只描述本场景、端元和掩膜。
    """
    common = case["common_normalization_mask"]
    if common.dtype != np.bool_ or not common.any():
        raise ValueError("A nonempty boolean common normalization mask is required")
    raw = case["fixed_raw_snow"][:, common].astype(np.float64)
    snow = case["fixed_snow_fraction"][:, common].astype(np.float64)
    rms = case["fixed_rms"][:, common].astype(np.float64)
    denominator = case["fixed_denominator"][:, common]
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.05):
        raise ValueError("Common-domain denominators must all exceed 0.05")
    np.testing.assert_allclose(raw / denominator, snow, rtol=2e-6, atol=2e-6)
    # 列顺序为五个雪比例，随后五个 RMS；相同 draws 保留方法间配对关系。
    result = paired_block_means(
        np.column_stack((snow.T, rms.T)), np.argwhere(common), repeats=repeats, seed=seed
    )
    rows = []
    for i, radius in enumerate(case["fixed_radii_um"]):
        rows.append(
            {
                "radius_um": float(radius),
                "common_n": int(common.sum()),
                "raw_snow_mean": float(raw[i].mean()),
                "normalized_snow_mean": float(snow[i].mean()),
                "normalized_snow_ci_low": result["interval_95"][0, i],
                "normalized_snow_ci_high": result["interval_95"][1, i],
                "rms_mean": float(rms[i].mean()),
                "rms_ci_low": result["interval_95"][0, i + len(snow)],
                "rms_ci_high": result["interval_95"][1, i + len(snow)],
                "denominator_mean": float(denominator[i].mean()),
                "denominator_p05": float(np.quantile(denominator[i], 0.05)),
                "denominator_p50": float(np.median(denominator[i])),
                "denominator_pass_fraction_common": float(np.mean(denominator[i] > 0.05)),
                "denominator_pass_fraction_input": float(
                    np.mean(case["fixed_denominator"][i][case["valid_mask"]] > 0.05)
                ),
                "negative_snow_fraction": float(np.mean(snow[i] < 0)),
            }
        )
    radii = np.asarray(case["fixed_radii_um"])
    first, last = (int(np.flatnonzero(radii == radius)[0]) for radius in (50, 1000))
    delta = {}
    for name, offset in (("normalized_snow", 0), ("rms", len(snow))):
        samples = (
            result["bootstrap_means"][:, last + offset]
            - result["bootstrap_means"][:, first + offset]
        )
        delta[name] = {
            "mean": float(result["mean"][last + offset] - result["mean"][first + offset]),
            "interval_95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return pd.DataFrame(rows), {
        key: result[key] for key in ("n", "spatial_blocks", "block_size_pixels", "repeats", "seed")
    } | {"comparison": "1000 um minus 50 um", "paired_delta": delta}


def representative_table(scan: dict) -> pd.DataFrame:
    """整理五个粒径、四个代表像元的原始系数、归一化比例和 RMS。"""
    return pd.DataFrame(
        [
            {
                "radius_um": radius,
                "representative": pixel,
                "raw_snow": scan["fractions"][ri, pixel, 0],
                "normalized_snow": scan["normalized"][ri, pixel, 0],
                "rms_reflectance": scan["rms"][ri, pixel],
            }
            for ri, radius in enumerate(scan["radii_um"])
            for pixel in range(scan["rms"].shape[1])
        ]
    )


def plot_spectrum(
    ax: Axes, wavelength_um: np.ndarray, reflectance: np.ndarray, **kwargs
) -> list[Line2D]:
    """在被剔除的宽波段间插入显示间断，保留原始输入和计算结果。"""
    wavelength = np.asarray(wavelength_um)
    values = np.asarray(reflectance)
    gaps = np.flatnonzero(np.diff(wavelength) > 0.04) + 1
    return ax.plot(np.insert(wavelength, gaps, np.nan), np.insert(values, gaps, np.nan), **kwargs)


def plot_representative_scan(case: dict, scan: dict) -> Figure:
    """同时展示实景雪比例扫描和原始系数重建的残差谱。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    names = [
        label("高雪指数示例", "High-NDSI example"),
        label("中等雪指数示例", "Intermediate-NDSI example"),
        label("高植被指数示例", "High-NDVI example"),
        label("低雪指数裸地示例", "Low-NDSI bare example"),
    ]
    if not case.get("heldout", False):
        names = [
            label("候选雪谱示例", "Candidate snow example"),
            label("植被端元自拟合", "Vegetation endmember control"),
            label("亮裸地端元自拟合", "Bright bare endmember control"),
            label("暗裸地端元自拟合", "Dark bare endmember control"),
        ]
    for pixel, (color, name) in enumerate(zip(COLORS, names)):
        axes[0, 0].plot(
            scan["radii_um"],
            scan["normalized"][:, pixel, 0] * 100,
            color=color,
            marker="o",
            label=name,
        )
        axes[0, 1].plot(
            scan["radii_um"], scan["rms"][:, pixel] * 100, color=color, marker="o", label=name
        )
    axes[0, 0].set(ylabel=label("去阴影雪比例 / %", "Normalized snow fraction / %"))
    axes[0, 1].set(ylabel=label("光谱 RMS / 反射率百分点", "Spectral RMS / reflectance pp"))
    for ax in axes[0]:
        ax.set_xlabel(label("固定雪粒半径 / μm", "Fixed snow radius / μm"))
        ax.legend(fontsize=7)
    pixel = 0
    plot_spectrum(
        axes[1, 0],
        case["wavelength_um"],
        case["representative_spectra"][pixel],
        color=".15",
        lw=2.2,
        label=label("观测", "Observed"),
    )
    for ri, (radius, color) in enumerate(zip(scan["radii_um"], COLORS)):
        plot_spectrum(
            axes[1, 0],
            case["wavelength_um"],
            scan["modeled"][ri, pixel],
            color=color,
            lw=1,
            label=f"{radius:g} μm",
        )
        plot_spectrum(
            axes[1, 1],
            case["wavelength_um"],
            scan["residual"][ri, pixel] * 100,
            color=color,
            lw=1,
            label=f"{radius:g} μm",
        )
    axes[1, 1].axhline(0, color=".3", lw=0.8)
    axes[1, 0].set_ylabel(label("反射率", "Reflectance"))
    axes[1, 1].set_ylabel(label("逐波段残差 / 反射率百分点", "Band residual / reflectance pp"))
    for ax in axes[1]:
        ax.set_xlabel(label("波长 / μm", "Wavelength / μm"))
        ax.legend(ncol=3, fontsize=6.5)
    fig.tight_layout()
    return readable_figure(fig)


def product_consistency(table: pd.DataFrame) -> pd.DataFrame:
    """在唯一 1 km 支持单元上重新计算两种雪标志的阈值敏感性。"""
    identity = ["roi", "tile_row_1km", "tile_column_1km"]
    if table.duplicated(identity).any():
        raise ValueError("Duplicated 1 km cells would inflate product-agreement sample counts")
    rows = []
    common = table["common_valid"].astype(str).str.lower().isin(["true", "1"])
    for roi, subset in table[common].groupby("roi", sort=True):
        for flag in ("flag12", "flag15"):
            for model in ("aviris_fsca", "modis_fsca"):
                for threshold in (0.1, 0.5, 0.9):
                    rows.append(
                        {
                            "roi": roi,
                            "flag": flag,
                            "model": model,
                            **binary_agreement(subset[flag], subset[model], threshold),
                        }
                    )
    return pd.DataFrame(rows)


def unique_native_units(table: pd.DataFrame) -> pd.DataFrame:
    """跨 ROI 合并同一原生状态单元；选择有效支持最多的记录。

    同一 ROI 内重复会报错。跨 ROI 的重叠边界按覆盖率降序、ROI 名升序
    确定保留记录，不使用雪比例或拟合误差择优。状态标志必须相同。
    """
    keys = ["tile_row_1km", "tile_column_1km"]
    if table.duplicated(["roi", *keys]).any():
        raise ValueError("Duplicated 1 km cells within one ROI")
    duplicated = table[table.duplicated(keys, keep=False)]
    for column in ("state_1km", "flag12", "flag15"):
        if column in table and (duplicated.groupby(keys)[column].nunique(dropna=False) > 1).any():
            raise ValueError("Overlapping ROIs disagree about the native MODIS state")
    return (
        table.sort_values(
            ["aviris_valid_area_fraction", "roi"], ascending=[False, True], kind="stable"
        )
        .drop_duplicates(keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )


def validation_ledger(table: pd.DataFrame, minimum_coverage: float = 0.95) -> pd.DataFrame:
    """追踪同一唯一 1 km 集合的逐级筛选与雪标志阳性数。

    Args:
        table: modis_1km_units.csv 的列；行键为 roi、tile_row_1km、tile_column_1km。
        minimum_coverage: 两种模型的有效面积占比阈值，范围 [0, 1]。

    Returns:
        四行账本：候选、有效状态、AVIRIS 面积支持、共同面积支持。
        n 为保留单元数，flag12_positive / flag15_positive 为其阳性数。

    Raises:
        ValueError: 重复空间单元或阈值无效。

    modis_valid_area_fraction 同时含影像 QA、AVIRIS 输入支持和模型归一化
    筛选的影响，账本只分解实际面积支持，不能单独分离每种影响。
    """
    if not np.isfinite(minimum_coverage) or not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must lie in [0, 1]")
    if table.duplicated(["roi", "tile_row_1km", "tile_column_1km"]).any():
        raise ValueError("Duplicated 1 km cells in validation ledger")
    table = unique_native_units(table)
    candidates = np.ones(len(table), dtype=bool)
    state = table.state_1km.notna().to_numpy() & (table.state_1km.to_numpy() != 65535)
    aviris = state & (table.aviris_valid_area_fraction.to_numpy() >= minimum_coverage)
    common = aviris & (table.modis_valid_area_fraction.to_numpy() >= minimum_coverage)
    rows = []
    for stage, mask in zip(
        ("candidates", "valid_state", "aviris_area", "common_area"),
        (candidates, state, aviris, common),
    ):
        subset = table.loc[mask]
        rows.append(
            {
                "stage": stage,
                "minimum_coverage": float(minimum_coverage),
                "n": int(mask.sum()),
                "flag12_positive": int(subset.flag12.eq(1).sum()),
                "flag15_positive": int(subset.flag15.eq(1).sum()),
            }
        )
    return pd.DataFrame(rows)


def state_flag_ledger(table: pd.DataFrame) -> pd.DataFrame:
    """分离产品状态 QA 与反射率解混支持，说明阳性在哪一环节消失。

    state QA 单独检查晴空、无云影、无卷云和内部云，最后排除邻云位。
    它不检查七波段 QC，也不将其统计冒充反射率解混比较的共同域。
    """
    table = unique_native_units(table)
    state = table.state_1km.to_numpy(np.uint16)
    support = (table.aviris_valid_area_fraction >= 0.95).to_numpy() & (state != 65535)
    clear = support & ((state & 3) == 0) & (((state >> 2) & 1) == 0)
    clear &= (((state >> 8) & 3) == 0) & (((state >> 10) & 1) == 0)
    isolated = clear & (((state >> 13) & 1) == 0)
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "n": int(mask.sum()),
                "flag12_positive": int(np.sum(mask & table.flag12.eq(1).to_numpy())),
                "flag15_positive": int(np.sum(mask & table.flag15.eq(1).to_numpy())),
            }
            for stage, mask in (
                ("aviris_support", support),
                ("clear_state", clear),
                ("no_adjacent_cloud", isolated),
            )
        ]
    )


def plot_modis_support(table: pd.DataFrame, spatial: dict) -> Figure:
    """原生正弦网格并列雪标志、面积支持和首个排除原因，显式显示未覆盖处。"""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    table = unique_native_units(table)
    rows, cols = table.tile_row_1km.to_numpy(), table.tile_column_1km.to_numpy()
    r0, c0, r1, c1 = rows.min(), cols.min(), rows.max() + 1, cols.max() + 1
    window = spatial["one_km_windows"][0]
    transform = window["transform"]
    wr0, _, wc0, _ = window["tile_window"]
    cell = transform[0]
    left, top = transform[2] + (c0 - wc0) * cell, transform[5] - (r0 - wr0) * cell
    extent = [
        left / 1000,
        (left + (c1 - c0) * cell) / 1000,
        (top - (r1 - r0) * cell) / 1000,
        top / 1000,
    ]

    def layer(value):
        array = np.full((r1 - r0, c1 - c0), np.nan)
        array[rows - r0, cols - c0] = value
        return array

    state = table.state_1km.to_numpy(np.uint16)
    clear = (
        (state != 65535)
        & ((state & 3) == 0)
        & (((state >> 2) & 1) == 0)
        & (((state >> 8) & 3) == 0)
        & (((state >> 10) & 1) == 0)
        & (((state >> 13) & 1) == 0)
    )
    reason = np.ones(len(table), int)
    supported = table.aviris_valid_area_fraction.to_numpy() >= 0.95
    reason[supported & ~clear] = 2
    reason[supported & clear] = 3
    reason[supported & clear & (table.modis_valid_area_fraction.to_numpy() >= 0.95)] = 4
    fig, axes_grid = plt.subplots(2, 2, figsize=(10.6, 8.4), layout="constrained")
    axes = axes_grid.ravel()
    flagmap = ListedColormap(["#f0f0f0", "#0072B2"])
    flagmap.set_bad("#bcbcbc")
    axes[0].imshow(
        layer(table.flag15), extent=extent, cmap=flagmap, vmin=0, vmax=1, interpolation="nearest"
    )
    axes[0].set_title(label("bit 15 雪标志", "bit 15 snow flag"))
    axes[0].legend(
        handles=[
            Patch(facecolor="#0072B2", label=label("阳性", "Positive")),
            Patch(facecolor="#f0f0f0", label=label("阴性", "Negative")),
        ],
        fontsize=7,
        loc="lower left",
    )
    for ax, key, title in zip(
        axes[1:3], ("aviris_valid_area_fraction", "modis_valid_area_fraction"), ("AVIRIS", "MODIS")
    ):
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#bcbcbc")
        image = ax.imshow(
            layer(table[key]), extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_title(title + label(" 有效面积比例", " valid area fraction"))
        fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.9)
    colors = ["#bcbcbc", "#d9d9d9", "#D55E00", "#E69F00", "#0072B2"]
    cmap = ListedColormap(colors)
    cmap.set_bad("#bcbcbc")
    axes[3].imshow(
        layer(reason),
        extent=extent,
        cmap=cmap,
        norm=BoundaryNorm(np.arange(6) - 0.5, 5),
        interpolation="nearest",
    )
    axes[3].set_title(label("首个排除原因", "First exclusion condition"))
    names = [
        label("无候选记录", "No candidate"),
        label("AVIRIS支持不足", "AVIRIS support"),
        label("云/邻云状态", "Cloud / adjacent"),
        label("MODIS输出支持不足", "MODIS output support"),
        label("共同有效", "Common valid"),
    ]
    axes[3].legend(
        handles=[Patch(facecolor=c, label=n) for c, n in zip(colors, names)],
        fontsize=6.5,
        loc="lower left",
    )
    for ax in axes:
        ax.set(xlabel=label("正弦投影 x / km", "Sinusoidal x / km"))
        ax.grid(False)
        ax.set_aspect("equal")
    axes[0].set_ylabel(label("正弦投影 y / km", "Sinusoidal y / km"))
    fig.suptitle(
        label(
            "2013-05-03 MOD09GA与AVIRIS；原生正弦网格；有效面积阈值95%",
            "2013-05-03 MOD09GA and AVIRIS; native sinusoidal grid; area threshold 95%",
        ),
        fontsize=10,
    )
    return readable_figure(fig)


def coverage_sensitivity(table: pd.DataFrame, thresholds: tuple = (0.8, 0.9, 0.95)) -> pd.DataFrame:
    """保持同一唯一 1 km 候选集合，比较有效面积阈值的样本与阳性保留情况。

    返回每阈值两行（aviris_area / common_area）；候选与状态有效数量由
    validation_ledger 单独展示。面积阈值和判定有雪的雪比例阈值含义不同。
    """
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Coverage thresholds must be nonempty and unique")
    return pd.concat(
        [
            validation_ledger(table, threshold).query("stage in ['aviris_area', 'common_area']")
            for threshold in thresholds
        ],
        ignore_index=True,
    )


def plot_validation_design(ledger: pd.DataFrame) -> Figure:
    """分别显示空间样本保留和参照阳性减少，避免把阴性单元画成无样本。"""
    selected = ledger.set_index("stage").loc[["candidates", "aviris_area", "common_area"]]
    thresholds = selected.minimum_coverage.unique()
    if len(thresholds) != 1:
        raise ValueError("Ledger stages must use one coverage threshold")
    percent = f"{thresholds[0] * 100:g}%"
    names = [
        label("候选", "Candidates"),
        f"AVIRIS ≥{percent}",
        label(f"共同 ≥{percent}", f"Both ≥{percent}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = axes[0].bar(names, selected.n, color=COLORS[:3])
    axes[0].bar_label(bars, padding=4)
    axes[0].set_ylim(0, selected.n.max() * 1.15)
    axes[0].set_ylabel(label("唯一 1 km 单元数", "Unique 1 km units"))
    axes[0].set_title(label("同一集合的筛选账本", "Filtering the same spatial units"))
    for column, color, title in (
        ("flag12_positive", COLORS[0], "bit 12"),
        ("flag15_positive", COLORS[1], "bit 15"),
    ):
        axes[1].plot(names, selected[column], "o-", color=color, label=title)
        for x, count in enumerate(selected[column]):
            axes[1].annotate(
                str(count),
                (x, count),
                xytext=(4, 5 if column.endswith("15_positive") else -14),
                textcoords="offset points",
                fontsize=9,
            )
    axes[1].set_ylim(-10, selected.flag15_positive.max() * 1.2)
    axes[1].set_ylabel(label("雪标志阳性数", "Positive snow flags"))
    axes[1].set_title(label("阳性类别的保留情况", "Retention of the positive class"))
    axes[1].legend()
    fig.tight_layout()
    return readable_figure(fig)


def worldview_metrics(directory: Path | None = None) -> pd.DataFrame:
    """由保留的独立参考样本逐行重算共同有效掩膜下的分层覆盖误差。"""
    directory = directory or demo_directory()
    methods = ("fixed200_six_bands", "multi_six_bands", "fixed200_four_bands", "multi_four_bands")
    rows = []
    paths = sorted(directory.rglob("*_sample_pixels.csv"))
    paths += sorted(directory.rglob("*_sample_pixels.csv.gz"))
    scenes = [path.name.removesuffix(".gz").removesuffix("_sample_pixels.csv") for path in paths]
    if len(scenes) != len(set(scenes)):
        raise ValueError("Duplicate scene CSV/GZ files would count samples twice")
    for path, scene in zip(paths, scenes):
        samples = pd.read_csv(path)
        common = (
            samples["common_four_methods_normalization_valid"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        for method in methods:
            for stratum in ("all", "no_snow", "mixed", "full_snow"):
                keep = common if stratum == "all" else common & (samples["stratum"] == stratum)
                metrics = difference_metrics(
                    samples.loc[keep, "reference_fsca"], samples.loc[keep, method + "_snow"]
                )
                rows.append({"scene": scene, "method": method, "stratum": stratum, **metrics})
    if not rows:
        raise FileNotFoundError("Missing WorldView sample CSV files in data/demo")
    return pd.DataFrame(rows)


def plot_worldview_strata(metrics: pd.DataFrame) -> Figure:
    """对比总体与混合雪误差，并在图中保留样本数量。"""
    scenes = sorted(metrics.scene.unique(), key=lambda scene: scene.split("_")[3])
    fig, axes = plt.subplots(1, len(scenes), figsize=(11, 4), squeeze=False, sharey=True)
    for ax, scene in zip(axes.ravel(), scenes):
        subset = metrics[(metrics.scene == scene) & (metrics.method == "multi_six_bands")]
        selected = subset.set_index("stratum").loc[["all", "no_snow", "mixed", "full_snow"]]
        bars = ax.bar(
            np.arange(4), selected.rmse * 100, color=[COLORS[0], "#9c9c9c", COLORS[1], COLORS[2]]
        )
        ax.set_xticks(
            np.arange(4),
            [
                label("全部", "All"),
                label("无雪", "No snow"),
                label("混合雪", "Mixed"),
                label("近纯雪", "Near pure"),
            ],
        )
        ax.set_title("2016-02-13" if "20160213" in scene else "2017-05-21 / 22")
        for bar, n in zip(bars, selected.n):
            ax.annotate(
                f"n={n}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.margins(y=0.2)
    axes[0, 0].set_ylabel(label("覆盖 RMSE / 百分点", "Cover RMSE / percentage points"))
    fig.tight_layout()
    return readable_figure(fig)


def plot_spatial_case(case: dict, spatial_metadata: dict) -> Figure:
    """用真实投影坐标显示影像、两种雪端元模型及其差异。"""
    transform = spatial_metadata["aviris_transform"]
    a, _, c, _, e, f = transform[:6]
    shape = case["valid_mask"].shape
    extent = np.asarray([c, c + a * shape[1], f + e * shape[0], f]) / 1000
    fig = plt.figure(figsize=(14, 5.2))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1, 0.055],
        hspace=0.4,
        left=0.06,
        right=0.99,
        bottom=0.21,
        top=0.9,
        wspace=0.15,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    axes[0].imshow(case["rgb"], extent=extent)
    axes[0].set_title(label("AVIRIS 真彩色", "AVIRIS true color"))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    cmap.set_under("#D55E00")
    cmap.set_over("#CC79A7")
    for ax, key, title in zip(
        axes[1:3],
        ("physical_snow_fraction", "image_snow_fraction"),
        (
            label("物理雪库 MGS（诊断）", "Physical MGS (diagnostic)"),
            label("影像雪谱 MGS（诊断）", "Image MGS (diagnostic)"),
        ),
    ):
        image = ax.imshow(
            np.where(case["valid_mask"], case[key], np.nan),
            extent=extent,
            vmin=0,
            vmax=1,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(title)
    fig.colorbar(
        image,
        cax=fig.add_subplot(grid[1, 1:3]),
        orientation="horizontal",
        label=label("去阴影雪比例", "Normalized snow fraction"),
        extend="both",
    )
    difference = case["image_snow_fraction"] - case["physical_snow_fraction"]
    difference_cmap = plt.get_cmap("PuOr").copy()
    difference_cmap.set_bad("#d9d9d9")
    delta = axes[3].imshow(
        np.where(case["valid_mask"], difference, np.nan),
        extent=extent,
        vmin=-0.3,
        vmax=0.3,
        cmap=difference_cmap,
        interpolation="nearest",
    )
    axes[3].set_title(label("影像谱 − 物理库", "Image − physical"))
    fig.colorbar(
        delta,
        cax=fig.add_subplot(grid[1, 3]),
        orientation="horizontal",
        label=label("比例差", "Fraction difference"),
        extend="both",
    )
    for ax in axes:
        ax.set_xlabel(label("东坐标 / km", "Easting / km"))
        ax.grid(False)
        ax.set_aspect("equal")
    axes[0].set_ylabel(label("北坐标 / km", "Northing / km"))
    axes[0].legend(
        handles=[Patch(facecolor="#d9d9d9", label="NoData")], loc="upper left", fontsize=7
    )
    # 数据投影单位为米；1 km 比例尺直接使用图中千米坐标。
    axes[0].plot([extent[0] + 0.4, extent[0] + 1.4], [extent[2] + 0.4] * 2, color="white", lw=3)
    axes[0].text(extent[0] + 0.9, extent[2] + 0.5, "1 km", color="white", ha="center", fontsize=8)
    fig.text(
        0.06,
        0.02,
        label(
            "来源 JPL AVIRIS，2013-05-03；WGS84 / UTM 11N；灰色为无效区域。",
            "Source: JPL AVIRIS, 2013-05-03; WGS84 / UTM 11N; gray: NoData.",
        ),
        fontsize=8,
    )
    return readable_figure(fig)


def readable_table(table: pd.DataFrame, columns: dict | None = None, precision: int = 3):
    """为中文教学展示选列、改列名、隐藏行号；原始 CSV 和数组保持原值。"""
    displayed = table.copy()
    if "scene" in displayed:

        def date_label(value):
            match = re.search(r"20\d{6}", str(value))
            if match is None:
                return str(value)
            date = match[0]
            return f"{date[:4]}-{date[4:6]}-{date[6:]}"

        displayed["scene"] = displayed["scene"].map(date_label)
    labels = {
        "all": "总体",
        "no_snow": "无雪",
        "mixed": "混合雪",
        "full_snow": "近纯雪",
        "low_mixed": "低至中等雪比例",
        "snow_dominated": "雪占优势",
        "strict7": "严格七波段",
        "selected5": "五波段对照",
        "physical_MGS": "物理 MGS",
        "image_MGS": "影像雪谱 MGS",
        "physical_FCLS": "物理 FCLS",
        "high_ndsi": "高 NDSI",
        "high_ndvi": "高 NDVI",
        "other": "其他",
        "candidates": "候选原生单元",
        "valid_state": "有效状态",
        "aviris_area": "AVIRIS 面积",
        "common_area": "共同面积",
        "aviris_support": "AVIRIS 支持",
        "clear_state": "晴空状态",
        "no_adjacent_cloud": "再排除邻云",
    }
    for column in ("stratum", "tier", "method", "spectral_group", "stage"):
        if column in displayed:
            displayed[column] = displayed[column].map(lambda x: labels.get(x, x))
    if columns is not None:
        displayed = displayed[list(columns)].rename(columns=columns)
    return displayed.style.hide(axis="index").format(precision=precision, na_rep="—", thousands=",")


def plot_coefficient_status(diagnostic: dict, spatial: dict) -> Figure:
    """并列无约束雪系数、全部系数可行性与 FCLS 去阴影雪比例。

    状态图按原始四端元加阴影共同判定；超界保留显式类别，灰色表示无数据。
    所有面板使用相同 UTM 11N 范围，投影坐标以 km 展示。
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    mask = diagnostic["valid_mask"]
    height, width = mask.shape
    a, _, left, _, e, top = spatial["aviris_transform"]
    extent = [left / 1000, (left + a * width) / 1000, (top + e * height) / 1000, top / 1000]

    def layer(value):
        result = np.full(mask.shape, np.nan)
        result[mask] = value
        return result

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7), layout="constrained")
    categories = ListedColormap(["#d9d9d9", "#0072B2", "#D55E00", "#CC79A7", "#E69F00"])
    categories.set_bad("#d9d9d9")
    for index, name in enumerate(("physical MGS", "image MGS")):
        raw = diagnostic["fractions"][index]
        status = coefficient_status(raw, diagnostic["shade"][index], tolerance=1e-7)
        cmap = plt.get_cmap("PuOr").copy()
        cmap.set_bad("#d9d9d9")
        signed = axes[0, index].imshow(
            layer(raw[:, 0]), extent=extent, cmap=cmap, vmin=-0.5, vmax=0.5, interpolation="nearest"
        )
        axes[0, index].set_title(label("物理雪库 MGS" if index == 0 else "影像雪谱 MGS", name))
        fig.colorbar(
            signed,
            ax=axes[0, index],
            orientation="horizontal",
            extend="both",
            label=label("原始雪系数（保留符号）", "Signed raw snow coefficient"),
            shrink=0.9,
        )
        axes[1, index].imshow(
            layer(status["status"]),
            extent=extent,
            cmap=categories,
            norm=BoundaryNorm(np.arange(6) - 0.5, 5),
            interpolation="nearest",
        )
        axes[1, index].set_title(label("全部端元及阴影的可行性", "All coefficients and shade"))
    fcls = diagnostic["normalized"][2, :, 0]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    physical = axes[0, 2].imshow(
        layer(fcls), extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
    )
    axes[0, 2].set_title(label("物理雪库 FCLS", "Physical FCLS"))
    fig.colorbar(
        physical,
        ax=axes[0, 2],
        orientation="horizontal",
        label=label("去阴影雪比例", "Normalized snow fraction"),
        shrink=0.9,
    )
    axes[1, 2].axis("off")
    names = [
        label("无解 / NoData", "Unsolved / NoData"),
        label("全部系数可行", "All feasible"),
        label("雪系数 < 0", "Snow < 0"),
        label("雪系数 > 1", "Snow > 1"),
        label("背景或阴影越界", "Background or shade outside [0,1]"),
    ]
    axes[1, 2].legend(
        handles=[Patch(facecolor=c, label=n) for c, n in zip(categories.colors, names)],
        loc="center",
        frameon=False,
        fontsize=10,
    )
    for ax in axes.ravel()[:]:
        if ax.axison:
            ax.set(
                xlabel=label("东坐标 / km", "Easting / km"),
                ylabel=label("北坐标 / km", "Northing / km"),
            )
            ax.set_aspect("equal")
            ax.grid(False)
    fig.suptitle(
        label(
            "2013-05-03 AVIRIS；WGS84 / UTM 11N；同一输入与范围",
            "2013-05-03 AVIRIS; WGS84 / UTM 11N; identical input and extent",
        ),
        fontsize=11,
    )
    return readable_figure(fig)


def plot_mismatch_summary(summary: pd.DataFrame) -> Figure:
    """把覆盖 RMSE 和光谱 RMS 分开画，突出它们评价不同对象。"""
    conditions = [
        "matched",
        "noise_005",
        "noise_010",
        "off_grid_radius",
        "unseen_background",
        "zenith_60_vs_0",
        "spectral_tilt",
    ]
    labels = [
        label(c, e)
        for c, e in zip(
            ["匹配", "噪声 .005", "噪声 .01", "半径网格外", "背景未入库", "角度失配", "谱形扰动"],
            ["Matched", "Noise .005", "Noise .01", "Off-grid", "Unseen bg.", "Angle", "Tilt"],
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for color, method in zip(COLORS, ("fixed200_MGS", "multi_MGS", "multi_FCLS")):
        selected = summary[summary.method == method].set_index("condition").loc[conditions]
        for ax, key in zip(axes, ("rmse_pp", "spectral_rms_pp")):
            ax.plot(np.arange(7), selected[key], "o-", color=color, label=method)
    for ax in axes:
        ax.set_xticks(np.arange(7), labels, rotation=30, ha="right")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(label("已知覆盖 RMSE / pp", "Known-cover RMSE / pp"))
    axes[1].set_ylabel(label("光谱 RMS / 反射率百分点", "Spectral RMS / reflectance pp"))
    return readable_figure(fig)


def plot_transfer_comparison(metrics: pd.DataFrame) -> Figure:
    """在相同混合雪像元上并列固定、多粒径和两种求解约束。"""
    scenes = sorted(metrics.scene.unique(), key=lambda x: x.split("_")[3])
    methods = ["fixed200_six_MGS", "multi_six_MGS", "fixed200_six_FCLS", "multi_six_FCLS"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), sharey=True, layout="constrained")
    for ax, scene in zip(axes, scenes):
        rows = (
            metrics[(metrics.scene == scene) & (metrics.stratum == "mixed")]
            .set_index("method")
            .loc[methods]
        )
        bars = ax.bar(np.arange(4), rows.rmse_pp, color=COLORS[:4])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xticks(np.arange(4), ["200 MGS", "Multi MGS", "200 FCLS", "Multi FCLS"], rotation=20)
        ax.set_title(f"{scene.split('_')[3]} · n={int(rows.n.iloc[0]):,}")
        ax.set_ylim(0, float(rows.rmse_pp.max()) * 1.2)
    axes[0].set_ylabel(label("混合雪覆盖 RMSE / pp", "Mixed-snow cover RMSE / pp"))
    return readable_figure(fig)


def readable_figure(figure: Figure, minimum_fontsize: float = 12) -> Figure:
    """提高图内文字的最小字号，并在普通布局中重新安排边距；不改变数据。"""
    from matplotlib.text import Text
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    for item in figure.findobj(Text):
        item.set_fontsize(max(item.get_fontsize(), minimum_fontsize))
    if not isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine):
        figure.tight_layout(pad=1.1)
    return figure


def plot_modis_external(metrics: pd.DataFrame) -> Figure:
    """并列两景严格七波段结果的总体与混合雪RMSE，固定纵轴并标样本数。"""
    scenes = sorted(metrics.scene.astype(str).unique())
    methods = ["fixed200_MGS", "multi_MGS", "fixed200_FCLS", "multi_FCLS"]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharey=True, layout="constrained")
    for column, scene in enumerate(scenes):
        for row, stratum in enumerate(("all", "mixed")):
            ax = axes[row, column]
            data = (
                metrics[
                    (metrics.scene.astype(str) == scene)
                    & (metrics.tier == "strict7")
                    & (metrics.stratum == stratum)
                ]
                .set_index("method")
                .loc[methods]
            )
            bars = ax.bar(np.arange(4), data.rmse_pp, color=COLORS[:4])
            ax.bar_label(bars, fmt="%.2f", padding=3)
            ax.set_xticks(
                np.arange(4), ["200 MGS", "Multi MGS", "200 FCLS", "Multi FCLS"], rotation=20
            )
            date = f"{scene[:4]}-{scene[4:6]}-{scene[6:]}"
            group = label(
                "总体" if stratum == "all" else "混合雪", "All" if stratum == "all" else "Mixed"
            )
            ax.set_title(f"{date} · {group} · n={int(data.n.iloc[0]):,}")
            ax.set_ylim(0, 42)
    for ax in axes[:, 0]:
        ax.set_ylabel(label("覆盖 RMSE / pp", "Cover RMSE / pp"))
    return readable_figure(fig, 13)
