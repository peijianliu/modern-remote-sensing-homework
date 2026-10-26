"""教学 Notebook 的轻量数据接口、统计与绘图。

本模块读取随案例提供的公开数据派生成果，不启动下载、MATLAB 或整景处理。
解混计算复用 linear 模块；地图读取 NumPy 数组与显式坐标元信息，无需 GIS 库。
"""

from __future__ import annotations

import json
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
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from .case_analysis import binary_agreement, fixed_radius_scan
from .linear import select_snow
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
    preferred = ["Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei"]
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
    """读取冻结正演并检查半径、波长、角度与光谱数组维度。"""
    data = loadmat(demo_directory(root) / "forward_models.mat")
    for key in ("radii_um", "wavelength_um", "angles_deg", "ice_options"):
        data[key] = data[key].ravel()
    expected = tuple(
        len(data[key]) for key in ("radii_um", "wavelength_um", "angles_deg", "ice_options")
    )
    for key in ("snicar_albedo", "ww_albedo"):
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid forward array: {key}")
    if not np.array_equal(data["radii_um"], np.arange(30, 1501, 10)):
        raise ValueError("The teaching cache requires the complete 30-1500 um radius grid")
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
                        "snicar_albedo": forward["broadband_snicar"][ri, ai, oi],
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
                forward["snicar_albedo"][ri, :, ai, 0],
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
    return fig


def plot_broadband_albedo(forward: dict) -> Figure:
    """比较粒径、角度和两种冰光学常数对加权反照率的影响。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharex=True)
    for color, angle in zip(COLORS, (0, 60)):
        ai = int(np.flatnonzero(forward["angles_deg"] == angle)[0])
        for oi, description in enumerate(("Warren 1984", "Warren–Brandt 2008")):
            axes[0].plot(
                forward["radii_um"],
                forward["broadband_snicar"][:, ai, oi],
                color=color,
                ls="-" if oi == 0 else "--",
                label=f"{angle}° · {description}",
            )
        difference = (
            forward["broadband_ww"][:, ai, 0] - forward["broadband_snicar"][:, ai, 0]
        ) * 100
        axes[1].plot(forward["radii_um"], difference, color=color, label=f"{angle}°")
    axes[0].set_ylabel(label("太阳谱加权反照率", "Solar-weighted albedo"))
    axes[1].set_ylabel(label("WW80 − SNICAR / 反射率百分点", "WW80 − SNICAR / reflectance pp"))
    axes[1].axhline(0, color=".4", ls=":", lw=0.8)
    for ax in axes:
        ax.set_xlabel(label("雪粒半径 / μm", "Snow radius / μm"))
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


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
        ylabel=label("阴影归一化雪比例 / %", "Shade-normalized snow fraction / %"),
        title=label("纯雪像元的覆盖估计", "Estimated cover of pure-snow pixels"),
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def load_aviris_case(root: Path | None = None) -> dict:
    """读取公开派生的空间数组和代表光谱，禁止 NumPy pickle 对象。"""
    with np.load(demo_directory(root) / "aviris_case.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def scan_representatives(case: dict) -> dict:
    """对相同的四条实景光谱重新计算五个固定粒径模型。"""
    with threadpool_limits(limits=1):
        return fixed_radius_scan(
            case["representative_spectra"],
            case["snow_spectra"],
            case["radii_um"],
            case["backgrounds"],
        )


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
        label("雪优势示例", "Snow-dominant example"),
        label("植被优势示例", "Vegetation-dominant example"),
        label("亮裸地示例", "Bright-bare example"),
        label("暗裸地示例", "Dark-bare example"),
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
    axes[0, 0].set(ylabel=label("归一化雪比例 / %", "Normalized snow fraction / %"))
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
    return fig


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
    return fig


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
    return fig


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
            label("物理雪库", "Physical snow library"),
            label("五种影像雪谱", "Five image snow spectra"),
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
        label=label("归一化雪比例", "Normalized snow fraction"),
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
    return fig
