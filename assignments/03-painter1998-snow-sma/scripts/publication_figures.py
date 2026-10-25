"""将已保存实验结果绘制为带图注的科研图件。

在项目根目录运行 ``.venv/Scripts/python.exe scripts/publication_figures.py``。
每次运行新建输出目录，以原实验数组、表格和指标作为输入。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio

ROOT = Path(__file__).resolve().parents[1]
NODATA_COLOR = "#d9d9d9"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
METHOD_COLORS = {"fixed_original": "#6c6c6c", "multi_original": "#0072B2", "multi_fcls": "#D55E00"}
METHOD_LABELS = {
    "fixed_original": "固定 200 μm / MGS",
    "multi_original": "多粒径 / MGS",
    "multi_fcls": "多粒径 / FCLS",
}
LOG = logging.getLogger(__name__)


def read_json(path: Path) -> dict[str, Any]:
    """以 UTF-8 编码读取 JSON 对象。"""
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> str:
    """以分块读取计算文件哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, content: Any) -> None:
    """以 UTF-8 编码新建 JSON 文件，路径已存在时拒绝覆盖。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def configure_style() -> None:
    """设置 PNG 和可编辑 SVG 的字体、线型及颜色。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "lines.linewidth": 1.9,
            "axes.titlepad": 10,
        }
    )


class FigureBundle:
    """汇总一次制图的图注和输入文件哈希。"""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.sources: dict[str, str] = {}
        self.catalog: list[dict[str, Any]] = []
        self.verifications: list[dict[str, Any]] = []

    def source(self, path: Path) -> Path:
        """登记输入文件及其哈希，重复文件只记录一次。"""
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        if relative not in self.sources:
            self.sources[relative] = hash_file(path)
        return path

    def csv(self, path: Path) -> pd.DataFrame:
        """登记来源并读取数值表。"""
        return pd.read_csv(self.source(path))

    def json(self, path: Path) -> dict[str, Any]:
        """登记来源并读取元数据。"""
        return read_json(self.source(path))

    def npz(self, path: Path) -> dict[str, np.ndarray]:
        """读取已保存的数值数组。"""
        with np.load(self.source(path)) as archive:
            return {name: archive[name] for name in archive.files}

    def save(
        self,
        fig: plt.Figure,
        name: str,
        title: str,
        caption: str,
        inputs: list[Path],
        limits: str = "",
    ) -> None:
        """导出 PNG、SVG 两种格式，并记录完整图注。"""
        for path in inputs:
            self.source(path)
        for extension in ("png", "svg"):
            target = self.output / "figures" / f"{name}.{extension}"
            if target.exists():
                raise FileExistsError(target)
            fig.savefig(target, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        self.catalog.append(
            {
                "id": name,
                "title": title,
                "png": str((self.output / "figures" / f"{name}.png").relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "svg": str((self.output / "figures" / f"{name}.svg").relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "caption": caption,
                "limits": limits,
                "sources": [str(p.relative_to(ROOT)).replace("\\", "/") for p in inputs],
            }
        )
        LOG.info("Saved %s", name)


def footer(fig: plt.Figure, text: str) -> None:
    """在图内添加简要的数据来源和方法说明。"""
    fig.text(0.5, 0.018, text, ha="center", va="bottom", fontsize=8, color="#555555")


def load_raster(
    bundle: FigureBundle, path: Path, band: int = 1
) -> tuple[np.ndarray, tuple[float, ...], str]:
    """读取地图子图的栅格值与投影坐标范围。"""
    with rasterio.open(bundle.source(path)) as dataset:
        array = dataset.read(band, masked=True).astype(float).filled(np.nan)
        bounds = dataset.bounds
        extent = (bounds.left / 1000, bounds.right / 1000, bounds.bottom / 1000, bounds.top / 1000)
        crs = dataset.crs.to_wkt()
    return array, extent, crs


def north_arrow(ax: plt.Axes, extent: tuple[float, ...], crs: str) -> None:
    """按地图中心位置计算真北方向，考虑正弦投影的方向偏转。"""
    x = (extent[0] + extent[1]) * 500
    y = (extent[2] + extent[3]) * 500
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    lon, lat = inverse.transform(x, y)
    xn, yn = forward.transform(lon, lat + 0.01)
    vector = np.array([(xn - x) / (extent[1] - extent[0]), (yn - y) / (extent[3] - extent[2])])
    vector = 0.12 * vector / np.linalg.norm(vector)
    anchor = np.array([0.84, 0.76])
    end = anchor + vector
    ax.annotate(
        "",
        xy=end,
        xytext=anchor,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#222222"},
    )
    ax.text(
        end[0],
        end[1] + 0.025,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
    )


def scale_bar(ax: plt.Axes, extent: tuple[float, ...], length_km: float = 2) -> None:
    """在投影坐标中按实际长度绘制比例尺。"""
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]
    x, y = extent[0] + 0.08 * width, extent[2] + 0.09 * height
    ax.add_patch(
        Rectangle(
            (x - 0.04 * width, y - 0.045 * height),
            length_km + 0.08 * width,
            0.13 * height,
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
            zorder=5,
        )
    )
    ax.plot([x, x + length_km], [y, y], color="#222222", lw=2, zorder=6)
    ax.plot([x, x], [y - 0.012 * height, y + 0.012 * height], color="#222222", lw=1, zorder=6)
    ax.plot(
        [x + length_km, x + length_km],
        [y - 0.012 * height, y + 0.012 * height],
        color="#222222",
        lw=1,
        zorder=6,
    )
    ax.text(
        x + length_km / 2,
        y + 0.025 * height,
        f"{length_km:g} km",
        ha="center",
        fontsize=8,
        zorder=6,
    )


def map_axes(
    ax: plt.Axes,
    array: np.ndarray,
    extent: tuple[float, ...],
    crs: str,
    title: str,
    cmap: Any,
    vmin: float,
    vmax: float,
    norm: Any = None,
) -> Any:
    """在固定空间范围内以最近邻方式显示原始像元。"""
    ax.grid(False)
    kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    image = ax.imshow(
        array, extent=extent, origin="upper", interpolation="nearest", cmap=cmap, **kwargs
    )
    ax.set_title(title)
    ax.set_xlabel("Easting / km")
    ax.set_ylabel("Northing / km")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(MaxNLocator(4 if abs(extent[0]) < 1000 else 3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    north_arrow(ax, extent, crs)
    scale_bar(ax, extent)
    return image


def sequential_cmap(name: str, overflow: bool = False) -> Any:
    """使用固定色带，并区分缺测值与超出显示范围的数值。"""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(NODATA_COLOR)
    if overflow:
        cmap.set_under("#8c6bb1")
        cmap.set_over("#b35806")
    return cmap


def add_nodata_legend(fig: plt.Figure, extra: list[Patch] | None = None) -> None:
    """用单独图例标明缺测值，使其与零值区分。"""
    handles = [Patch(facecolor=NODATA_COLOR, label="未报告 / NoData")]
    handles.extend(extra or [])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.051),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )


def physics_figures(bundle: FigureBundle, source: Path) -> None:
    """绘制物理正演光谱及预定的 0°、60° 对照。"""
    spectra_path = source / "tables/spectral_curves.csv"
    spectra = bundle.csv(spectra_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True, sharey=True)
    for ax, angle in zip(axes, (0, 60)):
        for color, radius, style in zip(COLORS, (50, 200, 500, 1000), ("-", "--", "-.", ":")):
            group = spectra[(spectra.angle_deg == angle) & (spectra.radius_um == radius)]
            ax.plot(
                group.wavelength_um, group.albedo, color=color, ls=style, label=f"r = {radius} μm"
            )
        ax.set(
            title=f"({chr(97 + (angle == 60))}) 太阳天顶角 {angle}°",
            xlim=(0.35, 2.5),
            ylim=(0, 1.03),
            xlabel="波长 / μm",
        )
        ax.axvspan(0.4, 0.7, color="#eeeeee", alpha=0.55, zorder=0)
        ax.text(0.55, 0.07, "可见光", ha="center", fontsize=8, color="#666666")
    axes[0].set_ylabel("方向—半球光谱反射率（无量纲）")
    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle("雪粒径主要改变近红外与短波红外反射率", fontsize=15, y=0.99)
    fig.subplots_adjust(bottom=0.17, top=0.84, wspace=0.08)
    footer(fig, "SNICAR-ADv3 / Warren 1984 · 球形、无杂质、100 m 厚雪 · 原数值重绘")
    bundle.save(
        fig,
        "01_snow_spectra_angles",
        "0°与60°的雪粒径光谱对照",
        "四条曲线的光学等效半径分别为50、200、500、1000 μm。左、右保持同一坐标轴。可见光差异较小；进入近红外和短波红外后，较粗雪粒曲线通常更低。60°表示太阳天顶角。",
        [spectra_path],
        "正演输出为纯净球形厚雪层的方向—半球反射率，实景BRDF、雪形与杂质差异列为后续因素。",
    )

    albedo_path = source / "tables/broadband_albedo.csv"
    table = bundle.csv(albedo_path)
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for angle, color in ((0, COLORS[0]), (60, COLORS[1])):
        for option, style in (("Warren 1984", "-"), ("Warren–Brandt 2008", "--")):
            group = table[(table.angle_deg == angle) & (table.ice_option == option)]
            ax.plot(
                group.radius_um, group.albedo, color=color, ls=style, label=f"{angle}° / {option}"
            )
    for radius, label, vertical in ((100, "新雪代表", 0.71), (1000, "老雪代表", 0.90)):
        ax.axvline(radius, color="#888888", lw=1, ls=":")
        ax.annotate(
            f"{label}\n{radius} μm",
            (radius, vertical),
            ha="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=.25", "fc": "white", "ec": "#cccccc"},
        )
    ax.set(
        xscale="log",
        xlim=(30, 1500),
        ylim=(float(table.albedo.min()) - 0.025, 0.94),
        xlabel="光学等效雪粒半径 / μm（对数轴）",
        ylabel="太阳谱加权反照率（无量纲）",
        title="粒径增加使纯净厚雪层的宽波段反照率降低",
    )
    ax.set_xticks(
        [30, 50, 100, 200, 500, 1000, 1500],
        labels=["30", "50", "100", "200", "500", "1000", "1500"],
    )
    ax.legend(loc="lower left", fontsize=8)
    fig.subplots_adjust(bottom=0.16, top=0.89)
    footer(fig, "0.2–5 μm 积分 · 中纬度冬季太阳谱 · 线型区分光学常数，颜色区分角度")
    values = table[(table.angle_deg == 60) & (table.ice_option == "Warren 1984")].set_index(
        "radius_um"
    )
    bundle.save(
        fig,
        "02_radius_broadband_albedo",
        "雪粒径与加权反照率",
        f"在60°、Warren1984条件下，100 μm与1000 μm反照率分别为{values.loc[100, 'albedo']:.4f}、{values.loc[1000, 'albedo']:.4f}。实线/虚线对照两套冰光学常数。",
        [albedo_path],
        "新雪、老雪用代表半径标记，标准光学库覆盖30–1500 μm。",
    )


def paper_figures(bundle: FigureBundle, source: Path, angle60: Path) -> None:
    """绘制 Fig. 3 对照结果，标明单位、范围和历史输入差异。"""
    path = source / "tables/fig3_grid.csv"
    table = bundle.csv(path)
    radii = np.sort(table.true_radius_um.unique())
    fractions = np.sort(table.true_snow_fraction.unique())
    grid = (
        table.pivot(index="true_radius_um", columns="true_snow_fraction", values="fixed_rms")
        .sort_index()
        .to_numpy()
        * 100
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.3), gridspec_kw={"width_ratios": [1.1, 1]})
    image = axes[0].pcolormesh(
        fractions * 100,
        radii,
        grid,
        shading="nearest",
        cmap="magma",
        vmin=0,
        vmax=float(grid.max()),
        rasterized=True,
    )
    axes[0].axhline(200, ls="--", lw=1, color="white")
    axes[0].text(3, 240, "匹配 200 μm", color="white", fontsize=9)
    axes[0].set(
        title="(a) 固定雪端元的光谱误差", xlabel="真实雪覆盖 / %", ylabel="真实雪粒半径 / μm"
    )
    fig.colorbar(image, ax=axes[0], pad=0.02, label="光谱 RMS / 反射率百分点 (pp)")
    pure = table[np.isclose(table.true_snow_fraction, 1)].sort_values("true_radius_um")
    for values, label, color, style in (
        (pure.fixed_snow_fraction * 100, "固定 200 μm / 0°", COLORS[0], "-"),
        (pure.minimum_snow_fraction * 100, "多粒径 MIN RMS / 0°", "#444444", "--"),
    ):
        axes[1].plot(pure.true_radius_um, values, label=label, color=color, ls=style)
    angle_path = angle60 / "tables/fig3_grid.csv"
    sixty = bundle.csv(angle_path)
    pure60 = sixty[np.isclose(sixty.true_snow_fraction, 1)].sort_values("true_radius_um")
    axes[1].plot(
        pure60.true_radius_um,
        pure60.fixed_snow_fraction * 100,
        color=COLORS[1],
        ls="-.",
        label="固定 200 μm / 60°",
    )
    axes[1].scatter(
        [50, 1500],
        [86, 120],
        marker="s",
        facecolor="none",
        edgecolor="#222222",
        s=50,
        label="1998原文端点约值（非逐点真值）",
        zorder=5,
    )
    axes[1].axhline(100, color="#777777", lw=0.8, ls=":")
    axes[1].axvline(200, color="#888888", lw=0.8, ls=":")
    axes[1].set(
        title="(b) 纯雪估计会低于或超过 100%",
        xlabel="真实雪粒半径 / μm",
        ylabel="阴影归一化雪比例 / %",
        xlim=(30, 1520),
        ylim=(78, 124),
    )
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Painter Fig. 3：粒径不匹配的覆盖偏差与光谱残差", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.84, wspace=0.32)
    footer(fig, "146 粒径 × 101 雪比例 × 224 波段 · WW80 + 现代 SNICAR 光学属性 · MGS，保留越界值")
    bundle.verifications.append(
        {
            "name": "fig3_grid_dimensions",
            "shape": list(grid.shape),
            "expected": [146, 101],
            "passed": list(grid.shape) == [146, 101],
        }
    )
    bundle.save(
        fig,
        "03_fig3_error_and_cover",
        "Fig. 3误差面与纯雪覆盖估计",
        f"(a)光谱RMS随粒径和真实覆盖变化；200 μm匹配行误差近于零。(b)0°下50与1500 μm纯雪被估计为{pure.iloc[0].fixed_snow_fraction * 100:.2f}%和{pure.iloc[-1].fixed_snow_fraction * 100:.2f}%。空方形是原文约86%与120%端点，来源：Painter et al. (1998)，doi:10.1016/S0034-4257(98)00041-8。",
        [path, angle_path],
        "光谱RMS与雪覆盖误差是不同量；0°是假设，60°为敏感性。历史1994响应、光学属性与quartz样本差异记录于方法表。",
    )

    residual_path = source / "tables/residual_480_vs_150.csv"
    residual = bundle.csv(residual_path)
    fig, axes = plt.subplots(
        2, 1, figsize=(8.6, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].plot(
        residual.wavelength_um, residual.observed, color="#222222", label="真值：480 μm纯雪"
    )
    axes[0].plot(
        residual.wavelength_um,
        residual.modeled,
        color=COLORS[0],
        ls="--",
        label="拟合：150 μm雪 + quartz + 零阴影",
    )
    axes[0].set(ylabel="反射率（无量纲）", title="错误雪粒径可以产生有结构的残差")
    axes[0].legend(fontsize=9)
    axes[1].plot(
        residual.wavelength_um,
        residual.E_c * 100,
        color="#D55E00",
        label=r"$E_c=R_c-\widehat{R}_c$",
    )
    axes[1].axhline(0, color="#444444", lw=0.8)
    for marker in (1.03, 1.26):
        axes[1].axvline(marker, color="#888888", ls=":", lw=0.8)
    axes[1].set(xlabel="波长 / μm", ylabel=r"$E_c$ / pp", xlim=(0.38, 2.5))
    axes[1].legend(loc="lower right")
    fig.subplots_adjust(bottom=0.14, top=0.91, hspace=0.12)
    footer(fig, "合成480 μm真值、150 μm拟合 · 原始系数重建 · 原文478 μm像元的流程类比")
    diff = np.max(np.abs(residual.observed - residual.modeled - residual.E_c))
    bundle.verifications.append(
        {
            "name": "synthetic_residual_identity",
            "max_abs_error": float(diff),
            "passed": bool(diff < 1e-12),
        }
    )
    bundle.save(
        fig,
        "04_synthetic_residual",
        "合成纯雪的拟合与误差项",
        "上图将480 μm纯雪光谱与固定150 μm雪端元的重建光谱叠加；下图保留有正有负的逐波段误差，虚线提示1.03和1.26 μm附近的光谱特征。",
        [residual_path],
        "图件来自已知比例的控制合成案例，用于检验固定端元的残差结构；原文Fig.9的478 μm像元另作定性参照。",
    )


def stability_figure(bundle: FigureBundle, source: Path) -> None:
    """保留固定端元方法的完整误差尺度，并放大候选库方法的误差范围。"""
    path = source / "tables/monte_carlo_summary.csv"
    table = bundle.csv(path)
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.7), sharex=True)
    settings = [
        (axes[0, 0], "cover_rmse", 100, "覆盖 RMSE / 百分点", False),
        (axes[0, 1], "radius_rmse_um", 1, "粒径 RMSE / μm", False),
        (axes[1, 0], "cover_rmse", 100, "覆盖 RMSE / 百分点（多粒径局部）", True),
        (axes[1, 1], "radius_rmse_um", 1, "粒径 RMSE / μm（多粒径局部）", True),
    ]
    for ax, metric, scale, ylabel, zoom in settings:
        for method in METHOD_LABELS:
            if zoom and method == "fixed_original":
                continue
            group = table[table.method == method].sort_values("noise_std")
            x = group.noise_std.to_numpy()
            mean = group[f"{metric}_mean"].to_numpy() * scale
            low = group[f"{metric}_p025"].to_numpy() * scale
            high = group[f"{metric}_p975"].to_numpy() * scale
            ax.plot(
                x,
                mean,
                marker="o",
                ls="--" if method == "multi_fcls" else "-",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                markersize=5,
            )
            ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.14)
        ax.set(
            ylabel=ylabel,
            xlabel="反射率噪声标准差 σ",
            xticks=[0, 0.005, 0.01],
            xlim=(-0.0003, 0.0103),
        )
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    for ax, label in zip(
        axes.ravel(),
        ("(a) 全尺度覆盖误差", "(b) 全尺度粒径误差", "(c) 多粒径覆盖细节", "(d) 多粒径粒径细节"),
    ):
        ax.set_title(label)
    fig.suptitle("多粒径与非负约束在同模型控制实验中的表现", fontsize=14, y=0.995)
    fig.subplots_adjust(bottom=0.12, top=0.9, left=0.09, right=0.98, hspace=0.4, wspace=0.32)
    footer(
        fig, "75 真值案例 · 每个σ重复20次 · 阴影0/0.3/0.7 · 阴影归一化覆盖 · 阴影带为重复分位范围"
    )
    bundle.save(
        fig,
        "05_noise_constraints",
        "噪声、端元库与求解约束",
        "上排保留固定200 μm模型的较大误差，下排用独立纵轴放大多粒径方法；两排不可按图面高度直接比较。半透明区间为20次噪声重复的2.5%–97.5%分位。粒径只在真值雪比例≥0.1时统计。",
        [path, source / "tables/monte_carlo_repeats.csv"],
        "正演与反演共享同一模型、端元和理想噪声设置，结果用于内部实现与噪声稳定性检验。",
    )


def fraction_maps(bundle: FigureBundle, source: Path) -> None:
    """绘制原始非阴影系数与阴影补量，两者合计为 1。"""
    path = source / "rasters/aviris_multi_fcls_spectral_fractions.tif"
    with rasterio.open(bundle.source(path)) as dataset:
        bands = dataset.read(masked=True).astype(float).filled(np.nan)
        descriptions = list(dataset.descriptions)
    _, extent, crs = load_raster(bundle, path)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 9.2))
    for index, (ax, label) in enumerate(
        zip(axes.ravel(), ("(a) 雪", "(b) 植被", "(c) 岩石", "(d) 土壤", "(e) 零光谱阴影"))
    ):
        image = map_axes(
            ax, bands[index] * 100, extent, crs, label, sequential_cmap("cividis"), 0, 100
        )
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(
        0.08,
        0.75,
        "物理系数 + 阴影 = 100%\n\n本图用原始光谱系数。\n阴影归一化比例另行显示。\n\nNoData保留灰色；\n各面板固定0–100%色标。",
        va="top",
        fontsize=11,
        linespacing=1.8,
    )
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.13, top=0.89, hspace=0.31, wspace=0.26)
    cax = fig.add_axes([0.905, 0.27, 0.016, 0.47])
    fig.colorbar(image, cax=cax, label="原始光谱混合系数 / %", ticks=[0, 25, 50, 75, 100])
    fig.suptitle("Mammoth：各端元在同一个30 m像元中的系数", fontsize=15, y=0.965)
    add_nodata_legend(fig)
    footer(
        fig,
        "AVIRIS-Classic 2013-05-03 · FCLS + 30–1100 μm雪库 · WGS84 / UTM 11N (EPSG:32611) · 真北箭头",
    )
    valid = np.isfinite(bands).all(axis=0)
    maximum = np.max(np.abs(bands[:, valid].sum(axis=0) - 1))
    bundle.verifications.append(
        {
            "name": "raw_fraction_sum",
            "max_abs_error": float(maximum),
            "band_names": descriptions,
            "passed": bool(maximum < 2e-6),
        }
    )
    bundle.save(
        fig,
        "06_aviris_raw_endmembers",
        "AVIRIS原始端元系数与阴影",
        f"五个面板显示同一FCLS模型的雪、植被、岩石、土壤及零阴影系数；有效像元内五项和为1（最大浮点偏差{maximum:.1e}）。用共同0–100%连续色标便于比较。显示未进行平滑或额外插值。",
        [path, source / "endmembers.json"],
        "图件展示含阴影的原始光谱系数；植被、岩石分别由实验室松针和花岗岩代表谱构成，砾石端元列为待补输入。",
    )

    models = ["aviris_fixed200_original", "aviris_multi_original", "aviris_multi_fcls"]
    titles = ["(a) 固定200 μm / MGS", "(b) 多粒径 / MGS", "(c) 多粒径 / FCLS"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.2))
    inputs, counts = [], []
    for ax, model, title in zip(axes, models, titles):
        raster = source / f"rasters/{model}_normalized_fractions.tif"
        values, extent, crs = load_raster(bundle, raster)
        image = map_axes(
            ax, values * 100, extent, crs, title, sequential_cmap("Blues", overflow=True), 0, 100
        )
        negative = int(np.sum(values < -1e-8))
        overflow = int(np.sum(values > 1 + 1e-8))
        ax.text(
            0.03,
            0.975,
            f"<0：{negative}；>100%：{overflow}",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 2},
        )
        counts.append({"model": model, "underflow": negative, "overflow": overflow})
        inputs.append(raster)
    fig.subplots_adjust(left=0.055, right=0.9, bottom=0.18, top=0.82, wspace=0.3)
    cax = fig.add_axes([0.925, 0.28, 0.012, 0.43])
    fig.colorbar(
        image, cax=cax, extend="both", label="阴影归一化雪比例 / %", ticks=[0, 25, 50, 75, 100]
    )
    fig.suptitle("原文模式系数越界诊断", fontsize=15, y=0.975)
    add_nodata_legend(
        fig, [Patch(facecolor="#8c6bb1", label="< 0%"), Patch(facecolor="#b35806", label="> 100%")]
    )
    footer(
        fig, "同一AVIRIS日期、端元、区域与色标 · EPSG:32611 · 保留负值和超过1的值，不裁剪原始结果"
    )
    bundle.save(
        fig,
        "07_aviris_snow_solver_comparison",
        "原文求解与FCLS雪覆盖对照",
        "三图采用相同范围与色标。紫色单列负雪比例，棕色单列超过100%的比例；每个面板给出越界像元数。MGS允许非物理系数，FCLS约束为非负且比例和为1。",
        inputs,
        "本图展示FCLS阴影归一化雪比例，分母失效像元标为灰色，覆盖精度由独立参考实验评价。",
    )
    write_json(bundle.output / "tables/snow_fraction_display_diagnostics.json", counts)


def grain_maps(bundle: FigureBundle, source: Path) -> None:
    """区分缺测、低雪比例及候选粒径位于库边界的情况。"""
    fig, axes = plt.subplots(1, 3, figsize=(13.9, 5.4))
    inputs = []
    for ax, label, title in zip(
        axes[:2],
        ("aviris_multi_original", "aviris_multi_fcls"),
        ("(a) MGS：候选有效半径", "(b) FCLS：候选有效半径"),
    ):
        path = source / f"rasters/{label}_grain_rms_condition.tif"
        array, extent, crs = load_raster(bundle, path)
        image = map_axes(ax, array, extent, crs, title, sequential_cmap("viridis"), 30, 1100)
        inputs.append(path)
    qa_path = source / "rasters/aviris_multi_fcls_qa.tif"
    qa, extent, crs = load_raster(bundle, qa_path)
    qa = np.nan_to_num(qa, nan=65535).astype(np.uint16)
    solution = bundle.npz(source / "arrays/aviris_multi_fcls.npz")
    valid_input = np.isfinite(solution["rms"]).reshape(qa.shape)
    categories = np.zeros(qa.shape, dtype=np.uint8)
    categories[(qa & 16) != 0] = 1
    categories[(qa & 8) != 0] = 2
    categories[(qa & 4) != 0] = 3
    categories[~valid_input | ((qa & 1) != 0)] = 4
    colorlist = ["#0072B2", "#E69F00", "#ececec", "#8c8c8c", "#333333"]
    labels = [
        "库内有效半径",
        "库边界30或1100 μm",
        "归一化雪比例<0.1",
        "归一化分母≤0.05",
        "无输入 / 无解",
    ]
    cmap = ListedColormap(colorlist)
    norm = BoundaryNorm(np.arange(-0.5, 5.5), cmap.N)
    map_axes(axes[2], categories, extent, crs, "(c) FCLS粒径报告状态", cmap, 0, 4, norm)
    axes[2].legend(
        handles=[Patch(facecolor=c, label=t) for c, t in zip(colorlist, labels)],
        loc="upper left",
        fontsize=7,
        framealpha=0.94,
        borderpad=0.4,
    )
    fig.subplots_adjust(left=0.05, right=0.94, bottom=0.25, top=0.83, wspace=0.36)
    cax = fig.add_axes([0.30, 0.145, 0.35, 0.018])
    fig.colorbar(
        image,
        cax=cax,
        orientation="horizontal",
        label="光学等效雪粒半径 / μm（候选参数）",
        ticks=[30, 200, 500, 800, 1100],
    )
    fig.suptitle("粒径图的空白与库边界有不同含义", fontsize=15, y=0.973)
    footer(
        fig,
        "AVIRIS 2013-05-03 · EPSG:32611 · (a)(b)未报告值为灰色 · (c)诊断按无输入→分母→低雪→边界优先",
    )
    summary = {label: int(np.sum(categories == index)) for index, label in enumerate(labels)}
    inputs.extend([qa_path, source / "arrays/aviris_multi_fcls.npz"])
    bundle.save(
        fig,
        "08_aviris_radius_and_quality",
        "AVIRIS有效粒径与质量状态",
        "两种求解的半径图共享30–1100 μm色标。右图另显示FCLS报告状态，橙色表示最低RMS选择的库边界状态；浅灰为低雪比例导致的半径空值。状态像元数："
        + "；".join(f"{k}={v}" for k, v in summary.items())
        + "。",
        inputs,
        "图件展示MIN RMS候选半径、求解模式差异和QA状态；原文Fig.5使用吸收积分方法，空间粒径绝对精度需同期实测。",
    )
    write_json(bundle.output / "tables/grain_quality_display_counts.json", summary)


def insert_masked_gaps(w: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """保留传感器内的波长顺序，并在缺测波段处断开光谱曲线。"""
    order = np.argsort(w, kind="stable")
    w, values = np.asarray(w)[order], np.asarray(values)[order]
    gaps = np.flatnonzero(np.diff(w) > 0.04) + 1
    x = np.insert(w, gaps, np.nan)
    y = np.insert(values, gaps, np.nan)
    return x, y


def aviris_spectral_fit(bundle: FigureBundle, source: Path) -> None:
    """绘制四个已保存代表像元的观测谱、重建谱及误差项 E_c。"""
    path = source / "tables/representative_pixel_residuals.csv"
    table = bundle.csv(path)
    fig, axes = plt.subplots(4, 2, figsize=(12.6, 11), sharex=True)
    names = {"snow": "雪", "vegetation": "植被", "rock": "岩石", "soil": "土壤"}
    row_records = []
    solutions = bundle.npz(source / "arrays/aviris_multi_fcls.npz")
    for index, name in enumerate(names):
        group = table[table.representative == name]
        pixel = int(group.pixel_index.iloc[0])
        spectrum, residual = axes[index]
        for column, color, style, label in (
            ("observed", "#222222", "-", "AVIRIS观测"),
            ("modeled", COLORS[0], "--", "FCLS原始系数重建"),
        ):
            x, y = insert_masked_gaps(group.wavelength_um.to_numpy(), group[column].to_numpy())
            spectrum.plot(x, y, color=color, ls=style, label=label)
        x, y = insert_masked_gaps(group.wavelength_um.to_numpy(), group.E_c.to_numpy() * 100)
        residual.plot(x, y, color="#D55E00")
        residual.axhline(0, color="#555555", lw=0.8)
        rms = float(np.sqrt(np.mean(group.E_c.to_numpy() ** 2)))
        saved_rms = float(solutions["rms"][pixel])
        row, column = divmod(pixel, 267)
        spectrum.set(
            title=f"{names[name]}端元最大归一化比例像元（row={row}, col={column}）", ylabel="反射率"
        )
        residual.set(title=f"同像元：光谱 RMS = {rms * 100:.2f} pp", ylabel=r"$E_c$ / pp")
        for ax in (spectrum, residual):
            for low, high in ((1.35, 1.45), (1.80, 1.98)):
                ax.axvspan(
                    low,
                    high,
                    facecolor="#dddddd",
                    alpha=0.4,
                    edgecolor="#aaaaaa",
                    hatch="//",
                    linewidth=0.2,
                    zorder=0,
                )
            ax.set_xlim(0.42, 2.45)
            ax.set_xticks([0.5, 1, 1.5, 2, 2.4])
        if index == 0:
            spectrum.legend(loc="upper right", fontsize=8)
            residual.text(
                0.985,
                0.97,
                "阴影带：排除的水汽波段",
                transform=residual.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        row_records.append(
            {
                "representative": name,
                "pixel_index": pixel,
                "row_zero_based": row,
                "column_zero_based": column,
                "spectral_rms": rms,
                "saved_rms": saved_rms,
                "max_abs_residual_pp": float(np.max(np.abs(group.E_c.to_numpy())) * 100),
            }
        )
    for ax in axes[-1]:
        ax.set_xlabel("波长 / μm")
    fig.suptitle("实景拟合的残差尖峰必须保留并解释", fontsize=15, y=0.987)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.075, top=0.925, hspace=0.52, wspace=0.24)
    footer(
        fig,
        "Mammoth AVIRIS / 182保留波段 · 不连接掩膜空缺 · 各残差面板独立纵轴 · 代表像元非随机抽样",
    )
    discrepancy = np.max(np.abs(table.observed - table.modeled - table.E_c))
    bundle.verifications.append(
        {
            "name": "real_residual_identity",
            "max_abs_error": float(discrepancy),
            "passed": bool(discrepancy < 1e-12),
        }
    )
    bundle.verifications.append(
        {
            "name": "representative_rms_agrees_saved_npz",
            "max_abs_error": float(
                max(abs(r["spectral_rms"] - r["saved_rms"]) for r in row_records)
            ),
            "passed": bool(
                all(abs(r["spectral_rms"] - r["saved_rms"]) < 1e-7 for r in row_records)
            ),
        }
    )
    bundle.save(
        fig,
        "09_aviris_observed_fit_residual",
        "AVIRIS观测光谱、重建光谱与误差项",
        "每行左边显示同一像元的观测/重建光谱，右边为Ec=观测−重建；单位pp为反射率百分点。代表像元按照对应端元的最大归一化比例选择，未按最小RMS挑选。灰色斜线带显示预先排除的水汽波段。",
        [path, source / "arrays/aviris_multi_fcls.npz", source / "band_selection.json"],
        "代表像元按最大比例选取，残差在掩膜边缘可显著增大。各行纵轴分别标明范围，误差比较依据数值RMS；残差按原始波段显示。",
    )
    pd.DataFrame(row_records).to_csv(
        bundle.output / "tables/representative_spectral_diagnostics.csv", index=False
    )


def coarse_and_scatter(bundle: FigureBundle, source: Path, south: Path) -> None:
    """各地图采用同一配对像元集合，散点图按区域分层。"""
    path = source / "tables/coarse_pixel_comparison.csv"
    table = bundle.csv(path)
    reference_raster = source / "rasters/real_modis_fcls_normalized_fractions.tif"
    sample, extent, crs = load_raster(bundle, reference_raster)
    paired = table.paired_valid.to_numpy(dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.5))
    for ax, field, title in zip(
        axes,
        ("aviris_mean_snow", "simulated_modis_snow", "real_modis_snow"),
        ("(a) 聚合AVIRIS模型参照", "(b) 模拟MODIS", "(c) 真实同日MOD09GA"),
    ):
        array = table[field].to_numpy().copy()
        array[~paired] = np.nan
        image = map_axes(
            ax,
            array.reshape(sample.shape) * 100,
            extent,
            crs,
            title,
            sequential_cmap("Blues"),
            0,
            100,
        )
    fig.subplots_adjust(left=0.065, right=0.9, bottom=0.23, top=0.78, wspace=0.34)
    cax = fig.add_axes([0.93, 0.31, 0.012, 0.31])
    fig.colorbar(image, cax=cax, label="阴影归一化雪比例 / %", ticks=[0, 25, 50, 75, 100])
    fig.suptitle(f"真实双源比较应保持同一有效样本：Mammoth n={paired.sum()}", fontsize=14, y=0.96)
    add_nodata_legend(fig)
    footer(
        fig,
        "2013-05-03 · 真实MODIS等面积Sinusoidal网格，约463.313 m · 所有面板均限定相同严格QA配对像元",
    )
    bundle.save(
        fig,
        "10_modis_common_valid_maps",
        "真实与模拟MODIS的共同有效像元",
        f"三图都只绘制严格QA且AVIRIS有效面积覆盖≥95%的同一批{paired.sum()}个配对像元，避免因遮罩不同造成图面比较误读。灰色标识共同有效集合外的像元。",
        [path, reference_raster],
        "配对像元少且AVIRIS也是模型估计；同日不保证完全同时，太阳/观测角度、大气残差和端元库差异仍存在。",
    )

    south_path = south / "tables/coarse_pixel_comparison.csv"
    low = bundle.csv(south_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.5), sharey=True)
    selected = [
        (table[table.paired_valid], "Mammoth有雪区", COLORS[0]),
        (low[low.paired_valid], "南侧低雪区", COLORS[1]),
    ]
    for ax, (group, label, color) in zip(axes, selected):
        x, y = group.aviris_mean_snow.to_numpy(), group.real_modis_snow.to_numpy()
        rmse = np.sqrt(np.mean((y - x) ** 2)) * 100
        ax.scatter(
            x * 100,
            y * 100,
            color=color,
            s=40,
            edgecolors="white",
            linewidth=0.4,
            alpha=0.8,
            label=f"严格QA n={len(group)}",
        )
        ax.plot([0, 100], [0, 100], color="#555555", ls="--", lw=1, label="1:1线")
        ax.axvline(10, color="#aaaaaa", ls=":", lw=0.8)
        ax.axhline(10, color="#aaaaaa", ls=":", lw=0.8)
        ax.set(
            title=f"{label}：参照雪≥10%仅{int(np.sum(x >= 0.1))}个",
            xlim=(0, 100),
            ylim=(0, 100),
            xlabel="面积聚合AVIRIS模型雪比例 / %",
            aspect="equal",
        )
        ax.text(
            0.04, 0.94, f"差异RMSE = {rmse:.2f} pp", transform=ax.transAxes, va="top", fontsize=10
        )
        ax.legend(loc="lower right")
    axes[0].set_ylabel("真实MODIS模型雪比例 / %")
    fig.suptitle("严格QA下的有雪与低雪样本", fontsize=14, y=0.985)
    fig.subplots_adjust(bottom=0.18, top=0.84, left=0.08, right=0.98, wspace=0.14)
    footer(fig, "两个区域分层呈现 · 相同0–100%坐标 · 虚线10%为报告阈值 · 双源模型一致性")
    bundle.save(
        fig,
        "11_strict_stratified_sensor_scatter",
        "严格QA双源分层散点",
        "左右分别显示Mammoth和南侧低雪区，均采用共同有效像元和同一百分比坐标；点到1:1线的距离反映两个模型估计的差异。低雪区点集中在原点附近，其小绝对差异不证明中高雪覆盖反演已经得到验证。",
        [path, south_path],
        "样本具有空间依赖，本实验评价双源一致性，地面覆盖与粒径标签列为后续采样需求。",
    )


def resolution_figure(bundle: FigureBundle, source: Path) -> None:
    """绘制固定端元字典的光谱与空间分辨率对照。"""
    path = source / "tables/controlled_resolution.csv"
    table = bundle.csv(path)
    raster = source / "rasters/combined_modis_same4_normalized_fractions.tif"
    sample, extent, crs = load_raster(bundle, raster)
    common = (table.coverage >= 0.95).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.5))
    for ax, field, title in zip(
        axes,
        ("reference_snow", "spatial_only_snow", "combined_snow"),
        ("(a) AVIRIS先解混后聚合", "(b) 仅空间聚合后解混", "(c) 光谱+空间聚合后解混"),
    ):
        array = table[field].to_numpy().copy()
        array[~common] = np.nan
        image = map_axes(
            ax,
            array.reshape(sample.shape) * 100,
            extent,
            crs,
            title,
            sequential_cmap("Blues"),
            0,
            100,
        )
    fig.subplots_adjust(left=0.065, right=0.9, bottom=0.23, top=0.78, wspace=0.34)
    cax = fig.add_axes([0.93, 0.31, 0.012, 0.31])
    fig.colorbar(image, cax=cax, label="阴影归一化雪比例 / %", ticks=[0, 25, 50, 75, 100])
    fig.suptitle("固定端元字典后再比较分辨率效应", fontsize=14, y=0.96)
    add_nodata_legend(fig)
    footer(
        fig, "相同雪/植被/岩石/土壤/阴影、粒径库与FCLS · MODIS Sinusoidal网格 · 覆盖≥95%共同范围"
    )
    metrics_path = source / "metrics.json"
    metrics = bundle.json(metrics_path)
    bundle.save(
        fig,
        "12_controlled_resolution_maps",
        "固定字典的光谱与空间退化",
        f"仅空间退化、光谱加空间退化相对AVIRIS模型参照的雪比例差异RMSE分别为{metrics['spatial_only']['rmse'] * 100:.3f}、{metrics['combined']['rmse'] * 100:.3f} pp；仅光谱退化细像元差异为{metrics['spectral_only']['rmse'] * 100:.3f} pp。整个对照保持同一字典，区别于早期在降波段时同时合并裸地类别的试验。",
        [path, raster, metrics_path],
        "最低RMS库选择、FCLS与阴影归一化包含非线性，因此处理顺序可能影响结果；本实验参照为AVIRIS模型估计。",
    )


def qa_figure(bundle: FigureBundle, source: Path) -> None:
    """单列放宽邻云筛选的敏感性结果，与严格 QA 基线对照。"""
    path = source / "tables/qa_tier_comparison.csv"
    table = bundle.csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), sharey=True)
    titles = {
        "strict7": "严格7波段基线",
        "strict5": "严格5波段对照",
        "adjacent5_sensitivity": "5波段 + 保留邻云",
    }
    for ax, tier, color in zip(axes, titles, [COLORS[0], COLORS[1], COLORS[3]]):
        group = table[table.tier == tier]
        x, y = group.reference_snow.to_numpy(), group.real_snow.to_numpy()
        ax.scatter(
            x * 100, y * 100, s=24, color=color, alpha=0.75, edgecolors="white", linewidth=0.2
        )
        ax.plot([0, 100], [0, 100], color="#555555", lw=1, ls="--")
        ax.set(
            title=f"{titles[tier]} / n={len(group)}",
            xlabel="AVIRIS模型参照 / %",
            xlim=(0, 100),
            ylim=(0, 100),
            aspect="equal",
        )
        ax.text(
            0.04,
            0.94,
            f"差异RMSE={np.sqrt(np.mean((y - x) ** 2)) * 100:.2f} pp",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        if tier == "adjacent5_sensitivity":
            ax.text(
                0.04,
                0.78,
                "敏感性结果\n包含潜在云污染",
                transform=ax.transAxes,
                color="#74456f",
                fontsize=9,
            )
    axes[0].set_ylabel("真实MODIS模型雪比例 / %")
    fig.suptitle("放宽QA得到的样本，单独作为敏感性分析", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.17, top=0.84, wspace=0.13)
    footer(fig, "三组共享部分像元 · 5波段为1/2/4/6/7 · 各QA策略单独统计")
    bundle.save(
        fig,
        "13_qa_sensitivity",
        "波段质量和邻云策略敏感性",
        "严格7波段、严格5波段、保留邻云5波段分别单独显示。同一像元可能出现在多个策略中，统计按QA策略分别计算；右图用于邻云敏感性分析。",
        [path, source / "metrics.json"],
        "5波段同时改变观测维度与筛选策略，不将其差异完全归因于QA；band5代码8是dead/interpolated、band3代码13是校正越界，未直接称为积雪饱和。",
    )


def locator_figure(bundle: FigureBundle, mammoth: Path, south: Path) -> None:
    """依据影像实际坐标绘制研究区位置图。"""
    metadata_path = ROOT / "data/cache/aviris_metadata.json"
    metadata = bundle.json(metadata_path)
    binary_path = ROOT / metadata["binary_path"]
    with rasterio.open(binary_path) as dataset:
        bounds, crs = dataset.bounds, dataset.crs
    transformer = Transformer.from_crs(crs, 4326, always_xy=True)
    corner_x = [bounds.left, bounds.right, bounds.right, bounds.left, bounds.left]
    corner_y = [bounds.bottom, bounds.bottom, bounds.top, bounds.top, bounds.bottom]
    lon, lat = transformer.transform(corner_x, corner_y)
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 6.5), gridspec_kw={"width_ratios": [0.85, 1.15]})
    for ax in axes:
        ax.plot(lon, lat, color="#777777", lw=1, label="AVIRIS整幅矩形包络")
        ax.set(xlabel="经度 / °E", ylabel="纬度 / °N")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inputs = [metadata_path, ROOT / metadata["header_path"]]
    for run, label, color in (
        (mammoth, "Mammoth西侧", COLORS[0]),
        (south, "南侧低雪区", COLORS[1]),
    ):
        raster = run / "rasters/aviris_multi_fcls_spectral_fractions.tif"
        with rasterio.open(bundle.source(raster)) as dataset:
            bounds = dataset.bounds
            t = Transformer.from_crs(dataset.crs, 4326, always_xy=True)
        x = [bounds.left, bounds.right, bounds.right, bounds.left, bounds.left]
        y = [bounds.bottom, bounds.bottom, bounds.top, bounds.top, bounds.bottom]
        rl, rt = t.transform(x, y)
        for ax in axes:
            ax.fill(
                rl, rt, facecolor=color, alpha=0.14, edgecolor=color, linewidth=1.8, label=label
            )
        center = t.transform((bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2)
        axes[1].text(center[0], center[1], label, fontsize=9, ha="center", color=color)
        inputs.append(raster)
    axes[0].set_title("(a) f130503t01p00r14整幅包络")
    axes[0].set_xticks([-119.2, -119.0])
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].set(title="(b) 两个实际分析ROI", xlim=(-119.19, -118.98), ylim=(37.49, 37.69))
    axes[1].xaxis.set_major_locator(MaxNLocator(4))
    axes[0].legend(loc="upper right", fontsize=7)
    axes[1].legend(loc="lower right", fontsize=7)
    axes[0].set_aspect(1 / np.cos(np.deg2rad(37.8)))
    axes[1].set_aspect(1 / np.cos(np.deg2rad(37.6)))
    fig.suptitle("Sierra Nevada：已运行AVIRIS区域的坐标定位", fontsize=14, y=0.985)
    fig.subplots_adjust(bottom=0.15, top=0.86, left=0.08, right=0.98, wspace=0.28)
    footer(fig, "WGS84经纬度定位示意 · 来源为实际栅格范围 · 矩形标识网格包络 · 有效性按QA确定")
    bundle.save(
        fig,
        "14_roi_coordinate_locator",
        "实际研究区域坐标定位",
        "左图显示2013-05-03完整AVIRIS文件的地理包络，右图显示两个30 m分析ROI。边界从原栅格投影坐标转换而来；可与图6–12的地图坐标和ROI中心位置核对。",
        inputs,
        "定位示意按实际栅格经纬度包络绘制，有效覆盖由输入QA确定。经纬度轴用于地理定位。",
    )


def palette_audit(bundle: FigureBundle) -> None:
    """使用 Machado 矩阵展示正常色觉、红色盲和绿色盲下的配色。

    矩阵取自 Machado et al. (2009) 模型的 severity=100 设置，
    数值依据 colorspacious/cvd.py 核对，用于检查图件配色。
    """
    matrices = {
        "正常色觉": np.eye(3),
        "protanopia 模拟": np.array(
            [
                [0.152286, 1.052583, -0.204868],
                [0.114503, 0.786281, 0.099216],
                [-0.003882, -0.048116, 1.051998],
            ]
        ),
        "deuteranopia 模拟": np.array(
            [
                [0.367322, 0.860646, -0.227968],
                [0.280085, 0.672501, 0.047413],
                [-0.011820, 0.042940, 0.968881],
            ]
        ),
    }
    palettes = {
        name: plt.get_cmap(name)(np.linspace(0, 1, 256))[:, :3]
        for name in ("cividis", "viridis", "magma", "Blues")
    }
    from matplotlib.colors import to_rgb

    palettes["线条色（另有线型）"] = np.array([to_rgb(color) for color in COLORS])
    palettes["QA类别（另有文字）"] = np.array(
        [to_rgb(color) for color in ("#0072B2", "#E69F00", "#ececec", "#8c8c8c", "#333333")]
    )
    fig, axes = plt.subplots(len(palettes), 3, figsize=(11.5, 6))
    for column, (label, matrix) in enumerate(matrices.items()):
        for row, (name, rgb) in enumerate(palettes.items()):
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            transformed = np.clip(linear @ matrix.T, 0, 1)
            srgb = np.where(
                transformed <= 0.0031308,
                12.92 * transformed,
                1.055 * transformed ** (1 / 2.4) - 0.055,
            )
            axes[row, column].imshow(srgb[None, :, :], aspect="auto", interpolation="nearest")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].grid(False)
            if row == 0:
                axes[row, column].set_title(label)
            if column == 0:
                axes[row, column].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.12, hspace=0.4, wspace=0.13)
    footer(
        fig,
        "Machado et al. (2009) severity=100 · 在线性sRGB变换后回到sRGB · 色标有数字，类别有文字，曲线有线型",
    )
    fig.savefig(
        bundle.output / "figures/palette_accessibility_audit.png", dpi=300, bbox_inches="tight"
    )
    fig.savefig(bundle.output / "figures/palette_accessibility_audit.svg", bbox_inches="tight")
    plt.close(fig)
    write_json(
        bundle.output / "tables/palette_accessibility_audit.json",
        {
            "checked_date": "2026-10-01",
            "model": "Machado et al. 2009 severity=100 in linear sRGB",
            "matrix_reference": "https://raw.githubusercontent.com/njsmith/colorspacious/master/colorspacious/cvd.py",
            "primary_model_reference": "https://doi.org/10.1109/TVCG.2009.113",
            "colormap_guidance": "https://matplotlib.org/stable/users/explain/colors/colormaps.html",
            "matrices": {label: matrix.tolist() for label, matrix in matrices.items()},
            "scope": "Palette-level color-vision simulation with visual inspection.",
        },
    )


def complete_run(base: Path, prefix: str) -> Path:
    """按版本号选择最近一次完成的源实验。"""
    runs = [
        path
        for path in sorted(base.glob(prefix + "*"))
        if (path / "run_manifest.json").exists()
        and read_json(path / "run_manifest.json").get("status") == "complete"
    ]
    if not runs:
        raise FileNotFoundError(f"No completed source run: {base / prefix}")
    return runs[-1]


def output_folder(date: str) -> Path:
    """新建带版本号的图件目录。"""
    base = ROOT / "experiments/painter1998" / date / "v2"
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while (base / f"10_publication_figures_r{number:03d}").exists():
        number += 1
    output = base / f"10_publication_figures_r{number:03d}"
    output.mkdir()
    (output / "figures").mkdir()
    (output / "tables").mkdir()
    return output


def write_gallery(bundle: FigureBundle) -> None:
    """生成中文图集，在每幅图下附图解和来源说明。"""
    lines = [
        "# Painter复现：注释图集",
        "",
        "记录日期：2026-10-01。此图集重新绘制已保存的v1实验数值，不重新解混、不平滑，也不修改任何原图。PNG为300 dpi；SVG保留文本及矢量坐标，像素地图与误差面按原像元表示。",
        "",
        "## 制图设置",
        "",
        "新版图件补充投影坐标、单位、NoData和越界色标；粒径图标明低雪、小分母及库边界，代表像元同时显示观测、重建与残差谱。数值沿用原实验输出。",
        "",
        "统一制图约定：百分比固定0–100%；超范围比例使用专色，未报告数据为灰色；半径固定30–1100 μm。空间图用实际投影坐标、真北箭头与投影网格长度比例尺；MODIS为Sinusoidal，箭头按当地真北方向计算。颜色采用感知有序cividis、viridis、magma或Blues；彩色线同时以标签/线型区分。",
        "",
        "## 图件与说明",
        "",
    ]
    for number, item in enumerate(bundle.catalog, 1):
        png = (ROOT / item["png"]).as_posix()
        svg = (ROOT / item["svg"]).as_posix()
        lines.extend(
            [
                f"### 图{number} {item['title']}",
                "",
                f"![{item['title']}]({png})",
                "",
                f"图解：{item['caption']}",
                "",
                f"边界：{item['limits']}",
                "",
                "数据来源："
                + "；".join(f"[{Path(p).name}]({(ROOT / p).as_posix()})" for p in item["sources"])
                + "。",
                "",
                f"[可编辑SVG]({svg})",
                "",
            ]
        )
    lines.extend(
        [
            "## 复现与核验",
            "",
            "缓存已经在本项目中，运行 `.venv/Scripts/python.exe scripts/publication_figures.py` 会创建下一编号图集；不需要重复下载。冷启动需先按根README准备SNICAR与v1实验缓存，下载和模型计算时间不包含在图集重绘时间中。",
            "",
            "重绘的数值来源和SHA-256在 `run_manifest.json` 中；图件索引在 `figure_catalog.json` 中。`verification.json`检查比例和、残差恒等式、保存RMS一致性、图像尺寸/300 dpi及源文件在运行前后的哈希，人工视觉核验另行记录。",
            "",
            "所有地图关闭平滑，只以nearest显示原网格。300 dpi提升文字与线条输出精度，不增加传感器的真实空间分辨率。低雪样本、无地面粒径真值与原文数据缺失的限制仍然保留。",
            "",
        ]
    )
    (bundle.output / "图集与图解.md").write_text("\n".join(lines), encoding="utf-8")
    (bundle.output / "README.md").write_text(
        "# 注释图件重绘\n\n本目录仅重绘已完成的v1缓存，不进行正演或解混计算。所有输出为新版本；原始数据与旧图未改写。\n\n入口：`scripts/publication_figures.py`。使用已验证项目`.venv`，不增加依赖。完整说明见[图集与图解](图集与图解.md)。\n\n字段：catalog中的id/title/png/svg/caption/limits/sources分别对应图件编号、标题、两种输出路径、图解、证据边界和输入；source_sha256记录源文件摘要。\n",
        encoding="utf-8",
    )


def main() -> None:
    """执行图件绘制并核对输入文件哈希。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-10-01")
    parser.add_argument("--source-version", default="v1")
    args = parser.parse_args()
    configure_style()
    output = output_folder(args.date)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output / "render.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    bundle = FigureBundle(output)
    base = ROOT / "experiments/painter1998" / args.date / args.source_version
    physics = complete_run(base, "01_physics")
    paper_runs = [p for p in sorted(base.glob("02_03_paper*")) if (p / "metrics.json").exists()]
    paper = [
        p
        for p in paper_runs
        if read_json(p / "metrics.json")["sources_and_assumptions"]["solar_angle_assumption_deg"]
        == 0
        and read_json(p / "metrics.json")["sources_and_assumptions"]["sensor"].startswith("2013")
    ][-1]
    sixty = [
        p
        for p in paper_runs
        if read_json(p / "metrics.json")["sources_and_assumptions"]["solar_angle_assumption_deg"]
        == 60
    ][-1]
    stability = complete_run(base, "04_stability")
    mammoth = base / "05_07_image_r001"
    south = base / "05_07_image_r002"
    qa = complete_run(base, "08_qa_sensitivity")
    resolution = complete_run(base, "09_resolution_control")
    manifest = {
        "stage": "10_publication_figures",
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": hash_file(Path(__file__)),
        "source_version": args.source_version,
        "selected_runs": {
            k: str(p.relative_to(ROOT)).replace("\\", "/")
            for k, p in {
                "physics": physics,
                "paper": paper,
                "angle60": sixty,
                "stability": stability,
                "mammoth": mammoth,
                "south": south,
                "qa": qa,
                "resolution": resolution,
            }.items()
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "matplotlib", "rasterio", "pyproj")
        },
        "method": "Saved arrays/tables only; no model refitting; fixed limits, nearest-neighbor display, 300 dpi PNG/SVG.",
        "source_files_modified": False,
    }
    write_json(output / "run_manifest_started.json", manifest)
    physics_figures(bundle, physics)
    paper_figures(bundle, paper, sixty)
    stability_figure(bundle, stability)
    fraction_maps(bundle, mammoth)
    grain_maps(bundle, mammoth)
    aviris_spectral_fit(bundle, mammoth)
    coarse_and_scatter(bundle, mammoth, south)
    resolution_figure(bundle, resolution)
    qa_figure(bundle, qa)
    locator_figure(bundle, mammoth, south)
    palette_audit(bundle)
    from PIL import Image

    image_records = []
    for item in bundle.catalog:
        with Image.open(ROOT / item["png"]) as png:
            dpi = png.info.get("dpi", (0, 0))
            image_records.append(
                {
                    "id": item["id"],
                    "width": png.width,
                    "height": png.height,
                    "dpi": list(dpi),
                    "passed": bool(png.width >= 2200 and abs(dpi[0] - 300) < 1),
                }
            )
    source_unchanged = all(
        hash_file(ROOT / name) == expected for name, expected in bundle.sources.items()
    )
    bundle.verifications.append(
        {
            "name": "source_hashes_unchanged",
            "files_checked": len(bundle.sources),
            "passed": source_unchanged,
        }
    )
    bundle.verifications.append(
        {
            "name": "png_dimensions_and_300dpi",
            "images": image_records,
            "passed": all(r["passed"] for r in image_records),
        }
    )
    verified = all(record["passed"] for record in bundle.verifications)
    write_json(
        output / "verification.json",
        {"automatic_checks": bundle.verifications, "all_passed": verified},
    )
    write_json(output / "figure_catalog.json", bundle.catalog)
    write_gallery(bundle)
    manifest.update(
        status="complete" if verified else "verification_failed",
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=bundle.sources,
        figure_count=len(bundle.catalog),
        automatic_verification_passed=verified,
    )
    write_json(output / "run_manifest.json", manifest)
    if not verified:
        raise RuntimeError("One or more figure checks failed; see verification.json")
    LOG.info("Complete: %s", output)
    print(output)


if __name__ == "__main__":
    main()
