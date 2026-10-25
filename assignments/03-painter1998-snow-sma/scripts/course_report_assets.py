"""按 A4 报告的图件尺寸绘制已保存实验结果。"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT
from publication_figures import configure_style, insert_masked_gaps, map_axes, sequential_cmap

PRINT_WIDTH_CM = 16.2
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def file_hash(path: Path) -> str:
    """以分块读取计算文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReportFigures:
    """记录报告图件及其输入文件。"""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.sources: dict[str, str] = {}
        self.figures: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []

    def source(self, path: Path) -> Path:
        """读取输入前记录其文件哈希。"""
        relative = path.relative_to(ROOT).as_posix()
        if relative not in self.sources:
            self.sources[relative] = file_hash(path)
        return path

    def csv(self, path: Path) -> pd.DataFrame:
        """读取已登记来源的数值表。"""
        return pd.read_csv(self.source(path))

    def json(self, path: Path) -> dict[str, Any]:
        """读取已登记来源的元数据。"""
        return json.loads(self.source(path).read_text(encoding="utf-8"))

    def raster(self, path: Path, band: int | None = 1) -> tuple[np.ndarray, tuple[float, ...], str]:
        """读取栅格值及以千米表示的投影范围。"""
        with rasterio.open(self.source(path)) as ds:
            values = ds.read(masked=True) if band is None else ds.read(band, masked=True)
            values = values.astype(float).filled(np.nan)
            bounds = ds.bounds
            extent = (
                bounds.left / 1000,
                bounds.right / 1000,
                bounds.bottom / 1000,
                bounds.top / 1000,
            )
            return values, extent, ds.crs.to_wkt()

    def save(self, fig: plt.Figure, name: str, report_number: int, note: str) -> None:
        """保存 300 dpi PNG 和 SVG，并记录报告排版尺寸。"""
        for suffix in ("png", "svg"):
            target = self.output / f"{name}.{suffix}"
            if target.exists():
                raise FileExistsError(target)
            fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with Image.open(self.output / f"{name}.png") as png:
            record = {
                "report_figure": report_number,
                "png": f"{name}.png",
                "svg": f"{name}.svg",
                "pixels": [png.width, png.height],
                "png_dpi": list(png.info["dpi"]),
                "print_width_cm": PRINT_WIDTH_CM,
                "print_height_cm": round(PRINT_WIDTH_CM * png.height / png.width, 2),
                "note": note,
            }
        self.figures.append(record)


def readable_style() -> None:
    """设置适合 16.2 cm 排版宽度的图中文字尺寸。"""
    configure_style()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "lines.linewidth": 1.8,
            "axes.titlepad": 10,
        }
    )


def readable_map(
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
    """使用最近邻显示栅格，并调整坐标、标注和图例尺寸。"""
    image = map_axes(ax, array, extent, crs, title, cmap, vmin, vmax, norm)
    ax.tick_params(labelsize=10.5)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    for text in ax.texts:
        text.set_fontsize(10.5)
    return image


def locator_figure(bundle: ReportFigures, original: Path, south: Path) -> None:
    """保持空间比例，显示狭长的 AVIRIS 原生航带。"""
    metadata = bundle.json(ROOT / "data/cache/aviris_metadata.json")
    bundle.source(ROOT / metadata["header_path"])
    left, bottom, right, top = np.asarray(metadata["bounds"]) / 1000
    envelope = np.array(
        [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    )
    records = []
    for path, label, color in (
        (original, "Mammoth 西侧", COLORS[0]),
        (south, "南侧低雪区", COLORS[1]),
    ):
        _, extent, _ = bundle.raster(path / "rasters/aviris_multi_fcls_spectral_fractions.tif")
        x0, x1, y0, y1 = extent
        points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        records.append((points, label, color))
    checks = [
        {
            "geometry": "scene_envelope",
            "is_valid": bool(Polygon(envelope).is_valid),
            "closed": bool(np.array_equal(envelope[0], envelope[-1])),
        }
    ]
    checks.extend(
        {
            "geometry": label,
            "is_valid": bool(Polygon(points).is_valid),
            "closed": bool(np.array_equal(points[0], points[-1])),
        }
        for points, label, _ in records
    )
    if not all(item["is_valid"] and item["closed"] for item in checks):
        raise ValueError("Invalid or unclosed locator polygon")
    fig, axes = plt.subplots(1, 2, figsize=(8, 6.25), gridspec_kw={"width_ratios": [0.90, 1.10]})
    for ax in axes:
        ax.plot(envelope[:, 0], envelope[:, 1], color="#666666", lw=1.5)
        for points, label, color in records:
            ax.fill(points[:, 0], points[:, 1], facecolor=color, alpha=0.18)
            ax.plot(points[:, 0], points[:, 1], color=color, lw=1.8)
        ax.set(xlabel="UTM Easting / km", ylabel="UTM Northing / km", aspect="equal")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=11)
    center_x = (left + right) / 2
    axes[0].set(
        xlim=(center_x - 55, center_x + 55),
        ylim=(bottom - 7, top + 7),
        title="(a) 原生航带矩形包络",
    )
    axes[0].text(
        0.5,
        0.04,
        f"宽 {right - left:.1f} km\n长 {top - bottom:.1f} km",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        bbox={"fc": "white", "ec": "#bbbbbb", "alpha": 0.95},
    )
    combined = np.vstack([points for points, _, _ in records])
    x0, y0 = combined.min(axis=0)
    x1, y1 = combined.max(axis=0)
    axes[1].set(xlim=(x0 - 4, x1 + 4), ylim=(y0 - 3, y1 + 3), title="(b) 两个 30 m 分析区")
    for points, label, color in records:
        center = points[:-1].mean(axis=0)
        axes[1].text(*center, label, ha="center", va="center", fontsize=12, color=color)
    axes[0].add_patch(
        Rectangle(
            (x0 - 4, y0 - 3),
            x1 - x0 + 8,
            y1 - y0 + 6,
            fill=False,
            edgecolor="#444444",
            lw=1.2,
            ls="--",
        )
    )
    handles = [plt.Line2D([], [], color="#666666", label="AVIRIS 整幅包络")]
    handles.extend(
        Patch(facecolor=color, alpha=0.35, edgecolor=color, label=label)
        for _, label, color in records
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=11)
    fig.text(
        0.5,
        0.005,
        "2013-05-03  |  WGS84 / UTM 11N (EPSG:32611)  |  几何等比例；框线为网格包络，有效性由QA确定",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.91, wspace=0.44)
    bundle.checks.append(
        {
            "name": "locator_polygons",
            "passed": True,
            "geometries": checks,
            "envelope_width_km": float(right - left),
            "envelope_height_km": float(top - bottom),
            "display_aspect": "equal in projected metres",
            "old_problem": "Tall scene equal-aspect axis became a thin strip and its legend obscured the scene top; input polygons had no self-intersection.",
        }
    )
    bundle.save(
        fig, "roi_locator_readable", 1, "原生UTM全幅及ROI等比例放大；独立图例，无连接线遮挡。"
    )


def fig3_figure(bundle: ReportFigures, paper: Path, angle60: Path) -> None:
    """纵向排列 Fig. 3 子图，留出色标和端点标签空间。"""
    table = bundle.csv(paper / "tables/fig3_grid.csv")
    sixty = bundle.csv(angle60 / "tables/fig3_grid.csv")
    grid = (
        table.pivot(
            index="true_radius_um", columns="true_snow_fraction", values="fixed_rms"
        ).sort_index()
        * 100
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.2))
    im = axes[0].pcolormesh(
        grid.columns.to_numpy() * 100,
        grid.index,
        grid,
        shading="nearest",
        cmap="magma",
        vmin=0,
        vmax=float(grid.to_numpy().max()),
        rasterized=True,
    )
    axes[0].axhline(200, color="white", lw=1, ls="--")
    axes[0].text(2, 245, "匹配半径 200 μm", color="white", fontsize=12)
    axes[0].set(
        title="(a) 固定 200 μm 雪端元的光谱误差",
        xlabel="真实雪比例 / %",
        ylabel="真实雪粒半径 / μm",
    )
    fig.colorbar(im, ax=axes[0], label="光谱 RMS / pp", pad=0.02)
    pure = table[np.isclose(table.true_snow_fraction, 1)].sort_values("true_radius_um")
    pure60 = sixty[np.isclose(sixty.true_snow_fraction, 1)].sort_values("true_radius_um")
    axes[1].plot(
        pure.true_radius_um,
        pure.fixed_snow_fraction * 100,
        color=COLORS[0],
        label="固定 200 μm，0°",
    )
    axes[1].plot(
        pure.true_radius_um,
        pure.minimum_snow_fraction * 100,
        color="#333333",
        ls="--",
        label="多粒径，0°",
    )
    axes[1].plot(
        pure60.true_radius_um,
        pure60.fixed_snow_fraction * 100,
        color=COLORS[1],
        ls="-.",
        label="固定 200 μm，60°",
    )
    axes[1].scatter(
        [50, 1500],
        [86, 120],
        s=65,
        marker="s",
        facecolor="none",
        edgecolor="black",
        label="原文端点约值",
        zorder=5,
    )
    axes[1].axhline(100, color="#555555", ls=":", lw=1)
    axes[1].axvline(200, color="#888888", ls=":", lw=1)
    axes[1].set(
        title="(b) 纯雪的覆盖估计",
        xlabel="真实雪粒半径 / μm",
        ylabel="阴影归一化雪比例 / %",
        xlim=(25, 1530),
        ylim=(78, 124),
    )
    axes[1].legend(loc="lower right", ncol=2, fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.08, top=0.94, hspace=0.47)
    bundle.save(fig, "fig3_readable", 4, "保存146×101误差面及纯雪曲线纵排放大，无插值/平滑。")


def fraction_figure(bundle: ReportFigures, original: Path) -> None:
    """将五幅比例图分为主图和同号续图。"""
    bands, extent, crs = bundle.raster(
        original / "rasters/aviris_multi_fcls_spectral_fractions.tif", None
    )
    bundle.source(original / "endmembers.json")
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for index, (ax, title) in enumerate(zip(axes.flat, ["(a) 雪", "(b) 植被", "(c) 岩石"])):
        image = readable_map(
            ax, bands[index] * 100, extent, crs, title, sequential_cmap("cividis"), 0, 100
        )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.04,
        0.82,
        "原始光谱系数\n\n雪 + 植被 + 岩石\n+ 土壤 + 阴影 = 100%",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=13,
        linespacing=1.5,
    )
    axes[1, 1].legend(
        handles=[Patch(color="#d9d9d9", label="NoData / 未报告")],
        loc="lower left",
        fontsize=12,
        frameon=False,
    )
    fig.subplots_adjust(left=0.105, right=0.94, bottom=0.19, top=0.96, wspace=0.38, hspace=0.42)
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.20, 0.047, 0.58, 0.016]),
        orientation="horizontal",
        ticks=[0, 25, 50, 75, 100],
    )
    cb.set_label("原始光谱混合系数 / %")
    valid = np.isfinite(bands).all(axis=0)
    deviation = float(np.max(np.abs(bands[:, valid].sum(axis=0) - 1)))
    bundle.checks.append(
        {"name": "raw_fraction_sum", "max_abs_error": deviation, "passed": deviation < 2e-6}
    )
    bundle.save(
        fig,
        "aviris_raw_endmembers_readable",
        5,
        "图5(a–c)：雪/植被/岩石，2列2行放大；EPSG:32611及0–100%色标不变。",
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    for index, ax, title in zip([3, 4], axes, ["(d) 土壤", "(e) 阴影"]):
        image = readable_map(
            ax, bands[index] * 100, extent, crs, title, sequential_cmap("cividis"), 0, 100
        )
    fig.subplots_adjust(left=0.105, right=0.94, bottom=0.31, top=0.92, wspace=0.38)
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.22, 0.10, 0.54, 0.025]),
        orientation="horizontal",
        ticks=[0, 25, 50, 75, 100],
    )
    cb.set_label("原始光谱混合系数 / %")
    bundle.save(
        fig,
        "aviris_raw_endmembers_continued",
        5,
        "图5(d–e)续图：土壤/阴影，1行2面板，与主图共用色标。",
    )


def grain_figure(bundle: ReportFigures, original: Path) -> None:
    """放大两幅粒径图，并单列分类 QA 图例。"""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.6))
    for ax, label, title in zip(
        axes[0],
        ["aviris_multi_original", "aviris_multi_fcls"],
        ["(a) MGS 候选半径", "(b) FCLS 候选半径"],
    ):
        array, extent, crs = bundle.raster(original / f"rasters/{label}_grain_rms_condition.tif")
        image = readable_map(ax, array, extent, crs, title, sequential_cmap("viridis"), 30, 1100)
    with np.load(bundle.source(original / "arrays/aviris_multi_fcls.npz")) as solution:
        qa = solution["qa"].reshape(array.shape)
        valid = np.isfinite(solution["rms"]).reshape(array.shape)
    categories = np.zeros(array.shape, dtype=np.uint8)
    categories[(qa & 16) != 0] = 1
    categories[(qa & 8) != 0] = 2
    categories[(qa & 4) != 0] = 3
    categories[~valid | ((qa & 1) != 0)] = 4
    colors = ["#0072B2", "#E69F00", "#ececec", "#8c8c8c", "#333333"]
    labels = [
        "库内有效半径",
        "30 / 1100 μm 库边界",
        "雪比例 < 0.1",
        "归一化分母 ≤ 0.05",
        "无有效输入 / 解",
    ]
    readable_map(
        axes[1, 0],
        categories,
        extent,
        crs,
        "(c) FCLS 报告状态",
        ListedColormap(colors),
        0,
        4,
        BoundaryNorm(np.arange(-0.5, 5.5), 5),
    )
    axes[1, 1].axis("off")
    axes[1, 1].legend(
        handles=[Patch(color=color, label=label) for color, label in zip(colors, labels)],
        loc="center left",
        fontsize=12,
        frameon=False,
        labelspacing=1.5,
        title="质量标记",
        title_fontsize=14,
    )
    fig.subplots_adjust(left=0.105, right=0.94, top=0.95, bottom=0.20, wspace=0.38, hspace=0.43)
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.23, 0.055, 0.54, 0.022]),
        orientation="horizontal",
        ticks=[30, 200, 500, 800, 1100],
    )
    cb.set_label("光学等效候选半径 / μm")
    bundle.save(
        fig, "grain_quality_readable", 6, "两半径图上排，QA与大字图例下排，数据及分类优先顺序不变。"
    )


def spectral_fit_figure(bundle: ReportFigures, original: Path) -> None:
    """绘制已保存的积雪、土壤光谱及带符号残差。"""
    table = bundle.csv(original / "tables/representative_pixel_residuals.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.6), gridspec_kw={"height_ratios": [1.7, 1]})
    error = 0.0
    for column, label in enumerate(["snow", "soil"]):
        group = table[table.representative == label]
        w = group.wavelength_um.to_numpy()
        for field, color, style, description in [
            ("observed", "#222222", "-", "AVIRIS 观测"),
            ("modeled", COLORS[0], "--", "原始系数重建"),
        ]:
            x, y = insert_masked_gaps(w, group[field].to_numpy())
            axes[0, column].plot(x, y, color=color, ls=style, label=description)
        x, y = insert_masked_gaps(w, group.E_c.to_numpy() * 100)
        axes[1, column].plot(x, y, color="#D55E00")
        axes[1, column].axhline(0, color="black", lw=0.8)
        axes[0, column].set(
            title=("(a) 雪" if label == "snow" else "(b) 土壤") + "最大比例像元",
            ylabel="反射率 / 1",
        )
        axes[0, column].legend(fontsize=11)
        axes[1, column].set(xlabel="波长 / μm", ylabel=r"$E_c$ / pp")
        for ax in axes[:, column]:
            ax.set_xlim(0.42, 2.45)
            ax.set_xticks([0.5, 1, 1.5, 2, 2.4])
            ax.tick_params(labelsize=11)
        error = max(error, float(np.max(np.abs(group.observed - group.modeled - group.E_c))))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.93, wspace=0.40, hspace=0.33)
    bundle.checks.append(
        {"name": "saved_Ec_identity", "max_abs_error": error, "passed": error < 1e-12}
    )
    bundle.save(
        fig,
        "aviris_two_pixel_fit",
        7,
        "保存的雪/土壤极值诊断像元，仅在真实波长间隙断线，保留尖峰。",
    )


def withheld_figure(bundle: ReportFigures, source: Path) -> None:
    """以简短方法名和横向箱线图展示留出波段误差。"""
    spectral = bundle.csv(source / "tables/band_residuals.csv")
    samples = bundle.csv(source / "tables/paired_sample_results.csv")
    names = {
        "wide_182_train": "182 波段基准",
        "conservative_train": "150 波段保守",
        "conservative_WB2008": "150 波段 / WB2008",
        "conservative_angle0": "150 波段 / 0°",
        "conservative_angle60": "150 波段 / 60°",
    }
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.2))
    for label, color in zip(list(names)[:3], COLORS):
        sub = spectral[spectral.method == label]
        x, y = insert_masked_gaps(sub.wavelength_um.to_numpy(), sub.band_rmse.to_numpy() * 100)
        axes[0].plot(x, y, label=names[label], color=color)
    for low, high in [(1.3, 1.5), (1.75, 2.05)]:
        axes[0].axvspan(low, high, color="gray", alpha=0.15)
    axes[0].set(
        title="(a) 同一 1200 像元的逐波段误差", xlabel="波长 / μm", ylabel="逐波段 RMSE / pp"
    )
    axes[0].legend(ncol=1, fontsize=11)
    methods = list(samples.method.unique())
    axes[1].boxplot(
        [samples.loc[samples.method == label, "heldout_rms"].to_numpy() * 100 for label in methods],
        labels=[names.get(label, label) for label in methods],
        showfliers=False,
        vert=False,
    )
    axes[1].set(title="(b) 相同 30 个未拟合波段", xlabel="公共留出波段 RMS / pp")
    axes[1].tick_params(axis="y", labelsize=12)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.09, top=0.94, hspace=0.55)
    bundle.save(
        fig, "withheld_bands_readable", 8, "按预定顺序重绘保存误差与1200样本箱线，短标签与大字体。"
    )


def scale_figure(bundle: ReportFigures, review: Path, original: Path) -> dict[str, Any]:
    """使用共同有效面积条件修正后的 219 个粗像元统计。"""
    table = bundle.csv(review / "tables/same_dictionary_control_corrected.csv")
    statistics = bundle.json(review / "metrics.json")["09_resolution_control_r001"]
    spatial = bundle.json(original / "spatial_metadata.json")
    transform = Affine(*spatial["modis_transform"])
    shape = (19, 40)
    extent = (
        transform.c / 1000,
        (transform.c + shape[1] * transform.a) / 1000,
        (transform.f + shape[0] * transform.e) / 1000,
        transform.f / 1000,
    )
    good = (table.coverage >= 0.95) & (table.reference_valid_area_fraction >= 0.95)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.8))
    for ax, field, title in zip(
        axes.flat,
        ["reference_snow", "spatial_only_snow", "combined_snow"],
        ["(a) 先解混再面积聚合", "(b) 仅空间退化", "(c) 光谱 + 空间退化"],
    ):
        values = table[field].to_numpy().copy()
        values[~good] = np.nan
        image = readable_map(
            ax,
            values.reshape(shape) * 100,
            extent,
            spatial["modis_crs"],
            title,
            sequential_cmap("Blues"),
            0,
            100,
        )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.08,
        0.95,
        "共同有效面积支持 ≥ 95%\nn = 219\n\n同一端元字典\n同一网格与色标",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=12,
        linespacing=1.5,
    )
    axes[1, 1].legend(
        handles=[Patch(color="#d9d9d9", label="NoData / 有效面积不足")],
        loc="lower left",
        fontsize=11,
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.19, top=0.95, hspace=0.48, wspace=0.40)
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.23, 0.074, 0.54, 0.026]),
        orientation="horizontal",
        ticks=[0, 25, 50, 75, 100],
    )
    cb.set_label("阴影归一化雪比例 / %")
    bundle.checks.append(
        {
            "name": "corrected_scale_support",
            "n": int(good.sum()),
            "expected": 219,
            "passed": int(good.sum()) == 219,
        }
    )
    bundle.save(
        fig,
        "scale_support_corrected",
        9,
        "三幅同网格/色标/面积支持地图2+1布局；数值乘100显示百分比。",
    )
    return statistics


def sensor_scatter_figure(bundle: ReportFigures, original: Path, south: Path) -> None:
    """固定坐标范围，放大按区域分层的严格 QA 散点图。"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.8), sharex=True, sharey=True)
    for ax, source, title, color in zip(
        axes, [original, south], ["(a) Mammoth 西侧", "(b) 南侧低雪区"], COLORS
    ):
        table = bundle.csv(source / "tables/coarse_pixel_comparison.csv")
        group = table[table.paired_valid]
        x = group.aviris_mean_snow.to_numpy() * 100
        y = group.real_modis_snow.to_numpy() * 100
        ax.scatter(x, y, s=48, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.plot([0, 100], [0, 100], color="black", lw=1, ls="--")
        ax.axvline(10, color="#888888", lw=0.8, ls=":")
        ax.axhline(10, color="#888888", lw=0.8, ls=":")
        ax.set(
            title=title,
            xlim=(0, 100),
            ylim=(0, 100),
            aspect="equal",
            xlabel="聚合 AVIRIS 雪比例 / %",
        )
        ax.text(
            0.04,
            0.94,
            f"n = {len(group)}\n差异 RMSE = {np.sqrt(np.mean((x - y) ** 2)):.2f} pp",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
        )
    axes[0].set_ylabel("真实 MODIS 雪比例 / %")
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.90, wspace=0.15)
    bundle.save(
        fig,
        "sensor_scatter_readable",
        10,
        "两个ROI共用0–100%坐标，严格QA值保持原值；1:1线及10%阈值。",
    )


def lab_figure(bundle: ReportFigures, source: Path) -> None:
    """纵向排列雪况级覆盖结果与 SSA 代理半径对照。"""
    table = bundle.csv(source / "tables/lab_model_results.csv")
    meta = bundle.csv(source / "tables/lab_condition_ssa_and_shape.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.7))
    methods = {
        "fixed200_original": "固定 200 μm / MGS",
        "multi_original": "多粒径 / MGS",
        "multi_fcls": "多粒径 / FCLS",
    }
    positions = np.arange(len(meta))
    for (method, label), color, marker in zip(
        methods.items(), [COLORS[0], COLORS[1], "#D55E00"], ["o", "s", "^"]
    ):
        values = (
            table[table.method == method]
            .set_index("name")
            .reindex(meta["name"])
            .snow_fraction.to_numpy()
            * 100
        )
        axes[0].plot(positions, values, label=label, color=color, marker=marker, ms=5)
    axes[0].axhline(100, color="black", lw=1, ls="--")
    condition_names = [
        "阳照干雪\n2013",
        "阴影干雪\n2013",
        "纯净湿雪\n2013",
        "落叶湿雪\n2013",
        "分解干雪\n2014",
        "雪壳干雪\n2014",
        "雾凇样\n2014",
        "纯净湿雪\n2014",
        "枝状干雪\n2015",
        "小粒干雪\n2015",
    ]
    if len(meta) != len(condition_names):
        raise ValueError("Lab-condition mapping must match the saved ten-condition table")
    axes[0].set(
        title="(a) 全雪目标的模型覆盖估计",
        ylabel="阴影归一化雪比例 / %",
        xticks=positions,
        xticklabels=condition_names,
    )
    axes[0].tick_params(axis="x", rotation=45, labelsize=10.5)
    axes[0].legend(ncol=1, fontsize=11, loc="lower left")
    paired = meta[meta.surface_reference_radius_um.notna()]
    for number, condition in enumerate(paired.itertuples(), 1):
        dry = condition.snow_type == 1
        axes[1].scatter(
            condition.surface_reference_radius_um,
            condition.nir_shape_radius_um,
            marker="o" if dry else "s",
            color=COLORS[0] if dry else "#D55E00",
            s=58,
        )
        axes[1].annotate(
            str(number),
            (condition.surface_reference_radius_um, condition.nir_shape_radius_um),
            xytext=(6, 8 if number % 2 else -14),
            textcoords="offset points",
            fontsize=12,
        )
    axes[1].plot([50, 1500], [50, 1500], "k--", lw=1)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlim=(50, 1500),
        ylim=(50, 1500),
        aspect="equal",
        title="(b) SSA 代理与谱形半径",
        xlabel="近表层 SSA 半径代理 / μm",
        ylabel="近红外谱形半径 / μm",
    )
    for axis in [axes[1].xaxis, axes[1].yaxis]:
        axis.set_ticks([50, 100, 200, 500, 1000, 1500])
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(plt.NullFormatter())
    axes[1].xaxis.set_ticks([50, 100, 200, 500, 1500])
    axes[1].legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", color=COLORS[0], label="干雪"),
            plt.Line2D([], [], marker="s", ls="", color="#D55E00", label="湿雪"),
        ],
        loc="lower right",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.95, hspace=0.68)
    axes[1].set_position([0.14, 0.10, 0.43, 0.32])
    fig.text(
        0.65,
        0.41,
        "右图编号\n\n1  阳照干雪（2013）\n2  阴影干雪（2013）\n3  纯净湿雪（2013）\n4  落叶湿雪（2013）\n5  分解干雪（2014）\n6  纯净湿雪（2014）\n7  枝状干雪（2015）",
        va="top",
        fontsize=11.5,
        linespacing=1.6,
    )
    bundle.save(
        fig,
        "lab_cover_ssa_readable",
        11,
        "条件级覆盖曲线/SSA代理纵排；半径和增益语义不变，编号保持1–7。",
    )


def worldview_figure(bundle: ReportFigures, source: Path) -> None:
    """绘制四种方法的已保存结果及参考分层，调整标注尺寸。"""
    scene = "LC08_L2SP_043033_20160213_02_T1"
    table = bundle.csv(source / f"tables/{scene}_sample_pixels.csv")
    methods = {
        "fixed200_six_bands": "(a) 固定 200 μm / 6 波段",
        "multi_six_bands": "(b) 多粒径 / 6 波段",
        "fixed200_four_bands": "(c) 固定 200 μm / 4 波段",
        "multi_four_bands": "(d) 多粒径 / 4 波段",
    }
    strata = {
        "no_snow": ("无雪 < 0.1", "#666666"),
        "mixed": ("混合 0.1–0.9", COLORS[0]),
        "full_snow": ("近纯雪 ≥ 0.9", "#CC79A7"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.2), sharex=True, sharey=True)
    for ax, (method, title) in zip(axes.flat, methods.items()):
        for key, (label, color) in strata.items():
            keep = table.stratum == key
            finite = keep & table[f"{method}_snow"].notna()
            ax.scatter(
                table.loc[finite, "reference_fsca"] * 100,
                table.loc[finite, f"{method}_snow"] * 100,
                s=9,
                color=color,
                alpha=0.28,
                linewidths=0,
                label=f"{label}：{finite.sum()}",
            )
        ax.plot([0, 100], [0, 100], color="black", lw=1)
        ax.set(
            title=title,
            xlim=(0, 100),
            ylim=(0, 100),
            aspect="equal",
            xlabel="WorldView 可见雪比例 / %",
            ylabel="Landsat 雪比例 / %",
        )
        legend = ax.legend(loc="upper left", fontsize=9.8, markerscale=2, labelspacing=0.5)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.95, wspace=0.21, hspace=0.29)
    bundle.save(
        fig, "worldview_scatter_readable", 12, "2016-02-13同日；原4方法各自有效样本及参考分层不变。"
    )


def run(output: Path) -> None:
    """使用已完成实验输出生成报告图件。"""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty new asset folder: {output}")
    readable_style()
    bundle = ReportFigures(output)
    index = bundle.json(ROOT / "reports/latest.json")
    selected = {label: ROOT / path for label, path in index["selected_runs"].items()}
    publication = bundle.json(selected["10_publication_figures"] / "run_manifest.json")[
        "selected_runs"
    ]
    original = ROOT / publication["mammoth"]
    south = ROOT / publication["south"]
    locator_figure(bundle, original, south)
    fig3_figure(bundle, ROOT / publication["paper"], ROOT / publication["angle60"])
    fraction_figure(bundle, original)
    grain_figure(bundle, original)
    spectral_fit_figure(bundle, original)
    withheld_figure(bundle, selected["12_aviris_band_sensitivity"])
    statistics = scale_figure(bundle, selected["15_review_reanalysis"], original)
    sensor_scatter_figure(bundle, original, south)
    lab_figure(bundle, selected["11_empirical_spectra"])
    worldview_figure(bundle, selected["13_worldview_validation"])
    unchanged = all(file_hash(ROOT / name) == digest for name, digest in bundle.sources.items())
    bundle.checks.append(
        {"name": "source_hashes_unchanged", "passed": unchanged, "count": len(bundle.sources)}
    )
    bundle.checks.append(
        {
            "name": "png_300dpi",
            "passed": all(abs(item["png_dpi"][0] - 300) < 1 for item in bundle.figures),
        }
    )
    manifest = {
        "source_files": [
            {"path": name, "sha256": digest} for name, digest in bundle.sources.items()
        ],
        "figure_files": bundle.figures,
        "scale_statistics": statistics,
        "no_refitting": True,
        "png_dpi": 300,
        "print_width_cm": PRINT_WIDTH_CM,
        "checks": bundle.checks,
        "automatic_checks_passed": all(item["passed"] for item in bundle.checks),
        "script_sha256": file_hash(Path(__file__)),
        "note": "Only geometry presentation, plot arrangement, units and labels change. All inputs preserved.",
    }
    with (output / "figure_sources.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if not manifest["automatic_checks_passed"]:
        raise RuntimeError("Figure checks failed; see figure_sources.json")
    print(
        json.dumps(
            {"output": str(output), "figures": bundle.figures, "checks": bundle.checks},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
