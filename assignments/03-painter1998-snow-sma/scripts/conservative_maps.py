"""绘制预定保守波段的 AVIRIS 诊断图，并单独保存结果版本。"""

from __future__ import annotations
import json
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT, new_run, finish_run, paired_metrics
from snowmix.image import prepare_roi, sensor_library, unmix_pixels, save_solution, summarize_result
from snowmix.validation import interval_mask
from publication_figures import configure_style, map_axes, sequential_cmap, footer
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm


def run(render_from: Path | None = None) -> Path:
    config = json.loads((ROOT / "configs/supplement_v2.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
    config["image"] = baseline["image"]
    config["forward"] = baseline["forward"]
    if render_from is not None:
        config["render_from_completed_solution"] = str(render_from.relative_to(ROOT))
    path = new_run(config, "16_conservative_maps")
    metadata = json.loads((ROOT / "data/cache/aviris_metadata.json").read_text(encoding="utf-8"))
    cube, valid, initial, transform, shape, crs, bounds = prepare_roi(config, metadata)
    w, fw = np.array(metadata["wavelength_um"]), np.array(metadata["fwhm_um"])
    mask = initial & interval_mask(w, config["band_sensitivity"]["conservative_intervals_um"])
    radius, snow, bg, _ = sensor_library(w[mask], fw[mask], 22, config)
    observation = cube[..., mask].reshape(-1, mask.sum()).astype(float)
    observation[~valid.ravel()] = np.nan
    if render_from is None:
        result = unmix_pixels(
            observation,
            snow,
            radius,
            bg[:, :3],
            config,
            solver="fcls",
            progress="conservative AVIRIS",
        )
    else:
        result = dict(np.load(render_from / "arrays/aviris_conservative_fcls.npz"))
        if len(result["rms"]) != np.prod(shape):
            raise ValueError("Stored solution grid does not match requested ROI")
    save_solution(
        path,
        "aviris_conservative_fcls",
        result,
        shape,
        transform,
        crs,
        ["snow", "vegetation", "rock", "soil"],
    )
    configure_style()
    extent = (bounds[0] / 1000, bounds[2] / 1000, bounds[1] / 1000, bounds[3] / 1000)
    old = np.load(
        ROOT / "experiments/painter1998/2026-10-01/v1/05_07_image_r001/arrays/aviris_multi_fcls.npz"
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.7))
    panels = [
        (old["normalized"][:, 0], "旧182波段 FCLS（历史有效区）", "Blues", 0, 1),
        (result["normalized"][:, 0], "保守150波段 FCLS（本轮有效区）", "Blues", 0, 1),
        (
            result["normalized"][:, 0] - old["normalized"][:, 0],
            "保守 − 旧 / 覆盖率差",
            "RdBu_r",
            -0.15,
            0.15,
        ),
    ]
    for ax, (array, title, ramp, lower, upper) in zip(axes, panels):
        cmap = sequential_cmap(ramp)
        image = map_axes(ax, array.reshape(shape), extent, crs.to_wkt(), title, cmap, lower, upper)
        fig.colorbar(
            image, ax=ax, shrink=0.72, label="比例 / 1", extend="both" if lower < 0 else "neither"
        )
    axes[0].legend(
        handles=[Patch(color="#d9d9d9", label="NoData / 无有效归一化结果")],
        loc="upper left",
        fontsize=7,
    )
    footer(fig, "2013-05-03 AVIRIS | 30 m | WGS84 / UTM 11N | 原始像元显示；波段与筛选敏感性")
    fig.subplots_adjust(bottom=0.17, wspace=0.35)
    fig.savefig(path / "figures/conservative_snow_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(path / "figures/conservative_snow_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.7))
    for ax, array, title, ramp, lower, upper, unit in zip(
        axes,
        [result["radius_um"], result["rms"] * 100, ((result["qa"] & 16) != 0).astype(float)],
        ["候选最低RMS半径", "150波段训练光谱RMS", "库边界标记（0正常，1边界）"],
        ["viridis", "magma", "cividis"],
        [30, 0, 0],
        [1100, 5, 1],
        ["μm", "反射率百分点", "标记"],
    ):
        array = array.copy()
        array[~valid.ravel()] = np.nan
        if upper == 1:
            cmap = ListedColormap(["#ebebeb", "#e69f00"])
            cmap.set_bad("#777777")
            image = map_axes(
                ax,
                array.reshape(shape),
                extent,
                crs.to_wkt(),
                "粒径候选库边界诊断",
                cmap,
                0,
                1,
                norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
            )
            ax.legend(
                handles=[
                    Patch(color="#ebebeb", label="未报告库边界（含低雪）"),
                    Patch(color="#e69f00", label="最佳候选位于30或1100 μm"),
                    Patch(color="#777777", label="无有效输入"),
                ],
                loc="upper left",
                fontsize=7,
            )
        else:
            image = map_axes(
                ax,
                array.reshape(shape),
                extent,
                crs.to_wkt(),
                title,
                sequential_cmap(ramp),
                lower,
                upper,
            )
            fig.colorbar(
                image, ax=ax, shrink=0.72, label=unit, extend="max" if upper == 5 else "neither"
            )
            ax.legend(
                handles=[
                    Patch(
                        color="#d9d9d9",
                        label="NoData / 粒径未定义" if upper == 1100 else "NoData / 无有效输入",
                    )
                ],
                loc="upper left",
                fontsize=7,
            )
    footer(
        fig,
        "2013-05-03 AVIRIS | 30 m | WGS84 / UTM 11N | 原始像元显示 | 低雪/小分母为空；边界为库选择状态；公共留出波段用于评分",
    )
    fig.subplots_adjust(bottom=0.17, wspace=0.35)
    fig.savefig(path / "figures/conservative_grain_rms_boundary.png", dpi=300, bbox_inches="tight")
    fig.savefig(path / "figures/conservative_grain_rms_boundary.svg", bbox_inches="tight")
    plt.close(fig)
    metrics = {
        "usable_band_count": int(mask.sum()),
        "valid_input_n": int(valid.sum()),
        "new_result": summarize_result(result),
        "cover_sensitivity_vs_previous_model": paired_metrics(
            old["normalized"][:, 0], result["normalized"][:, 0]
        ),
        "previous_source": "v1/05_07_image_r001; previous native-valid coverage rule differs at edges",
    }
    (path / "README.md").write_text(
        "# 保守波段的全ROI诊断地图\n\n在留出波段实验后，完整采用预先规定的150个波段重新解混，模型参数保持预定设置。输入native覆盖率>=.95，63k量级有效30m像元；与v1边缘筛选有细微不同，双方共同有限区域比较。四类物理端元＋零阴影；FCLS；相同30–1100μm候选库。\n\n![图1](figures/conservative_snow_comparison.png)\n\n图1：旧与保守波段雪比例及差值。共同0–1色标，差值±0.15超限有色标延伸，灰色为NoData；按原始像元显示。差值用于波段与边缘筛选敏感性分析。\n\n![图2](figures/conservative_grain_rms_boundary.png)\n\n图2：候选粒径、训练光谱RMS和边界标记。粒径在低雪比例或近全阴影时为空；图件展示模型候选参数；方法比较采用stage12的公共留出波段评分。\n",
        encoding="utf-8",
    )
    finish_run(path, metrics)
    if render_from is not None:
        with (path / "README.md").open("a", encoding="utf-8") as stream:
            stream.write("\n本版本读取已完成解混结果重新制图，未重复拟合或修改源结果。\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-from", type=Path)
    args = parser.parse_args()
    run(args.render_from.resolve() if args.render_from else None)
