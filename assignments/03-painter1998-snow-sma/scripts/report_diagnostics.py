"""由教学输入重绘 WW80 物理图和固定粒径的空间块区间图。"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix import teaching as t


def main() -> None:
    """保存报告使用的 PNG/SVG；输入与 Notebook 相同，不回写报告。"""
    t.configure_plots()
    output = Path(__file__).resolve().parents[1] / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    forward = t.load_forward()
    for name, figure in (
        ("ww80_spectra", t.plot_optical_spectra(forward)),
        ("ww80_broadband", t.plot_broadband_albedo(forward)),
    ):
        figure.set_size_inches(10, 4.2)
        for item in figure.findobj(Text):
            item.set_fontsize(13)
        figure.tight_layout(pad=1.1)
        for extension in ("png", "svg"):
            figure.savefig(output / f"{name}.{extension}", dpi=300)
        plt.close(figure)

    table, paired = t.fixed_radius_statistics(t.load_aviris_case())
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    for ax, metric, color, description in (
        (axes[0], "normalized_snow", t.COLORS[0], t.label("阴影归一化雪", "Normalized snow")),
        (axes[1], "rms", t.COLORS[1], t.label("光谱 RMS", "Spectral RMS")),
    ):
        mean = table[f"{metric}_mean"].to_numpy() * 100
        low = table[f"{metric}_ci_low"].to_numpy() * 100
        high = table[f"{metric}_ci_high"].to_numpy() * 100
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            fmt="o-",
            color=color,
            capsize=4,
            label=description,
        )
        ax.set_xticks(x, table.radius_um.astype(int).astype(str))
        ax.set_xlabel(t.label("固定雪粒半径 / μm", "Fixed snow radius / μm"))
        ax.legend(fontsize=11)
    axes[0].plot(
        x,
        table.raw_snow_mean * 100,
        "s--",
        color=t.COLORS[2],
        label=t.label("原始雪系数", "Raw snow coefficient"),
    )
    axes[0].axhline(0, color=".5", lw=0.8)
    axes[0].set_ylabel(t.label("雪系数或归一化比例 / %", "Snow coefficient or fraction / %"))
    axes[0].legend(fontsize=11)
    axes[1].set_ylabel(t.label("光谱 RMS / 反射率百分点", "Spectral RMS / reflectance pp"))
    fig.suptitle(
        t.label(
            f"共同像元 {paired['n']:,}；30 个空间块；误差棒为均值 95% 区间",
            f"Shared pixels: {paired['n']:,}; 30 blocks; 95% intervals for means",
        ),
        fontsize=13,
    )
    for item in fig.findobj(Text):
        item.set_fontsize(13)
    fig.tight_layout(pad=1.1)
    for extension in ("png", "svg"):
        fig.savefig(output / f"fixed_radius_intervals.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
