"""由唯一 1 km 教学表重绘综合报告的面积支持与阳性样本图。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from snowmix import teaching as t


def main() -> None:
    """保存 300 dpi PNG 与可编辑 SVG；不修改输入表或报告。"""
    t.configure_plots()
    units = pd.read_csv(t.demo_directory() / "modis_1km_units.csv")
    ledger = t.validation_ledger(units)
    figure = t.plot_validation_design(ledger)
    figure.set_size_inches(8.4, 3.8)
    percent = f"{ledger.minimum_coverage.iloc[0] * 100:g}%"
    for axis in figure.axes:
        axis.set_xticks(
            range(3),
            [
                t.label("候选", "Candidates"),
                f"AVIRIS\n≥{percent}",
                t.label(f"共同\n≥{percent}", f"Both\n≥{percent}"),
            ],
        )
    figure.axes[1].set_ylim(-15, 145)
    # 图按 16.2 cm 嵌入 A4；增大绘图字体以保证缩放后约 10 pt 可读。
    from matplotlib.text import Text

    for item in figure.findobj(Text):
        item.set_fontsize(13)
    figure.tight_layout(pad=1.2)
    output = Path(__file__).resolve().parents[1] / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(output / f"validation_support.{extension}", dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
