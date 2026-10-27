"""从公开教学输入重绘本轮新增实验图，报告与Notebook使用相同数组。"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import json

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from snowmix import teaching as t


def main() -> None:
    """保存300 dpi PNG与SVG；只写报告图件目录。"""
    t.configure_plots()
    demo = t.demo_directory()
    output = CASE / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    spatial = json.loads((demo / "spatial_metadata.json").read_text(encoding="utf-8"))
    figures = {
        "mismatch_validation": t.plot_mismatch_summary(pd.read_csv(demo / "mismatch_summary.csv")),
        "coefficient_status": t.plot_coefficient_status(t.load_coefficient_diagnostics(), spatial),
        "modis_spatial_support": t.plot_modis_support(
            pd.read_csv(demo / "modis_1km_units.csv"), spatial
        ),
        "public_transfer_validation": t.plot_transfer_comparison(
            pd.read_csv(demo / "transfer_metrics.csv")
        ),
        "independent_modis_validation": t.plot_modis_external(
            pd.read_csv(demo / "modis_external_metrics.csv")
        ),
    }
    for name, figure in figures.items():
        t.readable_figure(figure, minimum_fontsize=14)
        for extension in ("png", "svg"):
            figure.savefig(output / (name + "." + extension), dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
