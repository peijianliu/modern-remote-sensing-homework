"""教学统计的有限值、共同支持和分层样本检查。"""

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from snowmix import teaching


def test_difference_metrics_uses_common_finite_values():
    result = teaching.difference_metrics([0, 1, np.nan, 3], [0.1, 0.8, 2, np.inf])
    assert result["n"] == 2
    assert result["bias"] == pytest.approx(-0.05)
    assert result["mae"] == pytest.approx(0.15)
    assert result["rmse"] == pytest.approx(np.sqrt(0.025))
    empty = teaching.difference_metrics([np.nan], [1])
    assert empty["n"] == 0 and np.isnan(empty["rmse"])


def agreement_table():
    """四个空间单元中仅三个满足共同有效条件。"""
    return pd.DataFrame(
        {
            "roi": ["example"] * 4,
            "tile_row_1km": [0, 0, 1, 1],
            "tile_column_1km": [0, 1, 0, 1],
            "common_valid": [True, True, True, False],
            "flag12": [1, 1, 0, 1],
            "flag15": [0, 0, 0, 1],
            "aviris_fsca": [0.8, 0.2, 0, 0.9],
            "modis_fsca": [0.6, 0.3, 0.05, 0.8],
        }
    )


def test_product_agreement_uses_unique_common_units_and_keeps_missing_classes():
    result = teaching.product_consistency(agreement_table())
    assert len(result) == 12
    assert (result.n == 3).all()
    row = result[
        (result.flag == "flag12") & (result.model == "aviris_fsca") & (result.threshold == 0.5)
    ].iloc[0]
    assert (row.tp, row.fn, row.fp, row.tn) == (1, 1, 0, 1)
    assert row.recall == 0.5
    assert result[result.flag == "flag15"].recall.isna().all()


def test_product_agreement_rejects_replicated_one_km_flag():
    table = agreement_table()
    with pytest.raises(ValueError, match="Duplicated 1 km"):
        teaching.product_consistency(pd.concat([table, table.iloc[:1]], ignore_index=True))


def test_worldview_recomputes_shared_mask_and_strata(tmp_path):
    samples = pd.DataFrame(
        {
            "reference_fsca": [0, 0.5, 1, 0.8],
            "stratum": ["no_snow", "mixed", "full_snow", "mixed"],
            "common_four_methods_normalization_valid": [True, True, True, False],
        }
    )
    for method in (
        "fixed200_six_bands",
        "multi_six_bands",
        "fixed200_four_bands",
        "multi_four_bands",
    ):
        samples[method + "_snow"] = [0, 0.4, 0.9, 0]
    samples.to_csv(tmp_path / "example_sample_pixels.csv", index=False)
    metrics = teaching.worldview_metrics(tmp_path)
    assert len(metrics) == 16
    all_pixels = metrics[metrics.stratum == "all"]
    assert (all_pixels.n == 3).all()
    np.testing.assert_allclose(all_pixels.rmse, np.sqrt(0.02 / 3))
    assert (metrics[metrics.stratum == "mixed"].n == 1).all()


def test_english_labels_are_selected_when_no_supported_cjk_font(monkeypatch):
    monkeypatch.setattr(teaching.font_manager.fontManager, "ttflist", [])
    assert teaching.configure_plots() is False
    assert teaching.label("波长", "Wavelength") == "Wavelength"


def test_plot_spectrum_breaks_across_removed_atmospheric_bands():
    fig, ax = teaching.plt.subplots()
    wavelength = np.array([1.30, 1.31, 1.49, 1.50])
    values = np.array([0.2, 0.3, 0.4, 0.5])
    line = teaching.plot_spectrum(ax, wavelength, values)[0]
    assert np.isnan(line.get_ydata()).sum() == 1
    np.testing.assert_array_equal(values, [0.2, 0.3, 0.4, 0.5])
    teaching.plt.close(fig)


def test_offline_html_removes_remote_scripts_and_retains_local_formula_component():
    path = Path(__file__).resolve().parents[1] / "scripts/build_notebooks.py"
    spec = importlib.util.spec_from_file_location("case_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (
        '<html><head><script src="https://example.invalid/require.js"></script>'
        '<link rel="stylesheet" href="https://example.invalid/style.css"></head>'
        "<body><p>$$E_c=R_c-\\widehat R_c$$</p>"
        '<a href="02_fig3_sma.ipynb">Next</a></body></html>'
    )
    result = module.offline_html(source, "01_snow_optics")
    assert "example.invalid" not in result
    assert 'src="assets/mathjax/tex-svg-full.js"' in result
    assert 'href="02_fig3_sma.html"' in result
    assert 'href="notebooks/01_snow_optics.ipynb"' in result
    assert "E_c=R_c" in result
