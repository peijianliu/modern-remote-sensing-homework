"""共同分母、配对区块区间与面积阈值的回归检查。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from snowmix.case_analysis import paired_block_means
from snowmix import teaching as t

CASE = Path(__file__).resolve().parents[1]


def test_block_means_keep_pixel_weights_and_pairing():
    # 六块的像元数不同；不能把六个块均值等权平均。
    positions = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    value = np.array([0, 0, 0, 10, 10, 10, 10, 10], dtype=float)
    data = np.column_stack((value, value + 2))
    result = paired_block_means(data, positions, block_size_pixels=1, repeats=1000, seed=1998)
    np.testing.assert_allclose(result["mean"], [6.25, 8.25])
    np.testing.assert_allclose(np.diff(result["bootstrap_means"], axis=1), 2, atol=1e-14)
    repeated = paired_block_means(data, positions, block_size_pixels=1, repeats=1000, seed=1998)
    np.testing.assert_array_equal(result["interval_95"], repeated["interval_95"])


@pytest.mark.parametrize(
    "kind", ["nan", "duplicate", "fractional", "few_blocks", "bad_repeats", "bad_size"]
)
def test_block_inputs_are_explicit(kind):
    positions = np.column_stack((np.arange(6) * 50, np.zeros(6)))
    values = np.ones((6, 2))
    kwargs = {}
    if kind == "nan":
        values[0, 0] = np.nan
    elif kind == "duplicate":
        positions[0] = positions[1]
    elif kind == "fractional":
        positions[0, 0] = 0.5
    elif kind == "few_blocks":
        positions[:, 0] = np.arange(6)
    elif kind == "bad_repeats":
        kwargs["repeats"] = 1
    else:
        kwargs["block_size_pixels"] = 0
    with pytest.raises(ValueError):
        paired_block_means(values, positions, **kwargs)


def test_actual_denominators_and_bootstrap_use_shared_pixels():
    case = t.load_aviris_case()
    table, paired = t.fixed_radius_statistics(case)
    expected = pd.read_csv(CASE / "data/demo/fixed_radius_uncertainty.csv")
    np.testing.assert_allclose(table, expected, atol=1e-8, rtol=1e-6)
    assert table.common_n.eq(60949).all() and table.denominator_pass_fraction_common.eq(1).all()
    assert table.denominator_p05.min() > 0.29
    assert table.negative_snow_fraction.between(0.56, 0.58).all()
    assert paired["spatial_blocks"] == 30 and paired["repeats"] == 1000
    assert paired["paired_delta"]["rms"]["interval_95"][1] < 0
    assert paired["paired_delta"]["normalized_snow"]["interval_95"][1] < 0


def test_coverage_sensitivity_retains_same_unique_units():
    units = pd.read_csv(CASE / "data/demo/modis_1km_units.csv")
    output = t.coverage_sensitivity(units)
    np.testing.assert_array_equal(output.n, [114, 17, 107, 17, 100, 17])
    np.testing.assert_array_equal(output.flag15_positive, [21, 0, 20, 0, 20, 0])
    assert output.flag12_positive.eq(0).all()
    pd.testing.assert_frame_equal(
        output, pd.read_csv(CASE / "data/demo/coverage_sensitivity.csv"), check_dtype=False
    )
    with pytest.raises(ValueError):
        t.coverage_sensitivity(units, thresholds=(0.95, 0.95))


def test_complete_reference_not_required_or_distributed():
    assert not (CASE / "data/demo/forward_models.mat").exists()
    assert not (
        CASE.parents[1] / "site/assignments/03-painter1998/assets/mathjax/tex-svg-full.js"
    ).exists()
