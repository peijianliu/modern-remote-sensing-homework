"""检查近共线求解、配置执行、SSA 单位和空间区块统计。"""

import json
import numpy as np
import pytest
from snowmix.artifacts import ROOT
from snowmix.linear import fcls_batch
from snowmix.experiments import synthetic_inputs
from snowmix.validation import ssa_radius_um, parse_ssa_text, interval_mask, block_bootstrap_delta


def test_near_collinear_fcls_exact_observation():
    rng = np.random.default_rng(2026)
    v = rng.uniform(0.1, 0.9, 20)
    a = np.column_stack((v, v + 1e-7 * rng.normal(0, 0.3, 20)))
    truth = np.array([[0.2, 0.4]])
    fit = fcls_batch(truth @ a.T, a)
    np.testing.assert_allclose(fit, truth, atol=1e-7)
    # 系数容差考虑约 4e7 的条件数。
    # 以反射率 RMS 判定重建精度，兼容 MKL/OpenBLAS 的浮点累加差异。
    assert np.sqrt(np.mean((truth @ a.T - fit @ a.T) ** 2)) < 1e-8


@pytest.mark.full_data
@pytest.mark.skipif(
    not (ROOT / "data/cache/forward_models.mat").is_file(),
    reason="Optional full MATLAB reference is not part of teaching inputs",
)
def test_requested_radius_grid_and_ice_option_are_used():
    c = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
    c["synthetic"].update(radius_min_um=200, radius_max_um=300, radius_step_um=50)
    c["forward"]["ice_ri"] = 2
    _, radii, snow08, _, _, source = synthetic_inputs(c)
    np.testing.assert_array_equal(radii, [200, 250, 300])
    assert source["ice_option"] == 2
    c["forward"]["ice_ri"] = 1
    _, _, snow84, *_ = synthetic_inputs(c)
    assert np.max(abs(snow08 - snow84)) > 1e-5
    c["forward"]["depth_m"] = 2
    with pytest.raises(ValueError, match="differs from request"):
        synthetic_inputs(c)
    c["forward"]["depth_m"] = 100
    c["synthetic"]["radius_min_um"] = 205
    with pytest.raises(ValueError, match="cache"):
        synthetic_inputs(c)


def test_ssa_unit_and_comma_decimal_acquisition_only():
    assert ssa_radius_um(30) == pytest.approx(109.05125408942203)
    text = (
        "Snow condition/type identifier: test\nOverall calibration quality: Good\n"
        "1\t120\t40\t999\t999\nSample number\tSample signal\tReflectance\tSpecific surface area (SSA)\tDepth\n"
        "1\t100,1\t40,2\t30,5\t76\n2\t80\t20\t12\t73\n"
    )
    name, table, calibration = parse_ssa_text(text)
    assert (name, calibration, len(table)) == ("test", "Good", 2)
    assert table.ssa_m2_kg.iloc[0] == 30.5
    with pytest.raises(ValueError):
        ssa_radius_um(0)


def test_prespecified_band_mask_and_block_uncertainty():
    np.testing.assert_array_equal(
        interval_mask(np.array([0.5, 1.4, 1.6, 1.9]), [[0.42, 1.3], [1.5, 1.75]]), [1, 0, 1, 0]
    )
    blocks = np.repeat(np.arange(8), 10)
    delta = np.full(80, -0.01)
    result = block_bootstrap_delta(blocks, delta, 1998)
    np.testing.assert_allclose(result["interval_95"], [-0.01, -0.01])
