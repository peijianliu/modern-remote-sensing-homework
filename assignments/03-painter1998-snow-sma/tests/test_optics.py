"""WW80 的极限、广播与冻结 MATLAB 结果的精度回归。"""

import numpy as np
import pytest

from snowmix.optics import ww80_semi_infinite
from snowmix.teaching import load_forward, load_optical_inputs, recompute_ww80


def test_absorbing_conservative_and_isotropic_limits():
    np.testing.assert_array_equal(ww80_semi_infinite(0, [0.0, 0.9], 0.5), [0, 0])
    np.testing.assert_allclose(ww80_semi_infinite(1, [0, 0.9, 0.999999], 0.5), 1, atol=1e-14)
    xi = np.sqrt(3 * (1 - 0.8))
    expected = 0.8 / (1 + 2 * xi / 3) / (1 + xi * 0.5)
    assert ww80_semi_infinite(0.8, 0, 0.5) == pytest.approx(expected)
    assert np.isfinite(ww80_semi_infinite(0.9, 0.8, 0))


def test_broadcasting_and_float64_computation():
    omega = np.array([[0.9], [0.99]], dtype=np.float32)
    result = ww80_semi_infinite(omega, 0.8, [0, 0.5, 1])
    assert result.shape == (2, 3) and result.dtype == np.float64
    assert np.all(np.diff(result, axis=1) < 0)


@pytest.mark.parametrize(
    "omega,g,mu",
    [
        (-0.1, 0.8, 0.5),
        (1.1, 0.8, 0.5),
        (0.9, -0.1, 0.5),
        (0.9, 1, 0.5),
        (0.9, 0.8, -0.1),
        (0.9, 0.8, 1.1),
        (np.nan, 0.8, 0.5),
        (0.9, np.inf, 0.5),
        ([], [], []),
    ],
)
def test_invalid_inputs(omega, g, mu):
    with pytest.raises(ValueError):
        ww80_semi_infinite(omega, g, mu)


def test_nonbroadcastable_inputs():
    with pytest.raises(ValueError):
        ww80_semi_infinite([0.9, 0.8], [0.8, 0.7, 0.6], 0.5)


def test_full_grid_and_solar_weighted_precision_with_uint8_angles():
    forward, optics = load_forward(), load_optical_inputs()
    # 冻结角度为 uint8；正确转换精度应与 float64 输入完全一致。
    result = recompute_ww80(optics, forward["angles_deg"].astype(np.uint8))
    precise = recompute_ww80(optics, forward["angles_deg"].astype(np.float64))
    np.testing.assert_array_equal(result, precise)
    assert result.shape == (148, 480, 3, 2) and result.size == 426240
    assert np.max(np.abs(result - forward["ww_albedo"])) < 1e-6
    flux = forward["solar_flux"].T[None, :, :, None]
    broadband = np.sum(result * flux, axis=1) / np.sum(flux, axis=1)
    np.testing.assert_allclose(broadband, forward["broadband_ww"], atol=1e-6, rtol=0)
