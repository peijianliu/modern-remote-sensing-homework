import numpy as np
import pytest
from scipy.optimize import minimize
from snowmix.linear import mgs_operator, fcls_batch, select_snow
from snowmix.spectra import wavelength_to_um, gaussian_band_matrix, response_matrix
from snowmix.modis import quality_mask


def test_original_known_fractions_and_svd():
    rng = np.random.default_rng(1998)
    a = rng.uniform(0.05, 1, (40, 4))
    fractions = np.array([[0.3, 0.1, 0.2, 0.15], [1.2, -0.2, 0, 0], [0, 0, 0, 0]])
    y = fractions @ a.T
    operator = mgs_operator(a)
    np.testing.assert_allclose(y @ operator.T, fractions, atol=1e-12)
    np.testing.assert_allclose(operator @ y.T, np.linalg.lstsq(a, y.T, rcond=None)[0], atol=1e-12)
    result = select_snow(y, a[:, :1].T, [200], a[:, 1:])
    assert result.rms.max() < 1e-12
    assert result.shade[0] == pytest.approx(0.25)
    assert result.normalized[1, 0] > 1  # 保留无约束解的系数
    assert np.isnan(result.radius_um[2])


def test_fcls_against_independent_optimizer():
    rng = np.random.default_rng(31)
    a = rng.uniform(0.1, 1, (9, 4))
    y = rng.uniform(-0.1, 0.9, (6, 9))
    result = fcls_batch(y, a)
    for observation, actual in zip(y, result):
        optimized = minimize(
            lambda f: np.sum((observation - a @ f) ** 2),
            np.full(4, 0.15),
            jac=lambda f: 2 * a.T @ (a @ f - observation),
            bounds=[(0, 1)] * 4,
            constraints={
                "type": "ineq",
                "fun": lambda f: 1 - f.sum(),
                "jac": lambda f: -np.ones(4),
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        assert optimized.success
        np.testing.assert_allclose(actual, optimized.x, atol=2e-6)
    assert result.min() >= 0
    assert result.sum(axis=1).max() <= 1 + 1e-12


def test_candidate_selection_and_invalid_inputs():
    snow = np.array([[1, 0.8, 0.4, 0.1], [1, 0.6, 0.2, 0.05]])
    quartz = np.array([0.7, 0.7, 0.7, 0.7])
    y = np.array([0.6 * snow[1] + 0.4 * quartz, quartz, [np.nan, 1, 1, 1]])
    result = select_snow(y, snow, [50, 500], quartz)
    assert result.radius_um[0] == 500
    assert result.normalized[0, 0] == pytest.approx(0.6)
    assert np.isnan(result.radius_um[1])
    assert result.qa[2] & 1
    with pytest.raises(ValueError):
        mgs_operator(np.ones((4, 2)))


def test_grain_threshold_tolerates_roundoff_only():
    snow = np.array([[1, 0.6, 0.2, 0.05]])
    quartz = np.array([0.7, 0.7, 0.7, 0.7])
    fractions = np.array([0.1, 0.1 - 1e-7])
    y = 0.7 * (fractions[:, None] * snow + (1 - fractions[:, None]) * quartz)
    result = select_snow(y, snow, [500], quartz)
    assert result.radius_um[0] == 500
    assert np.isnan(result.radius_um[1])


@pytest.mark.requires_gis
def test_noninteger_aggregation_partial_coverage_and_mass():
    from affine import Affine
    from snowmix.geo import area_weights, aggregate

    source = Affine(3, 0, 0, 0, -3, 12)
    destination = Affine(5, 0, 0, 0, -5, 12)
    w = area_weights(source, (4, 4), "EPSG:32611", destination, (3, 3), "EPSG:32611")
    constant, coverage = aggregate(w, np.full(16, 0.42))
    np.testing.assert_allclose(constant, 0.42, atol=1e-12)
    assert coverage[-1] == pytest.approx(4 / 25)
    assert np.sum(w) * 25 == pytest.approx(16 * 9)
    values = np.arange(16.0)
    weighted, coverage = aggregate(w, values)
    assert np.sum(weighted[:, 0] * coverage) * 25 == pytest.approx(np.sum(values) * 9)
    valid = np.ones(16, dtype=bool)
    valid[0] = False
    _, smaller = aggregate(w, values, valid)
    assert smaller[0] < coverage[0]


def test_wavelength_response_and_mask():
    np.testing.assert_allclose(wavelength_to_um([500, 1000], "nm"), [0.5, 1])
    with pytest.raises(ValueError):
        wavelength_to_um([1], "feet")
    wavelength = np.linspace(0.4, 2.5, 224)
    weights = gaussian_band_matrix(wavelength, np.array([0.65, 1.03]), np.array([0.01, 0.01]))
    np.testing.assert_allclose(np.ones(224) @ weights, 1)
    mask = np.ones(224, bool)
    mask[(wavelength > 0.62) & (wavelength < 0.68)] = False
    response, coverage = response_matrix(
        wavelength, [np.array([0.62, 0.65, 0.68])], [np.array([0, 1, 0])], mask
    )
    assert coverage[0] < 0.99 and np.isnan(response).all()


def test_modis_snow_is_not_cloud():
    raw = np.full((4, 7), 5000, np.int16)
    qc = np.full(4, 1 << 30, np.uint32)
    state = np.array([0, (1 << 12) | (1 << 15), 1, 1 << 2], np.uint16)
    np.testing.assert_array_equal(quality_mask(raw, qc, state), [True, True, False, False])
    qc[0] |= 1 << 2
    assert not quality_mask(raw, qc, state)[0]
    raw[1, 3] = -28672
    assert not quality_mask(raw, qc, state)[1]
