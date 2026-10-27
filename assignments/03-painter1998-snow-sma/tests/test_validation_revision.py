"""本轮科学解释、跨ROI去重和完整公开输入重建的回归。"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest

from snowmix import teaching as t
from snowmix.linear import mgs_operator, select_snow
from snowmix.modis_validation import evaluate_flags


def test_excluded_spectra_cannot_be_downloaded_or_read(tmp_path):
    from scripts.acquire_sources import download
    from snowmix.spectra import read_ecostress

    for name in ("granite", "soil"):
        with pytest.raises(PermissionError, match="PUBLIC_DATA_POLICY"):
            download(name)
    for name in ("granite_h2.txt", "cryumbrept_87p3855.txt"):
        with pytest.raises(PermissionError, match="PUBLIC_DATA_POLICY"):
            read_ecostress(tmp_path / name)


def test_legacy_report_caller_passes_its_configured_experiment_batch(tmp_path, monkeypatch):
    from snowmix import __main__ as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    config = {"project": "painter1998", "run_date": "2026-10-27", "version": "v4"}
    arguments = cli.verification_arguments(config)
    assert arguments[-2:] == [
        "--experiments-dir",
        str(tmp_path / "experiments/painter1998/2026-10-27/v4"),
    ]


def test_positive_observation_can_need_negative_unconstrained_snow():
    snow = np.array([[0.9, 0.6, 0.2]])
    background = np.full(3, 0.4)
    observed = -0.1 * snow[0] + 0.8 * background
    assert (observed > 0).all()
    original = select_snow(observed, snow, [200], background)
    constrained = select_snow(observed, snow, [200], background, solver="fcls")
    assert original.fractions[0, 0] == pytest.approx(-0.1)
    assert int(original.qa[0]) & 64
    assert constrained.fractions.min() >= 0
    assert constrained.rms[0] > original.rms[0]
    # 背景正交补上的投影直接给出有符号雪系数，无需改变MGS算法。
    B = background[:, None]
    direction = snow[0] - B @ (mgs_operator(B) @ snow[0])
    coefficient = (observed @ direction) / (direction @ direction)
    assert coefficient == pytest.approx(original.fractions[0, 0])


def test_overlapping_roi_units_are_counted_once_with_predeclared_support_rule():
    table = pd.read_csv(t.demo_directory() / "modis_1km_units.csv")
    unique = t.unique_native_units(table)
    assert len(table) == 420 and len(unique) == 406
    assert not unique.duplicated(["tile_row_1km", "tile_column_1km"]).any()
    assert t.validation_ledger(unique).n.to_list() == [406, 406, 100, 17]
    assert t.state_flag_ledger(table).flag15_positive.to_list() == [20, 10, 0]
    bad = pd.concat([table, table.iloc[:1]])
    with pytest.raises(ValueError, match="Duplicated"):
        t.unique_native_units(bad)
    duplicate = (
        table[table.duplicated(["tile_row_1km", "tile_column_1km"], keep=False)].iloc[:1].copy()
    )
    duplicate["roi"] = "third"
    duplicate["state_1km"] += 1
    with pytest.raises(ValueError, match="disagree"):
        t.unique_native_units(pd.concat([table, duplicate]))


def test_independent_flags_use_native_cells_and_separate_reference_threshold():
    units = pd.DataFrame(
        {
            "scene": ["test"] * 4,
            "tile_row_1km": [0, 0, 1, 1],
            "tile_column_1km": [0, 1, 0, 1],
            "reference_fsca": [0.9, 0.2, 0, 0.8],
            "reference_coverage": [1, 1, 1, 0.5],
            "state_clear_no_adjacent": [1, 1, 1, 1],
            "flag12": [1, 0, 0, 1],
            "flag15": [1, 1, 0, 1],
        }
    )
    result = evaluate_flags(units)
    row = result[(result.bit == 15) & (result.reference_threshold == 0.1)].iloc[0]
    assert (row.tp, row.fn, row.fp, row.tn) == (2, 0, 0, 1)
    assert row.n == 3
    with pytest.raises(ValueError, match="Repeated"):
        evaluate_flags(pd.concat([units, units.iloc[:1]]))


@pytest.mark.requires_gis
def test_sparse_source_footprints_keep_original_indices_and_missing_support():
    from affine import Affine
    from snowmix.geo import area_weights, aggregate

    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    kwargs = dict(
        source_transform=Affine(30, 0, 0, 0, -30, 120),
        source_shape=(4, 4),
        source_crs=32611,
        destination_transform=Affine(60, 0, 0, 0, -60, 120),
        destination_shape=(2, 2),
        destination_crs=32611,
    )
    dense = area_weights(**kwargs)
    sparse = area_weights(**kwargs, source_valid=valid)
    values = np.full(16, 0.4)
    expected, expected_support = aggregate(dense, values, valid)
    actual, support = aggregate(sparse, values)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(support, expected_support)
    assert support[0] == pytest.approx(0.75)
    assert sparse.shape == (4, 16)
    assert area_weights(**kwargs, source_valid=np.zeros((4, 4), bool)).nnz == 0
    with pytest.raises(ValueError, match="boolean"):
        area_weights(**kwargs, source_valid=valid.astype(int))


def test_hdf_login_html_is_rejected_before_matlab(tmp_path):
    from scripts.prepare_modis_hdf import prepare

    path = tmp_path / "saved.hdf"
    path.write_bytes(b"<html>Earthdata Login</html>")
    with pytest.raises(ValueError, match="HDF4"):
        prepare(path, tmp_path / "cache/fields.mat")
    assert not (tmp_path / "cache").exists()


@pytest.mark.full_data
def test_full_public_aviris_rebuild_reselects_endmembers_and_all_coefficients(tmp_path):
    source = os.environ.get("PAINTER_SOURCE_ROOT")
    if not source:
        pytest.skip("Set PAINTER_SOURCE_ROOT to the complete public input workspace")
    from scripts.rebuild_public_case import rebuild

    output = tmp_path / "aviris"
    metrics = rebuild(Path(source), output)
    assert metrics["background_and_snow_endmembers_reselected"]
    assert metrics["pixels"] == 62908 and metrics["common_n"] == 62861
    with (
        np.load(output / "coefficient_diagnostics.npz") as current,
        np.load(t.demo_directory() / "coefficient_diagnostics.npz") as frozen,
    ):
        for name in ("fractions", "shade", "rms"):
            np.testing.assert_allclose(
                current[name], frozen[name], rtol=2e-5, atol=2e-6, equal_nan=True
            )
        np.testing.assert_array_equal(current["valid_mask"], frozen["valid_mask"])


@pytest.mark.full_data
def test_full_public_modis_reference_rebuild_retains_snow_and_no_snow(tmp_path):
    source = os.environ.get("PAINTER_SOURCE_ROOT")
    fields = os.environ.get("PAINTER_MODIS_FIELD_ROOT")
    if not source or not fields:
        pytest.skip("Set public source and verified HDF field roots for the complete rebuild")
    from scripts.prepare_modis_validation import prepare

    output = tmp_path / "modis"
    prepare(Path(source), Path(fields), output)
    support = pd.read_csv(output / "modis_external_support.csv")
    strict = support[support.tier == "strict7"]
    np.testing.assert_array_equal(strict.common_n, [1416, 1817])
    np.testing.assert_array_equal(strict.reference_positive_ge_0p1, [463, 22])
    assert (strict.no_snow_n > 0).all()
    metrics = pd.read_csv(output / "modis_external_metrics.csv")
    bins = metrics[
        (metrics.scene == 20160213) & (metrics.tier == "strict7") & (metrics.method == "multi_FCLS")
    ].set_index("stratum")
    assert bins.loc["low_mixed", "n"] == 344
    assert bins.loc["snow_dominated", "n"] == 119
    flags = pd.read_csv(output / "modis_external_flags.csv")
    primary = flags[(flags.reference_threshold == 0.1) & (flags.bit == 15)]
    np.testing.assert_array_equal(primary.reference_positive, [123, 5])
    assert (primary.reference_negative > 0).all()
