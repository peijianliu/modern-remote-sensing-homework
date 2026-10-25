"""吸收面积、稳定选样、固定半径解混和产品一致性检验。"""

import numpy as np
import pytest
import hashlib
import json
from pathlib import Path
import pandas as pd

from snowmix.case_analysis import (
    absorption_area,
    binary_agreement,
    fixed_radius_scan,
    select_quantile_indices,
)


def test_absorption_area_triangle_and_linear_continuum():
    wavelength = np.array([0.90, 0.95, 1.02, 1.09, 1.15])
    spectra = np.array([[1, 1, 0.5, 1, 1], [2, 2, 1, 2, 2]])
    np.testing.assert_allclose(absorption_area(wavelength, spectra), [0.035, 0.035], atol=1e-14)
    np.testing.assert_allclose(absorption_area(wavelength, wavelength[None]), [0], atol=1e-14)
    assert np.isnan(absorption_area(wavelength, np.zeros(5)))
    np.testing.assert_allclose(
        absorption_area(wavelength[::-1], spectra[:, ::-1]), [0.035, 0.035], atol=1e-14
    )
    with pytest.raises(ValueError):
        absorption_area(np.ones(5), spectra)


def test_quantile_selection_is_distinct_and_row_major_on_ties():
    indices = np.array([8, 2, 5, 1, 9])
    np.testing.assert_array_equal(select_quantile_indices(np.ones(5), indices), [1, 2, 5, 8, 9])
    with pytest.raises(ValueError):
        select_quantile_indices([1, np.nan], [0, 1])


def test_fixed_scan_reconstructs_raw_spectrum_and_known_fraction():
    snow = np.array([[1, 0.9, 0.7, 0.4, 0.2, 0.1], [1, 0.8, 0.5, 0.25, 0.1, 0.05]])
    bare = np.full((6, 1), 0.6)
    observed = 0.7 * snow[1] + 0.2 * bare[:, 0]
    fit = fixed_radius_scan(observed, snow, [50, 500], bare, requested_radii=(50, 500))
    np.testing.assert_allclose(fit["fractions"][1, 0], [0.7, 0.2], atol=1e-12)
    np.testing.assert_allclose(fit["normalized"][1, 0, 0], 0.7 / 0.9, atol=1e-12)
    np.testing.assert_allclose(fit["residual"], observed - fit["modeled"], atol=1e-12)
    np.testing.assert_allclose(
        fit["rms"], np.sqrt(np.mean(fit["residual"] ** 2, axis=-1)), atol=1e-12
    )
    light = fixed_radius_scan(
        observed, snow, [50, 500], bare, requested_radii=(50,), include_spectra=False
    )
    assert "residual" not in light and "modeled" not in light
    with pytest.raises(ValueError):
        fixed_radius_scan(observed, snow, [50, 500], bare, requested_radii=(200,))


def test_product_agreement_counts_and_undefined_metrics():
    result = binary_agreement([1, 1, 0, 0], [0.8, 0.1, 0.3, 0], 0.5)
    assert (result["tp"], result["fn"], result["fp"], result["tn"]) == (1, 1, 0, 2)
    assert result["recall"] == 0.5 and result["specificity"] == 1 and result["accuracy"] == 0.75
    empty = binary_agreement([1], [np.nan])
    assert empty["n"] == 0 and empty["recall"] is None and empty["specificity"] is None
    assert binary_agreement([0, 0], [0, 0])["recall"] is None
    with pytest.raises(ValueError):
        binary_agreement([2], [0])


DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


def test_frozen_demo_file_hashes_and_public_input_boundary():
    """交付的文件应与来源清单逐字节一致，原始受限谱与整景文件不随包。"""
    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    for asset in manifest["files"]:
        relative = Path(asset["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = DEMO / relative
        assert path.stat().st_size == asset["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
    assert not any(path.suffix in {".hdf", ".tif", ".nc"} for path in DEMO.iterdir())
    assert not any(
        "granite" in path.name.lower() or "cryumbrept" in path.name.lower()
        for path in DEMO.iterdir()
    )


def test_frozen_fixed_scan_common_domain_and_absorption_values():
    """所有固定半径的公平比较采用同一个可归一化像元集合。"""
    with np.load(DEMO / "aviris_case.npz", allow_pickle=False) as data:
        common = np.isfinite(data["fixed_snow_fraction"]).all(axis=0)
        np.testing.assert_array_equal(common, data["common_normalization_mask"])
        table = pd.read_csv(DEMO / "fixed_radius_summary.csv")
        assert np.all(table.common_normalized_n == int(common.sum()))
        for i in range(len(table)):
            np.testing.assert_allclose(
                table.common_normalized_snow_mean.iloc[i],
                np.mean(data["fixed_snow_fraction"][i][common], dtype=np.float64),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                table.common_rms_mean.iloc[i],
                np.mean(data["fixed_rms"][i][common], dtype=np.float64),
                atol=1e-12,
            )
        np.testing.assert_allclose(
            absorption_area(data["wavelength_um"], data["image_snow_spectra"]),
            data["image_absorption_area_um"],
            atol=1e-12,
        )
        assert len(np.unique(data["background_row_col"], axis=0)) == 3
        assert len(np.unique(data["image_endmember_row_col"], axis=0)) == 5


def test_frozen_one_km_units_remain_unique_and_missing_classes_are_null():
    """原生状态像元只计一次，面积支持与空类别处理可由教学表重算。"""
    units = pd.read_csv(DEMO / "modis_1km_units.csv")
    assert not units.duplicated(["roi", "tile_row_1km", "tile_column_1km"]).any()
    expected_mask = (
        (units.aviris_valid_area_fraction >= 0.95)
        & (units.modis_valid_area_fraction >= 0.95)
        & (units.state_1km != 65535)
    )
    np.testing.assert_array_equal(units.common_valid, expected_mask)
    # 同一正弦网格上的 1 km 单元由四个原始 500 m 单元组成。
    np.testing.assert_allclose(
        units.modis_valid_area_fraction * 4,
        np.round(units.modis_valid_area_fraction * 4),
        atol=1e-7,
    )
    table = pd.read_csv(DEMO / "product_consistency.csv")
    for _, row in table.iterrows():
        selected = units[(units.roi == row.roi) & units.common_valid]
        result = binary_agreement(
            selected[f"flag{int(row.bit)}"], selected[f"{row.model}_fsca"], row.threshold
        )
        for name in ["n", "tp", "fn", "fp", "tn", "reference_positive", "reference_negative"]:
            assert int(row[name]) == result[name]
        for name in ["recall", "specificity", "accuracy"]:
            if result[name] is None:
                assert pd.isna(row[name])
            else:
                assert row[name] == pytest.approx(result[name])
