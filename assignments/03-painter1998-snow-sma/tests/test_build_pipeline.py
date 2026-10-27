"""站点输出边界、无输出源码、练习副本和样本压缩的回归检查。"""

import gzip
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from scripts import build_notebooks as builder
from snowmix import teaching as t

CASE = Path(__file__).resolve().parents[1]


def test_output_root_rejects_source_ancestors_and_source_directories(tmp_path):
    for path in (
        builder.REPOSITORY,
        builder.SITE_SOURCE,
        CASE / "notebooks",
        CASE / "data",
        CASE / "snowmix",
        builder.REPOSITORY.parent,
        CASE / "build",
        CASE / "build/notebooks",
        CASE / "build/workspace/site",
    ):
        with pytest.raises(ValueError):
            builder.validate_output(path)
    assert builder.validate_output(builder.REPOSITORY / "build/site").name == "site"
    assert builder.validate_output(tmp_path / "site").name == "site"


def test_nonempty_output_with_unknown_files_is_preserved(tmp_path):
    (tmp_path / "my_notes.txt").write_text("private exercise", encoding="utf-8")
    with pytest.raises(ValueError, match="Nonempty"):
        builder.build(tmp_path)
    assert (tmp_path / "my_notes.txt").read_text(encoding="utf-8") == "private exercise"


def test_interactive_workspace_never_overwrites_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "CASE", tmp_path)
    (tmp_path / "notebooks").mkdir()
    for name in builder.NOTEBOOK_NAMES:
        (tmp_path / "notebooks" / (name + ".ipynb")).write_text("source", encoding="utf-8")
    workspace = builder.prepare_workspace()
    target = workspace / (builder.NOTEBOOK_NAMES[0] + ".ipynb")
    target.write_text("my exercise", encoding="utf-8")
    builder.prepare_workspace()
    assert target.read_text(encoding="utf-8") == "my exercise"


def test_source_notebooks_have_no_execution_artifacts():
    for path in (CASE / "notebooks").glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None and cell.outputs == []
            assert "execution" not in cell.metadata


def test_compressed_samples_keep_exact_original_bytes_rows_and_columns():
    manifest = json.loads((CASE / "data/demo/manifest.json").read_text(encoding="utf-8"))
    compressed = [
        entry
        for entry in manifest["files"]
        if entry["path"].startswith("LC08_") and entry["path"].endswith(".csv.gz")
    ]
    assert len(compressed) == 2
    for entry in compressed:
        path = CASE / "data/demo" / entry["path"]
        raw = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == entry["uncompressed_sha256"]
        assert len(raw) == entry["uncompressed_bytes"]
        assert pd.read_csv(path).shape == (12000, 44)


def test_duplicate_csv_and_gzip_scene_is_rejected(tmp_path):
    for suffix in (".csv", ".csv.gz"):
        (tmp_path / ("example_sample_pixels" + suffix)).write_bytes(b"not read")
    with pytest.raises(ValueError, match="Duplicate scene"):
        t.worldview_metrics(tmp_path)


def test_filter_ledger_is_nested_and_retains_valid_negative_units():
    units = pd.read_csv(CASE / "data/demo/modis_1km_units.csv")
    ledger = t.validation_ledger(units)
    np.testing.assert_array_equal(ledger.n, [406, 406, 100, 17])
    np.testing.assert_array_equal(ledger.flag15_positive, [116, 116, 20, 0])
    assert np.all(np.diff(ledger.n) <= 0)
    assert ledger.iloc[-1].n == 17
    with pytest.raises(ValueError, match="Duplicated"):
        t.validation_ledger(pd.concat([units, units.iloc[:1]]))
    with pytest.raises(ValueError):
        t.validation_ledger(units, minimum_coverage=np.nan)


def test_synthetic_confusion_matrix_example():
    from snowmix.case_analysis import binary_agreement

    result = binary_agreement(
        [1, 1, 1, 1, 0, 0, 0, 0], [0.9, 0.7, 0.2, 0.05, 0.8, 0.4, 0.1, 0], 0.5
    )
    assert (result["tp"], result["fn"], result["fp"], result["tn"]) == (2, 2, 1, 3)
    assert result["recall"] == 0.5 and result["specificity"] == 0.75


def test_filter_plot_labels_follow_exercise_threshold():
    units = pd.read_csv(CASE / "data/demo/modis_1km_units.csv")
    fig = t.plot_validation_design(t.validation_ledger(units, 0.8))
    labels = [text.get_text() for text in fig.axes[0].get_xticklabels()]
    assert all("95%" not in value for value in labels)
    assert sum("80%" in value for value in labels) == 2
    t.plt.close(fig)
