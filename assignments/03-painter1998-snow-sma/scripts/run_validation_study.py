"""重算失配对照和公开端元的独立覆盖转移试验。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from snowmix import teaching as t
from snowmix.validation_study import independent_transfer, mismatch_study


def run(output: Path) -> None:
    """从公开教学输入重解混，写入新目录；保留已有实验。"""
    if output.exists():
        raise FileExistsError("Use a new output directory")
    with np.load(t.demo_directory() / "transfer_library.npz", allow_pickle=False) as archive:
        library = dict(archive)
    selected = library["mismatch_band_indices"]

    def inputs(angle: float) -> dict:
        value = t.synthetic_inputs(angle_deg=angle)
        for key in ("snow", "quartz", "wavelength_um"):
            value[key] = value[key][..., selected]
        return value

    repeated, summary = mismatch_study(inputs(0), inputs(60), library["unseen_background_aviris"])
    metrics, paired, _ = independent_transfer(t.demo_directory(), library)
    output.mkdir(parents=True)
    for name, table in (
        ("mismatch_repeats", repeated),
        ("mismatch_summary", summary),
        ("transfer_metrics", metrics),
        ("transfer_paired_deltas", paired),
    ):
        table.to_csv(output / (name + ".csv"), index=False)
    (output / "metrics.json").write_text(
        json.dumps({"mismatch_cases": len(summary), "transfer_rows": len(metrics)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "stage": "validation_studies",
        "status": "complete",
        "seed": 1998,
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
