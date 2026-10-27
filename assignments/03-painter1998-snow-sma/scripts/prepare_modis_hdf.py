"""从公开MOD09GA HDF-EOS导出字段，并核对原生产品标识与输入哈希。

HDF4读取使用已有MATLAB入口；默认教学Notebook不调用本脚本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from scipy.io import loadmat

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from scripts.run_supplement import resolve_matlab


def prepare(input_path: Path, output_path: Path, matlab: str | None = None) -> dict:
    """只读取指定公开HDF，写入新缓存；已有结果不覆盖。"""
    if output_path.exists():
        raise FileExistsError("Choose a new MODIS extraction output")
    with input_path.open("rb") as stream:
        if stream.read(4) != b"\x0e\x03\x13\x01":
            raise ValueError(
                "Input is not an HDF4 file; an Earthdata login page may have been saved"
            )
        stream.seek(0)
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    executable = resolve_matlab(matlab)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def quote(path: Path) -> str:
        return "'" + path.resolve().as_posix().replace("'", "''") + "'"

    command = f"addpath({quote(CASE / 'matlab')}); export_modis({quote(input_path)},{quote(output_path)});"
    subprocess.run([str(executable), "-batch", command], cwd=CASE, check=True)
    data = loadmat(output_path, variable_names=["core_metadata", "band_raw", "qc", "state"])
    text = str(data["core_metadata"].item())
    match = re.search(r'LOCALGRANULEID.*?VALUE\s*=\s*"([^"]+)"', text, re.S)
    if match is None or match[1] != input_path.name:
        raise ValueError("Native MODIS granule identifier does not match the requested product")
    if data["band_raw"].shape != (2400, 2400, 7) or data["state"].shape != (1200, 1200):
        raise ValueError("Unexpected MOD09GA dimensions")
    value = {
        "product": match[1],
        "input_sha256": checksum,
        "field_order": "YDim, XDim, band",
        "native_id_verified": True,
        "source": f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MOD09GA.061/{input_path.stem}/{input_path.name}",
    }
    output_path.with_suffix(".source.json").write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matlab")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input.resolve(), args.output.resolve(), args.matlab), indent=2))
