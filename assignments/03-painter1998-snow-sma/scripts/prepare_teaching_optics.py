"""从官方 202105 光学库提取教学参数；不进行 Mie 属性计算或联网下载。

安装 full 依赖并准备官方 NetCDF 库后，显式运行 --prepare-optics。
默认教学直接读取已提供的散射参数包，无需执行本脚本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "7464158716a77a5933049b6c24865f61183b18fa38078a0033a23f7cbdfd49bd"


def prepare(optics_root: Path, output: Path) -> dict:
    """提取 148 个半径的两个光学选项，并记录每个上游文件的哈希。

    Args:
        optics_root: 包含 ice_Wrn84 和 ice_Wrn08 的官方解压目录。
        output: scattering_inputs.npz 的保存路径；已存在的不同内容保留。

    Returns:
        来源、精度转换和每个 NetCDF 文件的 SHA-256 清单。

    Raises:
        ValueError: 波长、物理量或属性维度与冻结版本不符。
        FileExistsError: 输出位置已有不同内容。
    """
    from netCDF4 import Dataset

    radii = np.arange(30, 1501, 10)
    wavelength = 0.205 + np.arange(480) * 0.01
    omega = np.empty((148, 480, 2), dtype=np.float64)
    asymmetry = np.empty((148, 480, 2), dtype=np.float32)
    sources = []
    for oi, stub in enumerate(("ice_Wrn84", "ice_Wrn08")):
        for ri, radius in enumerate(radii):
            relative = f"{stub}/{stub}_{radius:04d}.nc"
            path = optics_root / relative
            payload = path.read_bytes()
            # 内存方式兼容 Windows 上 netCDF C 库无法打开中文路径的情况。
            with Dataset(path.name, memory=payload) as dataset:
                if dataset.variables["wvl"].units not in {"m", "meter", "meters"}:
                    raise ValueError("Expected source wavelength unit: meter")
                np.testing.assert_allclose(dataset["wvl"][:] * 1e6, wavelength, atol=1e-12)
                omega[ri, :, oi] = dataset["ss_alb"][:]
                asymmetry[ri, :, oi] = dataset["asm_prm"][:]
            sources.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest()})
    if not np.isfinite(omega).all() or not np.isfinite(asymmetry).all():
        raise ValueError("Nonfinite source scattering parameters")
    from io import BytesIO

    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        ss_alb=omega,
        asm_prm=asymmetry,
        radii_um=radii,
        wavelength_um=wavelength,
        ice_options=np.array([1, 2]),
    )
    payload = buffer.getvalue()
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError("Different optics package exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest = {
        "title": "Teaching subset of SNICAR-ADv3 202105 ice optical properties",
        "authors": "Mark G. Flanner and SNICAR-ADv3 contributors",
        "version": "202105",
        "model_url": "https://github.com/mflanner/SNICARv3",
        "paper_doi": "10.5194/gmd-14-7673-2021",
        "archive_url": "http://snow.engin.umich.edu/opticalprops/snicar_v3_optics_202105/snicar_v3_optics.tar.gz",
        "archive_sha256": ARCHIVE_SHA256,
        "access": "Public accompanying model input library; research and education use described in the model paper",
        "license_note": "External optical input data attributed to SNICAR contributors; project MIT license covers original code, not a new license for this upstream library",
        "modifications": "Extracted ss_alb (float64) and asm_prm (float32) for two ice options and 30:10:1500 um; wavelength meter to um; no Mie recomputation",
        "array_axes": ["radius", "wavelength", "ice_option"],
        "shape": list(omega.shape),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "files": sources,
    }
    output.with_suffix(".source.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-optics", action="store_true", required=True)
    parser.add_argument("--optics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CASE / "data/demo/scattering_inputs.npz")
    args = parser.parse_args()
    manifest = prepare(args.optics_root, args.output)
    print(f"Prepared 148 × 480 × 2 scattering parameters: {manifest['output_sha256']}")


if __name__ == "__main__":
    main()
