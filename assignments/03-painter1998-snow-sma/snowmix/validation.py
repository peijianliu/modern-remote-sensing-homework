"""读取实验室、野外及参考验证数据，计算半径换算与分块统计量。"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def interval_mask(wavelength: np.ndarray, intervals: list[list[float]]) -> np.ndarray:
    """返回给定波长闭区间并集的布尔掩膜。"""
    wavelength = np.asarray(wavelength, dtype=float)
    mask = np.zeros(wavelength.shape, dtype=bool)
    for lower, upper in intervals:
        if lower >= upper:
            raise ValueError("Each interval must have lower < upper")
        mask |= (wavelength >= lower) & (wavelength <= upper)
    return mask


def ssa_radius_um(ssa_m2_kg: np.ndarray, ice_density: float = 917) -> np.ndarray:
    """将雪比表面积（SSA，m²/kg）换算为球形表面积—体积等效半径（μm）。

    换算式为 r = 3 / (ice_density × SSA)，ice_density 的单位为 kg/m³，
    结果由 m 转换为 μm。
    """
    ssa = np.asarray(ssa_m2_kg, dtype=float)
    if ice_density <= 0 or not np.isfinite(ssa).all() or (ssa <= 0).any():
        raise ValueError("SSA and ice density must be positive and finite")
    return 3e6 / (ice_density * ssa)


def read_lab_spectra(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """按文件声明的表头行数读取实验室光谱，保留大于 1 的原始反射率。"""
    text = path.read_text(encoding="cp1252")
    header = re.search(r"number of header rows\s*=\s*(\d+)", text.splitlines()[0])
    if header is None:
        raise ValueError("Missing documented laboratory header length")
    table = pd.read_csv(io.StringIO(text), skiprows=int(header.group(1)) - 1)
    spectral_columns = [
        name for name in table.columns if name.isdigit() and 350 <= int(name) <= 2500
    ]
    wavelengths = np.asarray(spectral_columns, dtype=float) / 1000
    if len(wavelengths) != 2151 or not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Unexpected laboratory wavelength schema")
    return (
        table.drop(columns=spectral_columns),
        wavelengths,
        table[spectral_columns].to_numpy(float),
    )


def parse_ssa_text(text: str) -> tuple[str, pd.DataFrame, str]:
    """读取 SSA 采样表，支持以逗号作为小数点的数据记录。"""
    condition = re.search(r"Snow condition/type identifier:\s*([^\r\n]+)", text)
    if condition is None:
        raise ValueError("Missing SSA snow-condition identifier")
    calibration = re.search(r"Overall calibration quality:\s*([^\r\n]+)", text)
    start = text.find("Sample number\tSample signal\tReflectance\tSpecific surface area")
    if start < 0:
        raise ValueError("Missing SSA acquisition table")
    rows = []
    for line in text[start:].splitlines()[1:]:
        fields = line.strip().split("\t")
        if len(fields) >= 5 and fields[0].isdigit():
            rows.append([float(value.replace(",", ".")) for value in fields[:5]])
    if not rows:
        raise ValueError("Empty SSA acquisition table")
    table = pd.DataFrame(
        rows, columns=["sample", "signal_mV", "reflectance_percent", "ssa_m2_kg", "height_cm"]
    )
    return (
        condition.group(1).strip(),
        table,
        calibration.group(1).strip() if calibration else "unreported",
    )


def read_ssa_archive(path: Path, ice_density: float = 917) -> pd.DataFrame:
    """以各雪况最高采样位置的 SSA 记录作为该雪况的表层参考。"""
    records = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".txt"):
                continue
            condition, profile, calibration = parse_ssa_text(archive.read(name).decode("cp1252"))
            surface = profile.iloc[int(np.argmax(profile.height_cm.to_numpy()))]
            records.append(
                {
                    "name": condition,
                    "surface_ssa_m2_kg": surface.ssa_m2_kg,
                    "surface_reference_radius_um": float(
                        ssa_radius_um(surface.ssa_m2_kg, ice_density)
                    ),
                    "ssa_surface_height_cm": surface.height_cm,
                    "ssa_profile_samples": len(profile),
                    "ssa_calibration": calibration,
                    "ssa_reference_scope": "condition-level uppermost SSA sample",
                }
            )
    return pd.DataFrame(records)


def read_field_archive(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """按数据集目标编号关联野外光谱和站点信息，将元数据中的 −99 视为缺测。"""
    with zipfile.ZipFile(path) as archive:
        spectral_name = next(name for name in archive.namelist() if name.endswith("_average.csv"))
        site_name = next(
            name
            for name in archive.namelist()
            if name.endswith("_measurement_site_characteristics.csv")
        )
        spectral = pd.read_csv(io.BytesIO(archive.read(spectral_name)))
        sites = pd.read_csv(io.BytesIO(archive.read(site_name)), na_values=[-99])
    sites = sites.set_index("ID").loc[spectral.columns[1:]].reset_index(names="ID")
    wavelengths = spectral.iloc[:, 0].to_numpy(float) / 1000
    return sites, wavelengths, spectral.iloc[:, 1:].to_numpy(float).T


def block_bootstrap_delta(
    blocks: np.ndarray, delta: np.ndarray, seed: int, repeats: int = 1000
) -> dict:
    """对配对差值的空间块均值进行 bootstrap，按块处理相邻像元的依赖。

    各块在重采样中等权；返回块均值差和重采样分布的 95% 分位区间。
    有效块少于 5 个时，区间输出为空。
    """
    valid = np.isfinite(delta)
    labels = np.unique(np.asarray(blocks)[valid])
    means = np.asarray([np.mean(np.asarray(delta)[valid & (blocks == label)]) for label in labels])
    if len(means) < 5:
        return {
            "blocks": len(means),
            "mean_block_delta": float(means.mean()) if len(means) else None,
            "interval_95": None,
        }
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), (repeats, len(means)))].mean(axis=1)
    return {
        "blocks": len(means),
        "mean_block_delta": float(means.mean()),
        "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }
