"""读取光谱库，计算传感器光谱响应的积分权重。"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.special import ndtr


def read_ecostress(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取 ECOSTRESS 文本光谱，将百分数反射率转换为无量纲数值。"""
    if path.name.lower() in {"granite_h2.txt", "cryumbrept_87p3855.txt"}:
        raise PermissionError(
            "This historical raw spectrum is excluded by PUBLIC_DATA_POLICY.md; current computations use public USGS or AVIRIS spectra"
        )
    metadata, rows = {}, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        else:
            values = line.split()
            if len(values) == 2:
                try:
                    rows.append(tuple(map(float, values)))
                except ValueError:
                    pass
    if "percent" not in metadata.get("Y Units", "").lower():
        raise ValueError("Expected documented percent reflectance")
    array = np.array(rows)
    order = np.argsort(array[:, 0])
    array = array[order]
    wavelengths, unique = np.unique(array[:, 0], return_index=True)
    return wavelengths, array[unique, 1] / 100, metadata


def load_terrestrial_library(target_um: np.ndarray, directory: Path):
    """将植被、岩石、土壤和石英代理光谱线性重采样到目标波长。"""
    entries = [
        ("vegetation", "lodgepole_pine.asc", read_usgs_ascii),
        ("rock", "granite_h2.txt", read_ecostress),
        ("soil", "cryumbrept_87p3855.txt", read_ecostress),
        ("quartz_proxy", "quartz_hs32.5854.asc", read_usgs_ascii),
    ]
    spectra, metadata = [], []
    for name, filename, reader in entries:
        wavelengths, reflectance, record = reader(directory / filename)
        spectra.append(linear_resample(wavelengths, reflectance, target_um))
        metadata.append({"class": name, "file": filename, "metadata": record})
    return np.column_stack(spectra), metadata


def modis_responses(path: Path):
    """先按积分归一化各通道相对光谱响应（RSR），再求各产品波段的通道平均。"""
    import pandas as pd

    wavelengths, responses = [], []
    for band in range(1, 8):
        table = pd.read_excel(path, sheet_name=f"band_{band}")
        target = np.unique(table["Wavelength"].to_numpy(dtype=float))
        channel_responses = []
        for _, channel in table.groupby("Channel"):
            channel = channel.sort_values("Wavelength")
            w = channel["Wavelength"].to_numpy(dtype=float)
            r = channel["RSR"].to_numpy(dtype=float)
            r = r / np.trapz(r, w)
            channel_responses.append(np.interp(target, w, r, left=0, right=0))
        wavelengths.append(target / 1000)
        responses.append(np.mean(channel_responses, axis=0))
    return wavelengths, responses


def read_usgs_ascii(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取 USGS splib05 定宽文本，返回 μm 波长、无量纲反射率和来源信息。"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for line in lines[16:]:
        # splib05 ASCII 每列固定 15 个字符；缺测值可能紧邻波长，需按列宽解析。
        fields = [line[i : i + 15].strip() for i in (0, 15, 30)]
        try:
            wavelength, reflectance = float(fields[0]), float(fields[1])
        except (ValueError, IndexError):
            continue
        if reflectance > -100 and wavelength > 0:
            rows.append((wavelength, reflectance))
    data = np.asarray(rows)
    if len(data) < 10:
        raise ValueError(f"No valid USGS reflectance data in {path}")
    order = np.argsort(data[:, 0])
    data = data[order]
    wavelength, unique = np.unique(data[:, 0], return_index=True)
    return (
        wavelength,
        data[unique, 1],
        {
            "title": lines[14].strip(),
            "history": lines[15].strip(),
            "wavelength_unit": "um",
            "reflectance_unit": "1",
        },
    )


def wavelength_to_um(values: np.ndarray, unit: str) -> np.ndarray:
    """将已声明单位的波长统一转换为 μm。"""
    normalized = unit.lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"nm", "nanometers", "nanometres"}:
        return np.asarray(values, dtype=float) / 1000
    if normalized in {"um", "micrometers", "micrometres", "microns"}:
        return np.asarray(values, dtype=float)
    raise ValueError(f"Unrecognized wavelength unit: {unit}")


def linear_resample(
    source_wavelength: np.ndarray, spectra: np.ndarray, destination_wavelength: np.ndarray
) -> np.ndarray:
    """在源光谱波长范围内逐谱线性插值，保留最后一维以外的数组形状。"""
    wavelength = np.asarray(source_wavelength)
    target = np.asarray(destination_wavelength)
    if np.any(target < wavelength[0]) or np.any(target > wavelength[-1]):
        raise ValueError("Spectral extrapolation is prohibited")
    flat = np.asarray(spectra).reshape(-1, len(wavelength))
    output = np.stack([np.interp(target, wavelength, row) for row in flat])
    return output.reshape(np.asarray(spectra).shape[:-1] + (len(target),))


def gaussian_band_matrix(
    wavelength_um: np.ndarray, centers_um: np.ndarray, fwhm_um: np.ndarray
) -> np.ndarray:
    """按给定中心波长和半高全宽（FWHM）积分高斯光谱响应。

    在波长单元内积分，可在 10 nm 正演网格上处理窄波段。
    返回源波长数 × 传感器波段数的权重矩阵，每列之和为 1。
    """
    wavelength = np.asarray(wavelength_um)
    centers = np.asarray(centers_um)
    fwhm = np.broadcast_to(np.asarray(fwhm_um), centers.shape)
    if (fwhm <= 0).any() or np.any(np.diff(wavelength) <= 0):
        raise ValueError("Positive FWHM and increasing wavelengths are required")
    edges = np.r_[
        wavelength[0] - (wavelength[1] - wavelength[0]) / 2,
        (wavelength[:-1] + wavelength[1:]) / 2,
        wavelength[-1] + (wavelength[-1] - wavelength[-2]) / 2,
    ]
    z = (edges[:, None] - centers[None, :]) / (fwhm[None, :] / 2.354820045)
    weights = np.diff(ndtr(z), axis=0)
    total = weights.sum(axis=0)
    if (total < 0.99).any():
        raise ValueError("Band response is insufficiently covered by model grid")
    return weights / total


def response_matrix(
    wavelength_um: np.ndarray,
    response_wavelengths: list[np.ndarray],
    responses: list[np.ndarray],
    valid: np.ndarray | None = None,
    minimum_coverage: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """在有效输入波长单元内积分表列相对光谱响应（RSR）。

    端元与观测光谱采用同一响应权重进行反射率卷积。返回权重矩阵与各波段
    的有效响应覆盖率；覆盖率低于 minimum_coverage 的权重列设为 NaN。
    """
    wavelength = np.asarray(wavelength_um, dtype=float)
    order = np.argsort(wavelength)
    sorted_wavelength = wavelength[order]
    if np.any(np.diff(sorted_wavelength) <= 0):
        raise ValueError("Duplicate sensor centers require an explicit detector/band mask")
    edges = np.r_[
        sorted_wavelength[0] - np.diff(sorted_wavelength)[0] / 2,
        (sorted_wavelength[:-1] + sorted_wavelength[1:]) / 2,
        sorted_wavelength[-1] + np.diff(sorted_wavelength)[-1] / 2,
    ]
    keep = np.ones(len(wavelength), dtype=bool) if valid is None else np.asarray(valid)[order]
    columns, coverage = [], []
    for rw, rs in zip(response_wavelengths, responses):
        rw = np.asarray(rw)
        rs = np.maximum(np.asarray(rs), 0)
        cumulative = np.r_[0, np.cumsum(np.diff(rw) * (rs[:-1] + rs[1:]) / 2)]
        raw = np.diff(np.interp(edges, rw, cumulative, left=0, right=cumulative[-1]))
        fraction = raw[keep].sum() / cumulative[-1]
        coverage.append(fraction)
        raw[~keep] = 0
        column = np.zeros(len(wavelength))
        if fraction >= minimum_coverage:
            column[order] = raw / raw.sum()
        else:
            column[:] = np.nan
        columns.append(column)
    return np.column_stack(columns), np.asarray(coverage)
