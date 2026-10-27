"""教学案例的吸收特征、固定粒径扫描和产品一致性统计。

本模块仅依赖 NumPy 与共享解混器；完整影像读取和投影由数据准备脚本完成。
反射率、系数为无量纲，波长及半径分别以 μm 表示。
"""

from __future__ import annotations

import numpy as np

from .linear import select_snow


def paired_block_means(
    values: np.ndarray,
    row_col: np.ndarray,
    block_size_pixels: int = 50,
    repeats: int = 1000,
    seed: int = 1998,
) -> dict:
    """对相同空间区块内的多列指标配对重采样，计算像元加权均值区间。

    Args:
        values: 有限值矩阵，形状 (pixels, metrics)；各列必须使用相同像元。
        row_col: 对应的零起算 ROI 行列号，形状 (pixels, 2)，不得重复。
        block_size_pixels: 正方形区块边长；30 m 网格的 50 像元对应 1.5 km。
        repeats: bootstrap 次数，至少 2 次。
        seed: 固定随机种子。区块从 ROI 左上角行列号 (0, 0) 划分。

    Returns:
        mean、interval_95、bootstrap_means，以及样本数和区块数。
        每次对所有指标抽取相同区块；差值须先逐次相减再求分位。
        重采样后以区块像元数加权，避免边缘小区块获得过大权重。

    Raises:
        ValueError: 输入未对齐、含无效值或重复位置，参数无效，或不足 5 个区块。
    """
    data = np.asarray(values, dtype=np.float64)
    positions = np.asarray(row_col, dtype=np.float64)
    if (
        data.ndim != 2
        or not data.shape[0]
        or not data.shape[1]
        or not np.isfinite(data).all()
        or positions.shape != (len(data), 2)
        or not np.isfinite(positions).all()
        or np.any(positions < 0)
        or np.any(positions != np.floor(positions))
        or len(np.unique(positions, axis=0)) != len(data)
    ):
        raise ValueError("Use aligned finite metrics and unique nonnegative row/column positions")
    if (
        isinstance(block_size_pixels, bool)
        or not isinstance(block_size_pixels, (int, np.integer))
        or block_size_pixels <= 0
        or isinstance(repeats, bool)
        or not isinstance(repeats, (int, np.integer))
        or repeats < 2
    ):
        raise ValueError("Block size must be a positive integer and repeats must be at least 2")
    blocks, inverse = np.unique(
        positions.astype(np.int64) // block_size_pixels, axis=0, return_inverse=True
    )
    if len(blocks) < 5:
        raise ValueError("At least 5 spatial blocks are required for interval estimation")
    count = np.bincount(inverse)
    totals = np.column_stack(
        [np.bincount(inverse, weights=data[:, column]) for column in range(data.shape[1])]
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(repeats, len(blocks)))
    # 不展开全部像元：区块计数与各列和是均值所需的充分统计量。
    means = totals[draws].sum(axis=1) / count[draws].sum(axis=1, keepdims=True)
    return {
        "n": len(data),
        "spatial_blocks": len(blocks),
        "block_size_pixels": int(block_size_pixels),
        "repeats": int(repeats),
        "seed": int(seed),
        "mean": data.mean(axis=0),
        "interval_95": np.quantile(means, [0.025, 0.975], axis=0),
        "bootstrap_means": means,
    }


def absorption_area(
    wavelength_um: np.ndarray, spectra: np.ndarray, lower_um: float = 0.95, upper_um: float = 1.09
) -> np.ndarray:
    """计算端点线性连续统去除后的吸收面积，单位 μm。

    Args:
        wavelength_um: 波长坐标，必须覆盖整个积分区间；仪器分段波长在内部排序。
        spectra: 最后一维为波长的一个或多个反射率光谱。
        lower_um: 积分下限；端点反射率采用相邻波段的线性插值。
        upper_um: 积分上限。

    Returns:
        每条光谱的梯形积分 ∫(1−R/C)dλ，形状为 spectra.shape[:-1]。
        含无效值或非正连续统的光谱输出 NaN。积分结果保留符号。

    Raises:
        ValueError: 波长、光谱维度或积分区间不符合要求。
    """
    wavelength = np.asarray(wavelength_um, dtype=float)
    values = np.asarray(spectra, dtype=float)
    if (
        wavelength.ndim != 1
        or len(wavelength) < 2
        or not np.isfinite(wavelength).all()
        or len(np.unique(wavelength)) != len(wavelength)
    ):
        raise ValueError("Wavelength coordinates must be finite and unique")
    if values.ndim == 0 or values.shape[-1] != len(wavelength):
        raise ValueError("Spectra and wavelength dimensions differ")
    order = np.argsort(wavelength, kind="stable")
    wavelength = wavelength[order]
    if not wavelength[0] <= lower_um < upper_um <= wavelength[-1]:
        raise ValueError("Absorption interval is outside the wavelength coordinates")
    rows = values.reshape(-1, len(wavelength))[:, order]

    def endpoint(coordinate: float) -> np.ndarray:
        index = min(max(int(np.searchsorted(wavelength, coordinate)), 1), len(wavelength) - 1)
        fraction = (coordinate - wavelength[index - 1]) / (
            wavelength[index] - wavelength[index - 1]
        )
        return rows[:, index - 1] * (1 - fraction) + rows[:, index] * fraction

    inside = (wavelength > lower_um) & (wavelength < upper_um)
    x = np.r_[lower_um, wavelength[inside], upper_um]
    lower, upper = endpoint(lower_um), endpoint(upper_um)
    reflectance = np.column_stack((lower, rows[:, inside], upper))
    position = (x - lower_um) / (upper_um - lower_um)
    continuum = lower[:, None] + (upper - lower)[:, None] * position
    valid = np.isfinite(reflectance).all(1) & (continuum > 0).all(1)
    areas = np.full(len(rows), np.nan)
    areas[valid] = np.trapz(1 - reflectance[valid] / continuum[valid], x, axis=1)
    return areas.reshape(values.shape[:-1])


def select_quantile_indices(
    values: np.ndarray, flat_indices: np.ndarray, quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)
) -> np.ndarray:
    """从分位目标附近选择不同像元；距离并列时按全图行优先序号选择。

    values 与 flat_indices 一一对应。非有限特征剔除；选中一个像元后，后续
    分位点不再使用它。返回原始图像的扁平序号，保持 quantiles 的顺序。
    """
    feature = np.asarray(values, dtype=float).ravel()
    indices = np.asarray(flat_indices, dtype=np.int64).ravel()
    q = np.asarray(quantiles, dtype=float)
    if feature.shape != indices.shape or len(np.unique(indices)) != len(indices):
        raise ValueError("Features require one unique pixel index per value")
    if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Quantiles must lie in [0, 1]")
    finite = np.isfinite(feature)
    feature, indices = feature[finite], indices[finite]
    if len(feature) < len(q):
        raise ValueError("Insufficient finite candidates for distinct endmembers")
    targets = np.quantile(feature, q)
    selected = []
    for target in targets:
        order = np.lexsort((indices, np.abs(feature - target)))
        selected.append(next(int(indices[i]) for i in order if int(indices[i]) not in selected))
    return np.asarray(selected, dtype=np.int64)


def fixed_radius_scan(
    observed: np.ndarray,
    snow_spectra: np.ndarray,
    radii_um: np.ndarray,
    backgrounds: np.ndarray,
    requested_radii: tuple = (50, 100, 200, 500, 1000),
    *,
    include_spectra: bool = True,
    solver: str = "original",
) -> dict:
    """对同一观测、背景和求解器逐个采用固定雪粒半径。

    返回数组的第一维为 requested_radii，第二维为像元。fractions 是原始
    非阴影系数，normalized 为阴影归一化比例；residual 和 modeled 默认保留，
    全区运行可将 include_spectra=False 以节约内存。
    """
    radii = np.asarray(radii_um, dtype=float).ravel()
    snow = np.asarray(snow_spectra, dtype=float)
    requested = np.asarray(requested_radii, dtype=float)
    if (
        requested.ndim != 1
        or not len(requested)
        or len(np.unique(requested)) != len(requested)
        or not np.isin(requested, radii).all()
    ):
        raise ValueError("Every requested radius must occur once in the source library")
    fields = ["fractions", "normalized", "shade", "rms", "condition", "qa"]
    if include_spectra:
        fields += ["residual", "modeled"]
    results = {name: [] for name in fields}
    for radius in requested:
        index = np.flatnonzero(radii == radius)
        if len(index) != 1:
            raise ValueError("Source radius coordinates must be unique")
        result = select_snow(observed, snow[index], [radius], backgrounds, solver=solver)
        for name in fields:
            results[name].append(getattr(result, name))
    return {"radii_um": requested, **{name: np.stack(results[name]) for name in fields}}


def binary_agreement(
    reference_flag: np.ndarray, estimated_fraction: np.ndarray, threshold: float = 0.1
) -> dict:
    """以产品标志为参照计算二分类一致性；分母为零的指标输出 None。

    输入须为同一支持单元上的一维值；调用方先按空间位置和有效覆盖率筛选。
    非有限输入剔除。返回计数、参照召回率、参照特异度及总体一致率。
    """
    reference = np.asarray(reference_flag, dtype=float).ravel()
    estimated = np.asarray(estimated_fraction, dtype=float).ravel()
    if reference.shape != estimated.shape or not 0 <= threshold <= 1:
        raise ValueError("Inputs must align and threshold must lie in [0, 1]")
    finite_reference = reference[np.isfinite(reference)]
    if not np.isin(finite_reference, [0, 1]).all():
        raise ValueError("Reference flags must be binary")
    finite = np.isfinite(reference) & np.isfinite(estimated)
    flag = reference[finite].astype(bool)
    prediction = estimated[finite] >= threshold
    tp = int(np.sum(flag & prediction))
    fn = int(np.sum(flag & ~prediction))
    fp = int(np.sum(~flag & prediction))
    tn = int(np.sum(~flag & ~prediction))
    n = int(finite.sum())
    return {
        "n": n,
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "reference_positive": tp + fn,
        "reference_negative": fp + tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / n if n else None,
    }


def select_scene_endmembers(
    cube: np.ndarray, valid: np.ndarray, wavelength_um: np.ndarray, bands: np.ndarray
) -> dict:
    """按固定指数和亮度规则选择三条背景及五条雪谱，不使用拟合或参考标签。

    Args:
        cube: (rows, columns, native_bands) 的无量纲反射率。
        valid: 同一网格的布尔输入掩膜。
        wavelength_um: 每个原始波段的中心波长，μm。
        bands: 用于解混的有效波段布尔掩膜。

    Returns:
        扁平行优先背景/雪谱索引、雪候选掩膜、背景矩阵(bands,3)、
        雪谱(5,bands)、吸收面积，以及NDSI/NDVI网格。

    Raises:
        ValueError: 维度或预设选样条件不足。
    """
    cube, valid = np.asarray(cube), np.asarray(valid)
    wave, bands = np.asarray(wavelength_um), np.asarray(bands)
    if cube.ndim != 3 or valid.shape != cube.shape[:2] or valid.dtype != np.bool_:
        raise ValueError("A reflectance cube and aligned boolean mask are required")
    if wave.shape != (cube.shape[2],) or bands.shape != wave.shape or bands.dtype != np.bool_:
        raise ValueError("Wavelength and band coordinates do not match the cube")

    def band(target: float) -> np.ndarray:
        return cube[..., np.argmin(abs(wave - target))]

    green, nir, red, swir = (band(v) for v in (0.55, 0.86, 0.65, 1.64))
    ndsi = (green - swir) / (green + swir + 1e-10)
    ndvi = (nir - red) / (nir + red + 1e-10)
    vegetation = np.flatnonzero(valid & (ndvi > 0.5) & (ndsi < 0))
    bare = np.flatnonzero(valid & (abs(ndvi) < 0.1) & (ndsi < 0))
    if not len(vegetation) or len(bare) < 2:
        raise ValueError("Prespecified background selection has insufficient candidates")
    selected = [int(vegetation[np.lexsort((vegetation, -ndvi.ravel()[vegetation]))[0]])]
    for target in np.quantile(green.ravel()[bare], [0.9, 0.1]):
        order = np.lexsort((bare, abs(green.ravel()[bare] - target)))
        selected.append(next(int(bare[i]) for i in order if int(bare[i]) not in selected))
    candidate = (
        valid & (ndsi > 0.7) & (abs(ndvi) < 0.15) & (green >= 0.65) & (green <= 1.1) & (nir > 0.11)
    )
    possible = np.flatnonzero(candidate)
    observed = cube[..., bands].reshape(-1, int(bands.sum()))
    areas = absorption_area(wave[bands], observed[possible])
    snow_indices = select_quantile_indices(areas, possible)
    image_snow = observed[snow_indices].astype(float)
    return {
        "background_indices": np.array(selected),
        "backgrounds": observed[selected].T.astype(float),
        "snow_indices": snow_indices,
        "snow_spectra": image_snow,
        "candidate": candidate,
        "absorption_area_um": absorption_area(wave[bands], image_snow),
        "ndsi": ndsi,
        "ndvi": ndvi,
    }
