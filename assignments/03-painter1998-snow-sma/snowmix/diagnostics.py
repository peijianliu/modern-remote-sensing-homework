"""区分光谱拟合、系数可行性和独立覆盖误差的纯 NumPy 诊断。"""

from __future__ import annotations

import numpy as np

from .case_analysis import paired_block_means


def coefficient_status(fractions: np.ndarray, shade: np.ndarray, tolerance: float = 1e-8) -> dict:
    """逐像元标记原始系数和阴影是否满足面积比例的范围。

    Args:
        fractions: (pixels, components)，不含阴影的原始系数，首列为雪。
        shade: (pixels,)，零反射率阴影补量。这里只判定范围，不改写系数。
        tolerance: 允许的浮点误差；低于 -tolerance 才计为负值。

    Returns:
        一维布尔掩膜，以及有明确含义的状态码：0 无解、1 可行、2 雪为负、
        3 雪超过 1、4 其他端元或阴影越界。状态有优先顺序，计数不重复。

    Raises:
        ValueError: 输入形状或容差不合法。
    """
    f, shadow = np.asarray(fractions, float), np.asarray(shade, float)
    if f.ndim != 2 or not f.shape[1] or shadow.shape != (len(f),):
        raise ValueError("Expected fractions (pixels, components) and shade (pixels,)")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    finite = np.isfinite(f).all(1) & np.isfinite(shadow)
    values = np.column_stack((f, shadow))
    feasible = finite & (values >= -tolerance).all(1) & (values <= 1 + tolerance).all(1)
    feasible &= np.isclose(values.sum(1), 1, atol=tolerance, rtol=0)
    negative = finite & (f[:, 0] < -tolerance)
    above = finite & (f[:, 0] > 1 + tolerance)
    status = np.zeros(len(f), np.uint8)
    status[finite] = 4
    status[feasible] = 1
    status[negative] = 2
    status[above] = 3
    return {
        "finite": finite,
        "feasible": feasible,
        "negative_snow": negative,
        "over_one_snow": above,
        "status": status,
    }


def paired_rmse_difference(
    reference: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    row_column: np.ndarray,
    *,
    repeats: int = 1000,
    seed: int = 1998,
) -> dict:
    """用相同的 1.5 km 空间区块抽样计算第二方法减第一方法的 RMSE 差。

    输入比例均无量纲；row_column 为原始 30 m 网格的行列号。先筛共同有限
    像元，再对每个重采样的均方误差开方，不能对像元绝对误差直接取平均。
    少于 5 个区块时只给点估计，区间为 None。返回比例单位，展示时乘 100。
    """
    truth, a, b = (np.asarray(v, float) for v in (reference, first, second))
    positions = np.asarray(row_column)
    if truth.ndim != 1 or a.shape != truth.shape or b.shape != truth.shape:
        raise ValueError("Reference and estimates must be aligned one-dimensional arrays")
    if positions.shape != (len(truth), 2):
        raise ValueError("Expected one row/column position per observation")
    valid = np.isfinite(truth) & np.isfinite(a) & np.isfinite(b)
    errors = np.column_stack(((a[valid] - truth[valid]) ** 2, (b[valid] - truth[valid]) ** 2))
    blocks = len(np.unique(positions[valid] // 50, axis=0))
    value = {
        "n": int(valid.sum()),
        "spatial_blocks": blocks,
        "repeats": repeats,
        "seed": seed,
        "rmse_first": None,
        "rmse_second": None,
        "difference": None,
        "ci_low": None,
        "ci_high": None,
    }
    if not len(errors):
        return value
    rmse = np.sqrt(errors.mean(0))
    value.update(
        rmse_first=float(rmse[0]), rmse_second=float(rmse[1]), difference=float(rmse[1] - rmse[0])
    )
    if blocks >= 5:
        bootstrap = paired_block_means(errors, positions[valid], repeats=repeats, seed=seed)
        resampled = np.sqrt(bootstrap["bootstrap_means"])
        low, high = np.quantile(resampled[:, 1] - resampled[:, 0], [0.025, 0.975])
        value.update(ci_low=float(low), ci_high=float(high))
    return value
