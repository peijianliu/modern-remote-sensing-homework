"""采用 MGS 和 FCLS 求解积雪光谱混合模型。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.linalg import solve_triangular


def mgs_operator(matrix: np.ndarray) -> np.ndarray:
    """用两遍正交化的修正 Gram–Schmidt（MGS）计算最小二乘线性算子。

    matrix 的维度为波段数 × 非阴影端元数。拟合采用无约束系数，
    零反射率阴影的比例由 1 − 非阴影系数之和得到。
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Endmembers must be a finite bands x physical-components matrix")
    m, k = a.shape
    if k > m or np.linalg.matrix_rank(a) != k:
        raise ValueError("Endmember matrix is rank deficient")
    q = np.zeros((m, k))
    r = np.zeros((k, k))
    scale = np.linalg.norm(a, axis=0).max()
    for j in range(k):
        v = a[:, j].copy()
        for _ in range(2):
            for i in range(j):
                correction = q[:, i] @ v
                r[i, j] += correction
                v -= correction * q[:, i]
        r[j, j] = np.linalg.norm(v)
        if r[j, j] <= np.finfo(float).eps * max(a.shape) * scale:
            raise ValueError("Numerically singular endmembers")
        q[:, j] = v / r[j, j]
    return solve_triangular(r, q.T)


def fcls_batch(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """通过活跃面枚举求解少量端元的全约束最小二乘（FCLS）。

    最小化 ||y − Af||²，约束为 f >= 0、sum(f) <= 1；剩余比例为零反射率
    阴影。逐一比较可行单纯形各个面上的解，包括阴影比例为零的边界。
    最多支持 6 个非阴影端元。
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    a = np.asarray(matrix, dtype=float)
    k = a.shape[1]
    if k > 6:
        raise ValueError("Active-face solver is limited to six physical components")
    best = np.zeros((len(y), k))
    best_error = np.einsum("ij,ij->i", y, y)
    observation_norm = best_error.copy()
    tolerance = 1e-10
    for size in range(1, k + 1):
        for support in combinations(range(k), size):
            ab = a[:, support]
            if np.linalg.matrix_rank(ab) < size:
                continue
            operator = mgs_operator(ab)
            unconstrained = y @ operator.T
            gram = ab.T @ ab
            cross = y @ ab
            ill_conditioned = np.linalg.cond(ab) > 10000
            # 近共线光谱通过线性算子求解，避免 A.T @ A 使条件数平方放大。
            direction = (
                operator @ (operator.T @ np.ones(size))
                if ill_conditioned
                else np.linalg.solve(gram, np.ones(size))
            )
            direction /= direction.sum()
            boundary = unconstrained + (1 - unconstrained.sum(axis=1))[:, None] * direction
            for candidate in (unconstrained, boundary):
                valid = (candidate.min(axis=1) >= -tolerance) & (
                    candidate.sum(axis=1) <= 1 + tolerance
                )
                # 修正浮点舍入引起的微小负值及系数和越界。
                candidate = np.maximum(candidate, 0)
                sums = candidate.sum(axis=1)
                above = sums > 1
                candidate[above] /= sums[above, None]
                if ill_conditioned:
                    residual = y - candidate @ ab.T
                    error = np.einsum("ij,ij->i", residual, residual)
                else:
                    error = observation_norm - 2 * np.sum(candidate * cross, axis=1)
                    error += np.sum((candidate @ gram) * candidate, axis=1)
                    error = np.maximum(error, 0)  # 修正平方范数计算中的相消误差
                take = valid & (error < best_error)
                if take.any():
                    best[take] = 0
                    best[np.ix_(np.flatnonzero(take), support)] = candidate[take]
                    best_error[take] = error[take]
    return best


@dataclass
class UnmixResult:
    """逐像元解混结果。

    fractions 为非阴影端元的原始系数；shade = 1 − sum(fractions)。
    normalized 为除以 1 − shade 后的端元比例。modeled 使用原始系数重建，
    residual 为逐波段残差 E_c = 观测 − 重建，rms 为残差的波段均方根。
    radius_um 为通过质量阈值的候选半径，单位 μm；qa 为质量位掩膜。
    """

    fractions: np.ndarray
    shade: np.ndarray
    normalized: np.ndarray
    radius_um: np.ndarray
    radius_index: np.ndarray
    rms: np.ndarray
    residual: np.ndarray
    modeled: np.ndarray
    condition: np.ndarray
    qa: np.ndarray


def select_snow(
    y: np.ndarray,
    snow_spectra: np.ndarray,
    radii_um: np.ndarray,
    backgrounds: np.ndarray,
    *,
    solver: str = "original",
    minimum_denominator: float = 0.05,
    minimum_snow: float = 0.1,
    condition_warning: float = 10000,
) -> UnmixResult:
    """逐个候选雪光谱拟合，以最小光谱 RMS 选择像元的雪粒半径。

    y、snow_spectra、backgrounds 的维度依次为像元数 × 波段数、
    半径数 × 波段数、波段数 × 非雪端元数。残差由原始系数重建。
    粒径输出要求阴影归一化分母超过 minimum_denominator，且归一化雪比例
    达到 minimum_snow；默认阈值分别为 0.05 和 0.1。

    qa 按位组合：1 输入含非有限值；2 无满列秩模型；4 归一化分母未超过
    阈值；8 归一化雪比例低于阈值；16 半径位于候选范围边界；32 条件数
    超过 condition_warning；64 原始端元系数或阴影比例超出 [0, 1]。
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    snow = np.atleast_2d(np.asarray(snow_spectra, dtype=np.float64))
    radii = np.asarray(radii_um, dtype=float)
    background = np.asarray(backgrounds, dtype=float)
    if background.ndim == 1:
        background = background[:, None]
    if snow.shape != (len(radii), y.shape[1]) or background.shape[0] != y.shape[1]:
        raise ValueError("Wavelength/band dimensions do not agree")
    if solver not in {"original", "fcls"}:
        raise ValueError("solver must be original or fcls")
    finite = np.isfinite(y).all(axis=1)
    data = y[finite]
    k = background.shape[1] + 1
    best = np.full((len(data), k), np.nan)
    best_mse = np.full(len(data), np.inf)
    best_index = np.full(len(data), -1, dtype=int)
    best_condition = np.full(len(data), np.nan)
    for index, s in enumerate(snow):
        a = np.column_stack((s, background))
        try:
            operator = mgs_operator(a)
        except ValueError:
            continue
        f = data @ operator.T if solver == "original" else fcls_batch(data, a)
        error = data - f @ a.T
        mse = np.mean(error * error, axis=1)
        take = mse < best_mse
        best[take] = f[take]
        best_mse[take] = mse[take]
        best_index[take] = index
        best_condition[take] = np.linalg.cond(a)
    fractions = np.full((len(y), k), np.nan)
    fractions[finite] = best
    selected = np.full(len(y), -1, dtype=int)
    selected[finite] = best_index
    condition = np.full(len(y), np.nan)
    condition[finite] = best_condition
    successful = selected >= 0
    modeled = np.full_like(y, np.nan)
    modeled[successful] = (
        fractions[successful, :1] * snow[selected[successful]]
        + fractions[successful, 1:] @ background.T
    )
    residual = y - modeled
    rms = np.sqrt(np.mean(residual * residual, axis=1))
    shade = 1 - fractions.sum(axis=1)
    denominator = 1 - shade
    normalized = np.full_like(fractions, np.nan)
    normalize = successful & (denominator > minimum_denominator)
    normalized[normalize] = fractions[normalize] / denominator[normalize, None]
    radius = np.full(len(y), np.nan)
    grain_valid = normalize & (normalized[:, 0] >= minimum_snow - 1e-10)
    radius[grain_valid] = radii[selected[grain_valid]]
    qa = np.zeros(len(y), dtype=np.uint16)
    qa[~finite] |= 1
    qa[finite & ~successful] |= 2
    qa[successful & ~normalize] |= 4
    qa[normalize & ~grain_valid] |= 8
    if len(radii) > 1:
        qa[grain_valid & ((selected == 0) | (selected == len(radii) - 1))] |= 16
    qa[condition > condition_warning] |= 32
    overflow = (fractions.min(axis=1) < -1e-8) | (fractions.max(axis=1) > 1 + 1e-8)
    overflow |= (shade < -1e-8) | (shade > 1 + 1e-8)
    qa[successful & overflow] |= 64
    return UnmixResult(
        fractions, shade, normalized, radius, selected, rms, residual, modeled, condition, qa
    )
