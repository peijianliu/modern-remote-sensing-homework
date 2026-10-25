"""纯净球形、半无限雪层的 WW80 直接入射 delta-Eddington 解。"""

from __future__ import annotations

import numpy as np


def ww80_semi_infinite(
    omega: np.ndarray | float, asymmetry: np.ndarray | float, coszen: np.ndarray | float
) -> np.ndarray:
    """由单次散射属性计算方向—半球光谱反照率。

    Args:
        omega: 单次散射反照率，有限无量纲值，范围 [0, 1]。
        asymmetry: 散射不对称因子 g，范围 [0, 1)。
        coszen: 太阳天顶角余弦 mu0，范围 [0, 1]，0 表示掠射极限。

    Returns:
        float64 反照率，形状为三个输入广播后的形状。半径、波长和角度
        由调用方安排数组轴；例如 (radius, wavelength, 1, ice_option)
        与 (1, 1, angle, 1) 广播为完整光谱网格。

    Raises:
        ValueError: 输入为空、非有限、超出上述物理范围或无法广播。

    公式对应冻结 SPIRES/twostream.m 的 delta-Eddington 半无限、直接入射
    分支；不包含有限层 adding-doubling、多层雪或杂质混合。
    """
    omega, g, mu = np.broadcast_arrays(
        np.asarray(omega, dtype=np.float64),
        np.asarray(asymmetry, dtype=np.float64),
        np.asarray(coszen, dtype=np.float64),
    )
    if not omega.size or not all(np.isfinite(x).all() for x in (omega, g, mu)):
        raise ValueError("Optical inputs must be nonempty and finite")
    if np.any((omega < 0) | (omega > 1)):
        raise ValueError("omega must lie in [0, 1]")
    if np.any((g < 0) | (g >= 1)):
        raise ValueError("asymmetry must lie in [0, 1)")
    if np.any((mu < 0) | (mu > 1)):
        raise ValueError("coszen must lie in [0, 1]")

    # 将强前向散射的 delta 峰去除。分解 1-g²，减少 g 接近 1 时的相消。
    one_minus_g2 = (1 - g) * (1 + g)
    omega_star = one_minus_g2 * omega / ((1 - omega) + omega * one_minus_g2)
    g_star = g / (1 + g)
    a = 1 - omega_star * g_star
    b = g_star / a
    xi = np.sqrt(3 * a * (1 - omega_star))
    p = 2 * xi / (3 * a)
    # 半无限闭式解没有有限层表达式中的 1-xi*mu 分母，避免可消奇点。
    return omega_star / (1 + p) * (1 - b * xi * mu) / (1 + xi * mu)
