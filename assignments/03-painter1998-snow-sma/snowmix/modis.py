"""MOD09GA C6.1 质量筛选，保留雪状态位用于积雪分析。"""

import numpy as np


def quality_mask(
    raw_bands: np.ndarray, qc: np.ndarray, state: np.ndarray, qc_fill: int = 787410671
) -> np.ndarray:
    """按反射率范围、晴空状态、波段质量及大气校正状态生成严格质量掩膜。

    state 的第 2、10、13 位依次为云影、内部云和邻云标记，第 8–9 位为
    卷云等级；这些标记均须为零。雪标记位保留。
    """
    raw = np.asarray(raw_bands)
    qc = np.asarray(qc, dtype=np.uint32)
    state = np.asarray(state, dtype=np.uint16)
    valid = np.all((raw >= -100) & (raw <= 16000) & (raw != -28672), axis=-1)
    valid &= (qc != qc_fill) & (state != 65535)
    valid &= (state & 3) == 0  # 晴空状态为 0；未设定状态剔除
    for bit in (2, 10, 13):
        valid &= ((state >> bit) & 1) == 0
    valid &= ((state >> 8) & 3) == 0  # 卷云等级为 0
    valid &= (qc & 3) == 0  # MODLAND 总体质量为最佳等级
    for band in range(raw.shape[-1]):
        valid &= ((qc >> (2 + 4 * band)) & 15) == 0
    valid &= ((qc >> 30) & 1) == 1  # 已执行大气校正
    return valid


def subset_quality_mask(raw_bands, qc, state, bands=(0, 1, 3, 5, 6), allow_adjacent=False):
    """针对选定波段执行 QA 敏感性筛选。

    bands 使用从 0 开始的波段索引；允许 MODLAND 总体质量为 0 或 1，
    所选各波段仍须为最高质量。allow_adjacent 控制是否纳入邻云像元。
    """
    raw = np.asarray(raw_bands)
    qc = np.asarray(qc, dtype=np.uint32)
    state = np.asarray(state, dtype=np.uint16)
    selected = raw[..., list(bands)]
    valid = ((selected >= -100) & (selected <= 16000) & (selected != -28672)).all(-1)
    valid &= (state != 65535) & (qc != 787410671) & ((state & 3) == 0) & ((qc & 3) <= 1)
    for bit in (2, 10):
        valid &= ((state >> bit) & 1) == 0
    valid &= ((state >> 8) & 3) == 0
    if not allow_adjacent:
        valid &= ((state >> 13) & 1) == 0
    for band in bands:
        valid &= ((qc >> (2 + 4 * band)) & 15) == 0
    return valid & (((qc >> 30) & 1) == 1)
