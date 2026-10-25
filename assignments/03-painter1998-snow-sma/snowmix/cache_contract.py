"""核对实验配置与雪光谱正演缓存的物理参数。"""

from __future__ import annotations
import numpy as np


def validate_forward_settings(data: dict, config: dict) -> dict:
    """核对雪粒形状、雪层厚度与密度、入射谱类型及底层反照率。

    缓存保存多组入射角和两种冰光学常数选项；具体半径网格、入射角和
    光学常数选项由调用实验筛选并核验。
    """
    stored = data["input_args"][0, 0]
    mapping = {
        "snow_shape": "sno_shp",
        "depth_m": "dz",
        "density_kg_m3": "rho_snw",
        "atmosphere": "atm",
        "substrate_albedo": "R_sfc_all_wvl",
    }
    effective = {}
    for requested_key, matlab_key in mapping.items():
        value = np.asarray(stored[matlab_key], dtype=float).ravel()
        if len(value) != 1:
            raise ValueError(
                "Only a single optically thick homogeneous layer is supported by this workflow"
            )
        actual = float(value[0])
        effective[requested_key] = actual
        requested = config.get("forward", {}).get(requested_key, actual)
        if not np.isclose(actual, requested):
            raise ValueError(
                f"Frozen forward cache {requested_key}={actual:g} differs from request={requested}; regenerate a new cache"
            )
    return effective
