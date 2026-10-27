"""同日MOD09GA与独立可见雪覆盖参照的轻量重算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .case_analysis import binary_agreement
from .diagnostics import coefficient_status, paired_rmse_difference
from .linear import select_snow


def evaluate_pixels(pixels: pd.DataFrame, library: dict, repeats: int = 1000) -> tuple:
    """从每个原生500 m单元的七波段观测重解混，在同一参照上分层评价。

    library的雪谱轴为(scene,radius,band)，背景为(band,component)。
    strict7和selected5分别使用各自QA后的四方法共同可归一化集合，
    不因类别或拟合效果改变样本。所有覆盖误差以百分点返回。
    """
    rows, deltas, supports = [], [], []
    predictions = {}
    radii = library["radii_um"]
    for i, scene in enumerate(library["scene_ids"]):
        group = pixels[pixels.scene.astype(str) == str(scene)].copy()
        observed = group[[f"B{b}" for b in range(1, 8)]].to_numpy()
        truth = group.reference_fsca.to_numpy()
        positions = group[["reference_row", "reference_column"]].to_numpy()
        for tier, bands in (("strict7", np.arange(7)), ("selected5", np.array([0, 1, 3, 5, 6]))):
            quality = group["qa_" + tier].astype(str).str.lower().isin(["true", "1"]).to_numpy()
            eligible = quality & (group.reference_coverage.to_numpy() >= 0.95) & np.isfinite(truth)
            fitted = {}
            for model, keep in (("fixed200", radii == 200), ("multi", np.ones(len(radii), bool))):
                for solver, mode in (("MGS", "original"), ("FCLS", "fcls")):
                    y = observed[:, bands].copy()
                    y[~eligible] = np.nan
                    with threadpool_limits(limits=1):
                        fitted[model + "_" + solver] = select_snow(
                            y,
                            library["snow"][i][keep][:, bands],
                            radii[keep],
                            library["backgrounds"][bands],
                            solver=mode,
                        )
            common = eligible & np.logical_and.reduce(
                [np.isfinite(result.normalized[:, 0]) for result in fitted.values()]
            )
            predictions[str(scene) + "_" + tier] = {
                key: value.normalized[:, 0] for key, value in fitted.items()
            }
            predictions[str(scene) + "_" + tier]["common"] = common
            supports.append(
                {
                    "scene": str(scene),
                    "tier": tier,
                    "reference_supported_n": int((group.reference_coverage >= 0.95).sum()),
                    "qa_supported_n": int(eligible.sum()),
                    "qa_reference_positive_ge_0p1": int(np.sum(eligible & (truth >= 0.1))),
                    "normalization_rejected_no_snow_n": int(
                        np.sum(eligible & ~common & (truth < 0.1))
                    ),
                    "normalization_rejected_snow_n": int(
                        np.sum(eligible & ~common & (truth >= 0.1))
                    ),
                    "common_n": int(common.sum()),
                    "reference_positive_ge_0p1": int(np.sum(common & (truth >= 0.1))),
                    "no_snow_n": int(np.sum(common & (truth < 0.1))),
                    "mixed_n": int(np.sum(common & (truth >= 0.1) & (truth < 0.9))),
                    "full_snow_n": int(np.sum(common & (truth >= 0.9))),
                }
            )
            for stratum in ("all", "no_snow", "mixed", "full_snow", "low_mixed", "snow_dominated"):
                selected = {
                    "all": np.ones(len(truth), bool),
                    "no_snow": truth < 0.1,
                    "mixed": (truth >= 0.1) & (truth < 0.9),
                    "full_snow": truth >= 0.9,
                    "low_mixed": (truth >= 0.1) & (truth < 0.5),
                    "snow_dominated": truth >= 0.5,
                }[stratum]
                use = common & selected
                for name, result in fitted.items():
                    error = result.normalized[use, 0] - truth[use]
                    status = coefficient_status(result.fractions[use], result.shade[use])
                    rows.append(
                        {
                            "scene": str(scene),
                            "tier": tier,
                            "method": name,
                            "stratum": stratum,
                            "n": int(use.sum()),
                            "bias_pp": float(error.mean() * 100) if len(error) else np.nan,
                            "rmse_pp": float(np.sqrt(np.mean(error**2)) * 100)
                            if len(error)
                            else np.nan,
                            "spectral_rms_pp": float(result.rms[use].mean() * 100)
                            if use.any()
                            else np.nan,
                            "negative_snow_fraction": float(status["negative_snow"].mean())
                            if use.any()
                            else np.nan,
                            "physical_feasible_fraction": float(status["feasible"].mean())
                            if use.any()
                            else np.nan,
                        }
                    )
                for first, second in (
                    ("fixed200_MGS", "multi_MGS"),
                    ("fixed200_FCLS", "multi_FCLS"),
                    ("multi_MGS", "multi_FCLS"),
                ):
                    value = paired_rmse_difference(
                        truth[use],
                        fitted[first].normalized[use, 0],
                        fitted[second].normalized[use, 0],
                        positions[use],
                        repeats=repeats,
                    )
                    deltas.append(
                        {
                            "scene": str(scene),
                            "tier": tier,
                            "stratum": stratum,
                            "first": first,
                            "second": second,
                            **value,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(deltas), pd.DataFrame(supports), predictions


def evaluate_flags(units: pd.DataFrame) -> pd.DataFrame:
    """按唯一1 km单元比较两种状态雪标志与独立参考，保留空类别为缺失值。

    参照的判雪阈值0.1、0.5、0.9单列；预测是原生二进制标志。只使用
    参考面积>=95%、晴空无邻云的单元，不附加七波段反射率质量要求。
    """
    identity = ["scene", "tile_row_1km", "tile_column_1km"]
    if units.duplicated(identity).any():
        raise ValueError("Repeated native 1 km cells in independent flag validation")
    records = []
    for scene, group in units.groupby("scene", sort=True):
        quality = group.state_clear_no_adjacent.astype(str).str.lower().isin(["true", "1"])
        keep = quality & (group.reference_coverage >= 0.95) & group.reference_fsca.notna()
        for threshold in (0.1, 0.5, 0.9):
            reference = (group.loc[keep, "reference_fsca"].to_numpy() >= threshold).astype(int)
            for bit in (12, 15):
                metrics = binary_agreement(reference, group.loc[keep, f"flag{bit}"].to_numpy(), 0.5)
                metrics.pop("threshold")
                records.append(
                    {"scene": str(scene), "bit": bit, "reference_threshold": threshold, **metrics}
                )
    return pd.DataFrame(records)
