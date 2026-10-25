"""按像元相交面积聚合，保留部分覆盖像元的有效面积比例。"""

from __future__ import annotations
import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from scipy.sparse import csr_matrix
import shapely

MODIS_CRS = CRS.from_proj4("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
MODIS_TILE_SIZE = 1111950.5196666667
MODIS_PIXEL_SIZE = MODIS_TILE_SIZE / 2400


def modis_roi_grid(bounds: tuple, source_crs: CRS, h: int = 8, v: int = 5):
    """将研究区边界投影到 MODIS 正弦网格，返回对应瓦片内的行列窗口。"""
    transformer = Transformer.from_crs(source_crs, MODIS_CRS, always_xy=True)
    left, bottom, right, top = bounds
    x, y = transformer.transform([left, left, right, right], [bottom, top, bottom, top])
    tile_left = -20015109.354 + h * MODIS_TILE_SIZE
    tile_top = 10007554.677 - v * MODIS_TILE_SIZE
    col0 = max(0, int(np.floor((min(x) - tile_left) / MODIS_PIXEL_SIZE)))
    col1 = min(2400, int(np.ceil((max(x) - tile_left) / MODIS_PIXEL_SIZE)))
    row0 = max(0, int(np.floor((tile_top - max(y)) / MODIS_PIXEL_SIZE)))
    row1 = min(2400, int(np.ceil((tile_top - min(y)) / MODIS_PIXEL_SIZE)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("ROI does not overlap the selected MODIS tile")
    transform = Affine(
        MODIS_PIXEL_SIZE,
        0,
        tile_left + col0 * MODIS_PIXEL_SIZE,
        0,
        -MODIS_PIXEL_SIZE,
        tile_top - row0 * MODIS_PIXEL_SIZE,
    )
    return transform, (row1 - row0, col1 - col0), (row0, row1, col0, col1)


def area_weights(
    source_transform: Affine,
    source_shape: tuple,
    source_crs: CRS,
    destination_transform: Affine,
    destination_shape: tuple,
    destination_crs: CRS,
) -> csr_matrix:
    """以相交面积 / 目标像元面积构建目标像元数 × 源像元数的稀疏矩阵。

    源像元的投影边界由四角连成的多边形近似，目标网格为北向上的矩形像元。
    权重适用于非整数分辨率比，行权重之和保留源数据实际覆盖的面积比例。
    """
    rows, columns = np.indices(source_shape)
    rows, columns = rows.ravel(), columns.ravel()
    vertices = []
    transformer = Transformer.from_crs(source_crs, destination_crs, always_xy=True)
    for dc, dr in ((0, 0), (1, 0), (1, 1), (0, 1)):
        x = (
            source_transform.a * (columns + dc)
            + source_transform.b * (rows + dr)
            + source_transform.c
        )
        y = (
            source_transform.d * (columns + dc)
            + source_transform.e * (rows + dr)
            + source_transform.f
        )
        if CRS.from_user_input(source_crs) != CRS.from_user_input(destination_crs):
            x, y = transformer.transform(x, y)
        vertices.append(np.column_stack((x, y)))
    source_polygons = shapely.polygons(np.stack(vertices, axis=1))
    dr, dc = np.indices(destination_shape)
    dx0 = destination_transform.a * dc.ravel() + destination_transform.c
    dy0 = destination_transform.e * dr.ravel() + destination_transform.f
    dx1 = destination_transform.a * (dc.ravel() + 1) + destination_transform.c
    dy1 = destination_transform.e * (dr.ravel() + 1) + destination_transform.f
    destination_polygons = shapely.box(
        np.minimum(dx0, dx1), np.minimum(dy0, dy1), np.maximum(dx0, dx1), np.maximum(dy0, dy1)
    )
    tree = shapely.STRtree(destination_polygons)
    source_indices, destination_indices = tree.query(source_polygons, predicate="intersects")
    areas = shapely.area(
        shapely.intersection(
            source_polygons[source_indices], destination_polygons[destination_indices]
        )
    )
    denominator = shapely.area(destination_polygons)[destination_indices]
    keep = areas > denominator * 1e-12
    weights = csr_matrix(
        (areas[keep] / denominator[keep], (destination_indices[keep], source_indices[keep])),
        shape=(np.prod(destination_shape), np.prod(source_shape)),
    )
    if np.max(np.asarray(weights.sum(axis=1))) > 1 + 1e-7:
        raise ValueError("Pixel intersections exceed the destination area")
    return weights


def aggregate(weights: csr_matrix, values: np.ndarray, valid: np.ndarray | None = None):
    """计算有效覆盖区域内的面积加权均值，同时返回有效面积 / 目标像元面积。

    源像元任一分量为非有限值或 valid 为 False 时，整行剔除。
    有效覆盖率为零的目标像元输出 NaN。
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    good = np.isfinite(values).all(axis=1)
    if valid is not None:
        good &= np.asarray(valid).ravel()
    coverage = np.asarray(weights @ good.astype(float)).ravel()
    total = weights @ np.where(good[:, None], values, 0)
    result = np.full(total.shape, np.nan)
    np.divide(total, coverage[:, None], out=result, where=coverage[:, None] > 0)
    return result, coverage
