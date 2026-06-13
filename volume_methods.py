"""Методы оценки объёма растительного покрова по облаку точек.

Все функции принимают (N, 3) numpy-массив точек растительности (в метрах) и
возвращают объём в м³. Для батчевой обработки используется реестр METHODS:
ключ метода → функция, возвращающая dict {колонка_csv: объём}.
"""
from __future__ import annotations

import alphashape
import numpy as np
from scipy.spatial import ConvexHull

MAX_ALPHA_POINTS = 5000

DEFAULT_VOXEL_SIZES = [0.006, 0.007,
                       0.008, 0.009, 0.01, 0.012, 0.018, 0.02, 0.022, 0.025]
DEFAULT_ALPHAS = [1.0, 5.0, 10.0, 20.0, 50.0]


def voxel_volume(points: np.ndarray, voxel_size: float = 0.01):
    if len(points) == 0:
        return 0.0, 0
    indices = np.floor(points / voxel_size).astype(np.int64)
    indices -= indices.min(axis=0)
    dims = indices.max(axis=0) + 1
    keys = (indices[:, 0] * dims[1] + indices[:, 1]) * dims[2] + indices[:, 2]
    n = int(np.unique(keys).size)
    return n * (voxel_size ** 3), n


def convex_hull_volume(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    return ConvexHull(points).volume


def _subsample(points: np.ndarray, limit: int = MAX_ALPHA_POINTS,
               seed: int = 0) -> np.ndarray:
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), limit, replace=False)
    return points[idx]


def alpha_shape_volume(points: np.ndarray, alpha: float) -> float:
    if len(points) < 4:
        return 0.0
    shape = alphashape.alphashape(_subsample(points), alpha)
    return getattr(shape, "volume", 0.0)


def alpha2d_height_volume(points: np.ndarray, alpha: float) -> float:
    """2D alpha shape по проекции на XY × высота (z_max - z_min)."""
    if len(points) < 3:
        return 0.0
    shape = alphashape.alphashape(_subsample(points[:, :2]), alpha)
    area = getattr(shape, "area", 0.0)
    height = float(points[:, 2].max() - points[:, 2].min())
    return area * height


def _voxel_label(size_m: float) -> str:
    mm = size_m * 1000
    s = f"{mm:g}".replace('.', '_')
    return f"voxel_{s}mm"


def _alpha_label(a: float) -> str:
    s = f"{a:g}".replace('.', '_')
    return f"alpha_{s}"


def _alpha2d_label(a: float) -> str:
    s = f"{a:g}".replace('.', '_')
    return f"alpha2d_h_{s}"


def method_voxel(points: np.ndarray, sizes=DEFAULT_VOXEL_SIZES) -> dict:
    out = {}
    for vs in sizes:
        vol, _ = voxel_volume(points, vs)
        out[_voxel_label(vs)] = vol
    return out


def method_convex_hull(points: np.ndarray) -> dict:
    return {"convex_hull": convex_hull_volume(points)}


def method_alpha(points: np.ndarray, alphas=DEFAULT_ALPHAS) -> dict:
    return {_alpha_label(a): alpha_shape_volume(points, a) for a in alphas}


def method_alpha2d_h(points: np.ndarray, alphas=DEFAULT_ALPHAS) -> dict:
    return {_alpha2d_label(a): alpha2d_height_volume(points, a) for a in alphas}


METHODS = {
    "voxel": method_voxel,
    "convex_hull": method_convex_hull,
    "alpha": method_alpha,
    "alpha2d_h": method_alpha2d_h,
}


def method_columns(name: str, *, voxel_sizes=DEFAULT_VOXEL_SIZES,
                   alphas=DEFAULT_ALPHAS) -> list[str]:
    """Список колонок, которые добавит метод в CSV (для шапки до запуска)."""
    if name == "voxel":
        return [_voxel_label(s) for s in voxel_sizes]
    if name == "convex_hull":
        return ["convex_hull"]
    if name == "alpha":
        return [_alpha_label(a) for a in alphas]
    if name == "alpha2d_h":
        return [_alpha2d_label(a) for a in alphas]
    raise ValueError(f"Неизвестный метод: {name}")


def run_method(name: str, points: np.ndarray, *,
               voxel_sizes=DEFAULT_VOXEL_SIZES,
               alphas=DEFAULT_ALPHAS) -> dict:
    if name == "voxel":
        return method_voxel(points, voxel_sizes)
    if name == "convex_hull":
        return method_convex_hull(points)
    if name == "alpha":
        return method_alpha(points, alphas)
    if name == "alpha2d_h":
        return method_alpha2d_h(points, alphas)
    raise ValueError(f"Неизвестный метод: {name}")
