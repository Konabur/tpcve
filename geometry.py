"""Геометрия облака точек: прореживание и объём через alpha-shape.

Две группы функций:

- Прореживание: `voxel_downsample` (вокселизация) и `random_downsample`
  (случайная подвыборка до N точек с фиксированным seed).
- Alpha-shape объём растительности:
  - `alpha_mesh` — 3D alpha-shape (alpha=0 → convex hull), возвращает меш и объём;
  - `alpha_layered` / `alpha_layered_multi` — послойный объём: облако режется по Z
    с шагом dz, в каждом слое берётся площадь 2D alpha-shape, V = Σ area·dz;
  - `_compute_one` — единица работы для пула процессов (3D или послойный режим).
"""
from __future__ import annotations

import alphashape
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError


# --- downsample ---

def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(pcd.voxel_down_sample(voxel_size=voxel_m).points)


def random_downsample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    n = min(n, len(points))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), n, replace=False)
    return points[idx]


# --- alpha-shape геометрия ---

def alpha_mesh(points: np.ndarray, alpha: float):
    """Возвращает (vertices, faces, volume) или (None, None, 0.0).

    alpha == 0 → 3D convex hull (alphashape lib возвращает 2D-полигон
    при α=0, поэтому используем scipy.ConvexHull напрямую).
    """
    if len(points) < 4:
        return None, None, 0.0
    if alpha <= 0:
        try:
            hull = ConvexHull(points)
        except QhullError:
            return None, None, 0.0
        return points[hull.vertices], hull.simplices, float(hull.volume)
    shape = alphashape.alphashape(points, alpha)
    verts = getattr(shape, "vertices", None)
    faces = getattr(shape, "faces", None)
    vol = float(getattr(shape, "volume", 0.0) or 0.0)
    if verts is None or faces is None or len(faces) == 0:
        return None, None, vol
    return np.asarray(verts), np.asarray(faces), vol


def _polygon_rings(geom):
    """Из shapely (Multi)Polygon вернуть список замкнутых XY-колец."""
    rings = []
    geoms = getattr(geom, "geoms", None)
    if geoms is None:
        geoms = [geom]
    for g in geoms:
        ext = getattr(g, "exterior", None)
        if ext is None:
            continue
        rings.append(np.asarray(ext.coords))
    return rings


def _layer_triangles(xy: np.ndarray):
    """Delaunay-тесселяция XY-слоя + (area, circumradius) на треугольник.

    Возвращает (area, circ) — массивы по числу треугольников; либо (None, None)
    если слой вырожден / Qhull упал. Кэшируется снаружи: на одну тесселяцию
    можно потом много раз фильтровать по разным α (circ < 1/α).
    """
    if len(xy) < 3:
        return None, None
    try:
        tri = Delaunay(xy)
    except QhullError:
        return None, None
    s = tri.simplices
    pa, pb, pc = xy[s[:, 0]], xy[s[:, 1]], xy[s[:, 2]]
    a = np.linalg.norm(pb - pc, axis=1)
    b = np.linalg.norm(pa - pc, axis=1)
    c = np.linalg.norm(pa - pb, axis=1)
    sp = 0.5 * (a + b + c)
    area = np.sqrt(np.maximum(sp * (sp - a) * (sp - b) * (sp - c), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        circ = np.where(area > 0, (a * b * c) / (4.0 * area), np.inf)
    return area, circ


def _alpha_2d_area_fast(xy: np.ndarray, alpha: float) -> float:
    """Площадь 2D alpha-shape множества XY-точек для одного alpha.

    Считается как сумма площадей треугольников Delaunay, у которых радиус
    описанной окружности < 1/alpha. При alpha <= 0 фильтр снимается — это даёт
    площадь всего 2D convex hull слоя.
    """
    area, circ = _layer_triangles(xy)
    if area is None:
        return 0.0
    if alpha <= 0:
        return float(area.sum())
    return float(area[circ < (1.0 / alpha)].sum())


def alpha_layered(points: np.ndarray, alpha: float, dz: float,
                  *, with_rings: bool = False):
    """Послойный объём: облако режется по Z с шагом dz, в каждом слое берётся
    площадь 2D alpha-shape, объём = Σ area_слоя · dz.

    Площадь слоя — сумма треугольников Delaunay с circumradius < 1/alpha; это
    учитывает и внутренние пустоты shape.

    Возвращает (rings_per_layer, total_volume). `rings_per_layer` нужен только
    для визуализации и заполняется при with_rings=True; иначе он пустой.
    """
    if len(points) < 3 or dz <= 0:
        return [], 0.0
    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min <= 0:
        return [], 0.0

    edges = np.arange(z_min, z_max + dz, dz)
    layers = []
    total = 0.0
    for z0, z1 in zip(edges[:-1], edges[1:]):
        mask = (z >= z0) & (z < z1)
        layer_xy = points[mask, :2]
        if len(layer_xy) < 3:
            continue
        if with_rings:
            polygon = alphashape.alphashape(layer_xy, alpha)
            area = float(getattr(polygon, "area", 0.0) or 0.0)
            rings = _polygon_rings(polygon) if area > 0 else []
        else:
            area = _alpha_2d_area_fast(layer_xy, alpha)
            rings = []
        if area <= 0:
            continue
        total += area * (z1 - z0)
        layers.append((0.5 * (z0 + z1), rings))
    return layers, total


def alpha_layered_multi(points: np.ndarray, alphas: list[float],
                        dz: float) -> dict[float, float]:
    """Послойный объём сразу для набора alpha. Возвращает {alpha: volume}.

    На каждый Z-слой строится одна Delaunay-тесселяция, затем для каждого alpha
    площадь набирается фильтром circumradius < 1/alpha по уже готовым
    треугольникам. Дешевле, чем вызывать `alpha_layered` по разу на alpha,
    в число-alpha раз (триангуляция доминирует по времени).
    """
    out = {a: 0.0 for a in alphas}
    if len(points) < 3 or dz <= 0:
        return out
    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min <= 0:
        return out

    edges = np.arange(z_min, z_max + dz, dz)
    order = np.argsort(z, kind="stable")
    z_sorted = z[order]
    pts_sorted = points[order]
    bounds = np.searchsorted(z_sorted, edges)

    # alpha == 0 → площадь convex hull слоя (без фильтра по circumradius),
    # согласовано с `alphashape.alphashape(pts, 0)` в 3D-режиме.
    inv_alphas = [(a, (1.0 / a) if a > 0 else float("inf")) for a in alphas]
    for i, (z0, z1) in enumerate(zip(edges[:-1], edges[1:])):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi - lo < 3:
            continue
        layer_xy = pts_sorted[lo:hi, :2]
        area, circ = _layer_triangles(layer_xy)
        if area is None:
            continue
        thickness = z1 - z0
        for a, inv_a in inv_alphas:
            s = float(area[circ < inv_a].sum())
            if s > 0:
                out[a] += s * thickness
    return out


def _compute_one(task):
    """Единица работы для ProcessPoolExecutor: посчитать alpha-объём.

    Кортеж задачи `task` задаёт режим по типу элемента `alpha`:

    - один alpha (float):
        `(idx, points, alpha, layered, dz[, with_rings])`
        → `(idx, kind, payload, volume)`, где kind="layers"|"mesh",
          payload — кольца слоёв либо (vertices, faces) для визуализации.
    - набор alpha (list/tuple), только послойный режим:
        `(idx, points, [alphas], layered=True, dz)`
        → `(idx, "layers_multi", None, {alpha: volume})`.

    `idx` — непрозрачный ключ задачи, возвращается как есть (вызывающий по нему
    раскладывает результаты).
    """
    if len(task) == 6:
        idx, points, alpha, layered, dz, with_rings = task
    else:
        idx, points, alpha, layered, dz = task
        with_rings = False

    # batch-ветка: alpha — список/кортеж
    if isinstance(alpha, (list, tuple)):
        if not layered:
            raise ValueError(
                "multi-alpha _compute_one поддерживает только layered-режим")
        vols = alpha_layered_multi(points, list(alpha), dz)
        return idx, "layers_multi", None, vols

    if layered:
        layers, vol = alpha_layered(points, alpha, dz, with_rings=with_rings)
        return idx, "layers", layers, vol
    v, f, vol = alpha_mesh(points, alpha)
    return idx, "mesh", (v, f), vol
