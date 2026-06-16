"""4-панельная фигура: исходное облако и три метода оценки биомассы
(воксельный, послойная alpha-shape, CHM) на одном выбранном облаке.

Выбор облака:
    --cloud <path>     — явный путь к .pcd/.npz/...
    --list  <txt>      — взять облако с медианной биомассой из --list
    --stage Z31|Z65    — фильтр стадии (по подстроке в пути: '0828'→Z31, '1002'→Z65)

Дефолты параметров:
    voxel    = 30 мм
    alpha    = 30
    layer_dz = 50 мм   (все слои с градиентом прозрачности)
    cell     = 20 мм
    p        = 95
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core.io import pick_median_biomass, stage_from_path
from cloud_pipeline import PreprocessConfig, preprocess_cloud
from geometry import _layer_triangles


# Единый print-friendly colormap: perceptually uniform, CVD-safe, читается в ч/б.
CMAP = "cividis"


# -----------------------------------------------------------------------------
# CLI

def parse_args(argv: Iterable[str] | None = None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.env_file and os.path.exists(pre_args.env_file):
        load_dotenv(pre_args.env_file, override=True)
    elif os.path.exists(".env"):
        load_dotenv(".env", override=True)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", default=None)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cloud", help="Явный путь к облаку")
    src.add_argument("--list", dest="list_file",
                     help="Список plot'ов; выберется облако с медианной биомассой")
    p.add_argument("--base-dir", default=os.getenv("TPCVE_BASE_DIR", "data"))
    p.add_argument("--stage", default=None, choices=["Z31", "Z65"],
                   help="Фильтр стадии при выборе из --list. Стадия "
                        "определяется по подстроке в пути: '0828'→Z31, '1002'→Z65.")

    p.add_argument("--voxel-size-mm", type=float, default=30.0)
    p.add_argument("--alpha", type=float, default=30.0)
    p.add_argument("--layer-dz-mm", type=float, default=50.0)
    p.add_argument("--cell-size-mm", type=float, default=20.0)
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--xy-size", type=float, default=1.0,
                   help="Сторона XY-окна (м) для всех панелей; центрируется "
                        "на XY-центре облака (default: 1.0)")
    p.add_argument("--z-size", type=float, default=1.0,
                   help="Высота Z-окна (м) для 3D-панелей; от z_min облака "
                        "(default: 1.0)")

    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower()
                   in ("1", "true", "yes"))
    p.add_argument("--downsample", type=float,
                   default=float(os.getenv("TPCVE_DOWNSAMPLE", "0") or 0))
    p.add_argument("--sor-neighbors", type=int, default=20)
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO", "2.0")))
    p.add_argument("--min-range", type=float,
                   default=float(os.getenv("TPCVE_MIN_RANGE", "0") or 0))
    p.add_argument("--height-threshold", type=float, default=0.04)

    p.add_argument("--max-scatter", type=int, default=20000,
                   help="Если облако крупнее — рандомный сабсэмпл для scatter "
                        "(не влияет на сами методы) (default: 20000)")
    p.add_argument("--output", default=None,
                   help="Путь к PNG; если не задан — "
                        "results/figures/methods_compare/<stem>_<params>.png. "
                        "Игнорируется при --separate.")
    p.add_argument("--separate", action="store_true",
                   help="Вместо 4-панельной фигуры сохранить три парных PNG "
                        "(A+B, A+C, A+D) в "
                        "results/figures/methods_compare/<stem><suffix>/{voxel,alpha,chm}.png")
    p.add_argument("--color-gamma", type=float, default=0.6,
                   help="Gamma для PowerNorm высот; <1 осветляет нижнюю часть "
                        "шкалы (default: 0.6)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Метрики/сетки

def occupied_voxel_centers(points: np.ndarray, voxel_size: float
                           ) -> np.ndarray:
    """Уникальные центры заполненных вокселей (для bar3d/Poly3DCollection)."""
    if len(points) == 0:
        return np.zeros((0, 3))
    idx = np.floor(points / voxel_size).astype(np.int64)
    keys = idx.view([("", idx.dtype)] * 3).ravel()
    _, first = np.unique(keys, return_index=True)
    uniq_idx = idx[first]
    return (uniq_idx + 0.5) * voxel_size


def chm_grid(points: np.ndarray, cell_size: float, percentile: float
             ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """2D-сетка перцентильных высот для CHM-heatmap.

    Возвращает (grid[ny, nx], extent=(xmin, xmax, ymin, ymax)).
    Пустые ячейки — NaN.
    """
    if len(points) == 0:
        return np.zeros((1, 1)), (0.0, 1.0, 0.0, 1.0)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, ymin = float(x.min()), float(y.min())
    xmax, ymax = float(x.max()), float(y.max())
    ix = np.floor((x - xmin) / cell_size).astype(np.int64)
    iy = np.floor((y - ymin) / cell_size).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    key = ix * ny + iy
    order = np.lexsort((z, key))
    key_s, z_s = key[order], z[order]
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(key_s)) + 1, [len(key_s)]))
    starts, ends = boundaries[:-1], boundaries[1:]
    lengths = ends - starts
    idx_f = (percentile / 100.0) * (lengths - 1)
    idx_lo = np.floor(idx_f).astype(np.int64)
    idx_hi = np.minimum(idx_lo + 1, lengths - 1)
    frac = idx_f - idx_lo
    lo = z_s[starts + idx_lo]
    hi = z_s[starts + idx_hi]
    heights = lo + frac * (hi - lo)
    cell_keys = key_s[starts]
    cell_ix = cell_keys // ny
    cell_iy = cell_keys % ny
    grid = np.full((ny, nx), np.nan)  # [row=y, col=x] для imshow
    grid[cell_iy, cell_ix] = heights
    extent = (xmin, xmin + nx * cell_size, ymin, ymin + ny * cell_size)
    return grid, extent


def alpha_layers(points: np.ndarray, alpha: float, dz: float,
                 n_layers: int | None = None
                 ) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Возвращает список (z_mid, vertices_xy, triangles) по всем непустым
    слоям (или n_layers нижним, если задано)."""
    if len(points) < 3 or dz <= 0:
        return []
    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min <= 0:
        return []
    edges = np.arange(z_min, z_max + dz, dz)
    order = np.argsort(z, kind="stable")
    z_sorted = z[order]
    pts_sorted = points[order]
    bounds = np.searchsorted(z_sorted, edges)
    inv_a = (1.0 / alpha) if alpha > 0 else float("inf")

    out: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i, (z0, z1) in enumerate(zip(edges[:-1], edges[1:])):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi - lo < 3:
            continue
        layer_xy = pts_sorted[lo:hi, :2]
        from scipy.spatial import Delaunay, QhullError
        try:
            tri = Delaunay(layer_xy)
        except QhullError:
            continue
        area, circ = _layer_triangles(layer_xy)
        if area is None:
            continue
        keep = circ < inv_a
        simplices = tri.simplices[keep]
        if len(simplices) == 0:
            continue
        out.append((0.5 * (z0 + z1), layer_xy, simplices))
        if n_layers is not None and len(out) >= n_layers:
            break
    return out


# -----------------------------------------------------------------------------
# Отрисовка

def _xy_extent(pts: np.ndarray, side: float = 1.0
               ) -> tuple[float, float, float, float]:
    """Квадратное XY-окно стороной `side` (м), центрированное на XY-середине
    bbox облака. Та же геометрия, что и в _set_equal_aspect — чтобы CHM-heatmap
    совпал по площади с 3D-панелями."""
    if len(pts) == 0:
        return (-0.5 * side, 0.5 * side, -0.5 * side, 0.5 * side)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    h = 0.5 * side
    return (cx - h, cx + h, cy - h, cy + h)


def _style_3d_axes(ax) -> None:
    """Светлый фон panes, светло-серая сетка, чёрные подписи/тики."""
    white = (1.0, 1.0, 1.0, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(white)
        try:
            axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
            axis._axinfo["grid"]["linewidth"] = 0.4
        except Exception:
            pass
    ax.tick_params(colors="black")
    ax.title.set_color("black")


def _set_equal_aspect(ax, pts: np.ndarray, xy_side: float = 1.0,
                      z_side: float = 1.0) -> None:
    if len(pts) == 0:
        return
    z_min = float(pts[:, 2].min())
    x0, x1, y0, y1 = _xy_extent(pts, side=xy_side)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z_min, z_min + z_side)
    try:
        ax.set_box_aspect((1, 1, z_side / xy_side))
    except Exception:
        pass
    _style_3d_axes(ax)


def _scatter_cloud(ax, pts: np.ndarray, *, s=1.5, alpha=0.45, norm=None):
    if len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=pts[:, 2], cmap=CMAP, norm=norm,
               s=s, alpha=alpha, linewidths=0)


def _subsample(pts: np.ndarray, n_max: int, rng) -> np.ndarray:
    if len(pts) <= n_max:
        return pts
    idx = rng.choice(len(pts), size=n_max, replace=False)
    return pts[idx]


def panel_a_raw(ax, veg: np.ndarray, xy_side: float = 1.0,
                z_side: float = 1.0, norm=None):
    _scatter_cloud(ax, veg, s=6, alpha=0.85, norm=norm)
    _set_equal_aspect(ax, veg, xy_side=xy_side, z_side=z_side)
    ax.set_title("A. Облако растительности")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (м)")


def _cube_faces(c: np.ndarray, size: float) -> np.ndarray:
    """6 граней куба со стороной size, центр c. Возвращает (6, 4, 3)."""
    h = 0.5 * size
    x, y, z = c
    return np.array([
        [[x-h, y-h, z-h], [x+h, y-h, z-h], [x+h, y+h, z-h], [x-h, y+h, z-h]],
        [[x-h, y-h, z+h], [x+h, y-h, z+h], [x+h, y+h, z+h], [x-h, y+h, z+h]],
        [[x-h, y-h, z-h], [x+h, y-h, z-h], [x+h, y-h, z+h], [x-h, y-h, z+h]],
        [[x-h, y+h, z-h], [x+h, y+h, z-h], [x+h, y+h, z+h], [x-h, y+h, z+h]],
        [[x-h, y-h, z-h], [x-h, y+h, z-h], [x-h, y+h, z+h], [x-h, y-h, z+h]],
        [[x+h, y-h, z-h], [x+h, y+h, z-h], [x+h, y+h, z+h], [x+h, y-h, z+h]],
    ])


def panel_b_voxel(ax, veg: np.ndarray, voxel_size: float,
                  xy_side: float = 1.0, z_side: float = 1.0,
                  max_cubes: int = 50000, norm=None):
    _scatter_cloud(ax, veg, s=3, alpha=0.20, norm=norm)
    centers = occupied_voxel_centers(veg, voxel_size)
    if len(centers) > max_cubes:
        # защита от лютого числа граней; в обычных условиях не срабатывает
        rng = np.random.default_rng(0)
        idx = rng.choice(len(centers), size=max_cubes, replace=False)
        centers_draw = centers[idx]
        tqdm.write(f"[panel B] вокселей {len(centers)} > {max_cubes}, "
                   f"рисуем сабсэмпл")
    else:
        centers_draw = centers
    if norm is None:
        norm = Normalize(vmin=veg[:, 2].min() if len(veg) else 0.0,
                         vmax=veg[:, 2].max() if len(veg) else 1.0)
    cmap = plt.get_cmap(CMAP)
    faces = []
    face_colors = []
    iterator = tqdm(centers_draw, desc="voxels", leave=False,
                    disable=len(centers_draw) < 2000)
    for c in iterator:
        cube = _cube_faces(c, voxel_size)
        faces.extend(cube)
        col = cmap(norm(c[2]))
        face_colors.extend([col] * 6)
    if faces:
        poly = Poly3DCollection(faces, alpha=0.55, linewidths=0.3,
                                edgecolors=(0, 0, 0, 0.45))
        poly.set_facecolor(face_colors)
        ax.add_collection3d(poly)
    _set_equal_aspect(ax, veg, xy_side=xy_side, z_side=z_side)
    ax.set_title(f"B. Воксели {voxel_size*1000:.0f} мм "
                 f"(n={len(centers)})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (м)")


def panel_c_alpha(ax, veg: np.ndarray, alpha: float, dz: float,
                  xy_side: float = 1.0, z_side: float = 1.0, norm=None):
    _scatter_cloud(ax, veg, s=3, alpha=0.20, norm=norm)
    layers = alpha_layers(veg, alpha=alpha, dz=dz, n_layers=None)
    if not layers:
        ax.set_title(f"C. Alpha-shape α={alpha:g}, dz={dz*1000:.0f} мм (нет слоёв)")
        return
    if norm is None:
        z_mids = np.array([z for z, _, _ in layers])
        z_lo, z_hi = float(z_mids.min()), float(z_mids.max())
        norm = Normalize(vmin=z_lo, vmax=z_hi if z_hi > z_lo else z_lo + 1e-6)
    cmap = plt.get_cmap(CMAP)
    n = len(layers)
    # Градиент прозрачности: нижние слои прозрачнее (видны верхние),
    # верхние — плотнее. Линейно 0.25 → 0.70 для печатного контраста.
    alphas = np.linspace(0.25, 0.70, n)
    for (z_mid, xy, simplices), a in tqdm(list(zip(layers, alphas)),
                                          desc="alpha layers", leave=False):
        verts = []
        for s in simplices:
            tri = xy[s]
            verts.append(np.column_stack([tri, np.full(3, z_mid)]))
        rgba = list(cmap(norm(z_mid)))
        rgba[3] = float(a)
        poly = Poly3DCollection(verts, linewidths=0.2,
                                edgecolors=(0, 0, 0, 0.30))
        poly.set_facecolor(rgba)
        ax.add_collection3d(poly)
    _set_equal_aspect(ax, veg, xy_side=xy_side, z_side=z_side)
    ax.set_title(f"C. Alpha-shape α={alpha:g}, dz={dz*1000:.0f} мм "
                 f"({n} слоёв)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (м)")


def panel_d_chm(ax, veg: np.ndarray, cell_size: float, percentile: float, fig,
                xy_side: float = 1.0, norm=None):
    grid, extent = chm_grid(veg, cell_size=cell_size, percentile=percentile)
    im = ax.imshow(grid, origin="lower", extent=extent,
                   cmap=CMAP, norm=norm, interpolation="nearest",
                   aspect="equal")
    x0, x1, y0, y1 = _xy_extent(veg, side=xy_side)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(f"D. CHM cell={cell_size*1000:.0f} мм, p{percentile:g}")
    ax.set_xlabel("x (м)"); ax.set_ylabel("y (м)")
    ax.tick_params(colors="black")
    # colorbar в inset, чтобы не отъедать ширину у основной оси
    cax = inset_axes(ax, width="4%", height="100%", loc="center left",
                     bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
                     bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("высота (м)")
    cbar.ax.tick_params(colors="black")


# -----------------------------------------------------------------------------
# main

def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if args.cloud:
        cloud_path = Path(args.cloud)
        picked_bm: float | None = None
        stage = stage_from_path(str(cloud_path))
    else:
        cloud_path, picked_bm, stage = pick_median_biomass(
            args.list_file, Path(args.base_dir), stage=args.stage)
    if not cloud_path.exists():
        raise SystemExit(f"Не найдено: {cloud_path}")
    info = f"Облако: {cloud_path}"
    if picked_bm is not None:
        info += f"  (биомасса={picked_bm:g}, медиана"
        if args.stage:
            info += f" для {args.stage}"
        info += ")"
    if stage:
        info += f"  [стадия={stage}]"
    print(info)

    cfg = PreprocessConfig(
        units=args.units, flip_z=args.flip_z, downsample=args.downsample,
        sor_std_ratio=args.sor_std_ratio, sor_neighbors=args.sor_neighbors,
        min_range=args.min_range, height_threshold=args.height_threshold,
        verbose=args.verbose,
    )
    res = preprocess_cloud(str(cloud_path), cfg)
    veg = res.vegetation
    print(f"  n_input={res.n_input}  n_after_sor={res.n_after_sor}  "
          f"n_veg={len(veg)}")
    if len(veg) < 10:
        raise SystemExit("Слишком мало точек растительности для визуализации")

    rng = np.random.default_rng(0)
    veg_draw = _subsample(veg, args.max_scatter, rng)

    xy_side = float(args.xy_size)
    z_side = float(args.z_size)
    # Шкала высот фиксируется по z_side для сопоставимости Z31 vs Z65.
    # gamma<1 осветляет нижнюю половину cividis, чтобы Z31 (max~0.6 м) не уходил
    # в почти-чёрное при общем vmax=1.0 м.
    norm = PowerNorm(gamma=args.color_gamma, vmin=0.0, vmax=z_side)

    suptitle = f"{cloud_path.stem}"
    if stage:
        suptitle += f"  [{stage}]"
    if picked_bm is not None:
        suptitle += f"  (biomass={picked_bm:g})"

    suffix = (f"_v{args.voxel_size_mm:g}"
              f"_a{args.alpha:g}"
              f"_dz{args.layer_dz_mm:g}"
              f"_c{args.cell_size_mm:g}"
              f"_p{args.percentile:g}")
    if stage:
        suffix = f"_{stage}" + suffix

    if args.separate:
        if args.output:
            tqdm.write("[warn] --output игнорируется при --separate")
        out_dir = (Path("results") / "figures" / "methods_compare"
                   / (cloud_path.stem + suffix))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Жёсткие figure-bbox для левой и правой панелей — одинаковы во всех
        # трёх парах, чтобы A нигде не «гулял».
        LEFT_BBOX = (0.02, 0.04, 0.44, 0.86)
        RIGHT_BBOX = (0.50, 0.04, 0.44, 0.86)
        # 3D-куб matplotlib занимает ~78% своего ax-rect; 2D ужимаем до той же
        # доли, центрируя в RIGHT_BBOX, чтобы визуально совпасть с 3D-кубом.
        CUBE_FRAC = 0.78

        def _save_pair(name: str, draw_right, is_2d: bool = False):
            fig = plt.figure(figsize=(12, 6))
            ax_left = fig.add_subplot(1, 2, 1, projection="3d")
            panel_a_raw(ax_left, veg_draw, xy_side=xy_side, z_side=z_side,
                        norm=norm)
            ax_right = draw_right(fig)
            fig.suptitle(suptitle, fontsize=13)
            ax_left.set_position(LEFT_BBOX)
            if is_2d:
                x0, y0, w, h = RIGHT_BBOX
                cx, cy = x0 + w / 2, y0 + h / 2
                w2, h2 = w * CUBE_FRAC, h * CUBE_FRAC
                ax_right.set_position([cx - w2 / 2, cy - h2 / 2, w2, h2])
            else:
                ax_right.set_position(RIGHT_BBOX)
            out = out_dir / f"{name}.png"
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            tqdm.write(f"Сохранено: {out}")

        with tqdm(total=3, desc="rendering", unit="pair") as bar:
            def _voxel(fig):
                ax = fig.add_subplot(1, 2, 2, projection="3d")
                panel_b_voxel(ax, veg_draw,
                              voxel_size=args.voxel_size_mm / 1000.0,
                              xy_side=xy_side, z_side=z_side, norm=norm)
                return ax
            _save_pair("voxel", _voxel)
            bar.update(1)

            def _alpha(fig):
                ax = fig.add_subplot(1, 2, 2, projection="3d")
                panel_c_alpha(ax, veg, alpha=args.alpha,
                              dz=args.layer_dz_mm / 1000.0,
                              xy_side=xy_side, z_side=z_side, norm=norm)
                return ax
            _save_pair("alpha", _alpha)
            bar.update(1)

            def _chm(fig):
                ax = fig.add_subplot(1, 2, 2)
                panel_d_chm(ax, veg, cell_size=args.cell_size_mm / 1000.0,
                            percentile=args.percentile, fig=fig,
                            xy_side=xy_side, norm=norm)
                return ax
            _save_pair("chm", _chm, is_2d=True)
            bar.update(1)
        return 0

    fig = plt.figure(figsize=(16, 12))
    ax_a = fig.add_subplot(2, 2, 1, projection="3d")
    ax_b = fig.add_subplot(2, 2, 2, projection="3d")
    ax_c = fig.add_subplot(2, 2, 3, projection="3d")
    ax_d = fig.add_subplot(2, 2, 4)

    with tqdm(total=4, desc="rendering", unit="panel") as bar:
        panel_a_raw(ax_a, veg_draw, xy_side=xy_side, z_side=z_side, norm=norm)
        bar.update(1)
        panel_b_voxel(ax_b, veg_draw, voxel_size=args.voxel_size_mm / 1000.0,
                      xy_side=xy_side, z_side=z_side, norm=norm)
        bar.update(1)
        panel_c_alpha(ax_c, veg, alpha=args.alpha,
                      dz=args.layer_dz_mm / 1000.0,
                      xy_side=xy_side, z_side=z_side, norm=norm)
        bar.update(1)
        panel_d_chm(ax_d, veg, cell_size=args.cell_size_mm / 1000.0,
                    percentile=args.percentile, fig=fig, xy_side=xy_side,
                    norm=norm)
        bar.update(1)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output:
        out = Path(args.output)
    else:
        out = (Path("results") / "figures" / "methods_compare"
               / (cloud_path.stem + suffix + ".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    tqdm.write(f"Сохранено: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
