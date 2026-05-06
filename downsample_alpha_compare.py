"""То же что downsample_compare.py, но дополнительно для каждого downsample
строит 3D alpha-shape и сохраняет его как меш поверх точек.

Использование:
    uv run python downsample_alpha_compare.py data/Yanco-1-1-1-b/1-1-1-b.pcd \\
        --auto --alphas 5 10 20
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import alphashape
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from downsample_compare import (
    random_downsample,
    sor as apply_sor,
    voxel_downsample,
)
from generate_cloud import load_real_cloud


def alpha_mesh(points: np.ndarray, alpha: float):
    """Возвращает (vertices, faces, volume) или (None, None, 0.0)."""
    if len(points) < 4:
        return None, None, 0.0
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


def alpha_layered(points: np.ndarray, alpha: float, dz: float):
    """Послойный объём: режем по Z с шагом dz, в каждом слое 2D alpha-shape.

    Возвращает (rings_per_layer, total_volume), где rings_per_layer —
    список (z_center, [ring_xy, ...]) для отрисовки.
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
        polygon = alphashape.alphashape(layer_xy, alpha)
        area = float(getattr(polygon, "area", 0.0) or 0.0)
        if area <= 0:
            continue
        total += area * (z1 - z0)
        layers.append((0.5 * (z0 + z1), _polygon_rings(polygon)))
    return layers, total


def _compute_one(task):
    """Воркер для пула процессов.

    task: (idx, points, alpha, layered, dz)
    return: (idx, kind, payload, volume)
    """
    idx, points, alpha, layered, dz = task
    if layered:
        layers, vol = alpha_layered(points, alpha, dz)
        return idx, "layers", layers, vol
    v, f, vol = alpha_mesh(points, alpha)
    return idx, "mesh", (v, f), vol


def add_cloud(fig, points, row, col, color, name):
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=color, opacity=0.6),
        name=name, showlegend=False,
    ), row=row, col=col)


def add_mesh(fig, verts, faces, row, col, color):
    if verts is None or faces is None:
        return
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=0.35, flatshading=True,
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)


def add_layers(fig, layers, row, col, color):
    for z_c, rings in layers:
        for ring in rings:
            fig.add_trace(go.Scatter3d(
                x=ring[:, 0], y=ring[:, 1],
                z=np.full(len(ring), z_c),
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=col)


def make_figure(voxel_pts, random_pts, alphas, voxel_mm, seed, source_name,
                layered=False, layer_dz=0.02, volumes_out=None,
                workers=1):
    n_rows = len(alphas)
    tag = f"layered dz={layer_dz*1000:g}мм" if layered else "3D"

    # Сборка задач: для каждой α — две задачи (voxel, random).
    # idx кодирует (row, col): row = i (1..n_rows), col = 1 (voxel) | 2 (random).
    tasks = []
    for i, a in enumerate(alphas, start=1):
        tasks.append(((i, 1), voxel_pts, a, layered, layer_dz))
        tasks.append(((i, 2), random_pts, a, layered, layer_dz))

    results: dict[tuple[int, int], tuple] = {}
    bar = tqdm(total=len(tasks), desc=f"alpha-shape ({tag}, workers={workers})",
               unit="mesh", leave=False, dynamic_ncols=True)
    if workers <= 1:
        for task in tasks:
            idx, kind, payload, vol = _compute_one(task)
            results[idx] = (kind, payload, vol)
            bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_compute_one, t) for t in tasks]
            for fut in as_completed(futures):
                idx, kind, payload, vol = fut.result()
                results[idx] = (kind, payload, vol)
                bar.update(1)
    bar.close()

    titles = []
    items = {"voxel": [], "random": []}
    for i, a in enumerate(alphas, start=1):
        v_kind, v_payload, v_vol = results[(i, 1)]
        r_kind, r_payload, r_vol = results[(i, 2)]
        items["voxel"].append((v_kind, v_payload, v_vol))
        items["random"].append((r_kind, r_payload, r_vol))
        titles.append(f"Voxel · α={a:g} ({tag}) · V={v_vol:.5f} м³")
        titles.append(f"Random · α={a:g} ({tag}) · V={r_vol:.5f} м³")
        if volumes_out is not None:
            volumes_out.append((a, v_vol, r_vol))

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]] * n_rows,
        subplot_titles=titles,
        horizontal_spacing=0.02, vertical_spacing=0.04,
    )
    for i in range(1, n_rows + 1):
        add_cloud(fig, voxel_pts, i, 1, "steelblue", "voxel")
        add_cloud(fig, random_pts, i, 2, "orange", "random")
        for col_idx, key, color in ((1, "voxel", "steelblue"),
                                    (2, "random", "orange")):
            kind, payload, _ = items[key][i - 1]
            if kind == "mesh":
                v, f = payload
                add_mesh(fig, v, f, i, col_idx, color)
            else:
                add_layers(fig, payload, i, col_idx, color)

    scene_kw = dict(
        xaxis_title="X, м", yaxis_title="Y, м", zaxis_title="Z, м",
        aspectmode="data",
    )
    layout = dict(
        title=f"{source_name}: voxel={voxel_mm:g} мм | "
              f"voxel N={len(voxel_pts):,}, random N={len(random_pts):,} (seed={seed})",
        height=550 * n_rows, margin=dict(l=0, r=0, t=80, b=0),
    )
    for k in range(1, n_rows * 2 + 1):
        layout[f"scene{'' if k == 1 else k}"] = scene_kw
    fig.update_layout(**layout)
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cloud")
    p.add_argument("voxel_sizes_mm", type=float, nargs="*",
                   help="Размеры вокселей в мм (через пробел)")
    p.add_argument("--auto", action="store_true",
                   help="Добавить размер = среднее nn-расстояние")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[5.0, 10.0, 20.0],
                   help="Значения alpha (default: 5 10 20)")
    p.add_argument("--layered", action="store_true",
                   help="Послойный объём: режем по Z с шагом --layer-dz, "
                        "в каждом слое 2D alpha-shape, V = Σ area_i · dz")
    p.add_argument("--layer-dz", type=float, default=20.0,
                   help="Толщина слоя в мм (default: 20)")
    p.add_argument("--units", default="auto",
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sor", action="store_true",
                   help="Применить SOR после каждого downsample")
    p.add_argument("--sor-neighbors", type=int, default=20)
    p.add_argument("--sor-std-ratio", type=float, default=2.0)
    p.add_argument("--output-dir", default="results/downsample_alpha")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2),
                   help="Процессы для параллельного alpha-shape "
                        "(default: cpu/2)")
    args = p.parse_args()

    data = load_real_cloud(args.cloud, units=args.units)
    pts = np.asarray(data["all_pts_noisy"])
    source_name = Path(args.cloud).stem
    print(f"Исходное облако {source_name}: {len(pts):,} точек")

    sizes_mm = list(args.voxel_sizes_mm)
    if args.auto:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        nn = np.asarray(pcd.compute_nearest_neighbor_distance())
        mean_nn_mm = float(nn.mean()) * 1000
        print(f"  --auto: среднее nn-расстояние = {mean_nn_mm:.3f} мм")
        sizes_mm.append(mean_nn_mm)

    if not sizes_mm:
        raise SystemExit("Не задано ни одного размера (передайте voxel_sizes_mm или --auto)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []  # (size_mm, alpha, v_vol, r_vol)
    for size_mm in sizes_mm:
        size_m = size_mm / 1000.0
        v_pts = voxel_downsample(pts, size_m)
        r_pts = random_downsample(pts, len(v_pts), args.seed)
        msg = (f"  voxel={size_mm:g} мм → {len(v_pts):,} | "
               f"random({args.seed}) → {len(r_pts):,}")
        if args.sor:
            n_v0, n_r0 = len(v_pts), len(r_pts)
            v_pts = apply_sor(v_pts, args.sor_neighbors, args.sor_std_ratio)
            r_pts = apply_sor(r_pts, args.sor_neighbors, args.sor_std_ratio)
            msg += (f" | SOR(k={args.sor_neighbors},σ={args.sor_std_ratio}): "
                    f"voxel {n_v0:,}→{len(v_pts):,}, "
                    f"random {n_r0:,}→{len(r_pts):,}")
        print(msg)

        vols = []
        fig = make_figure(v_pts, r_pts, args.alphas, size_mm,
                          args.seed, source_name,
                          layered=args.layered,
                          layer_dz=args.layer_dz / 1000.0,
                          volumes_out=vols,
                          workers=args.workers)
        suffix = f"_layered{args.layer_dz:g}mm" if args.layered else ""
        out = out_dir / f"{source_name}_voxel_{size_mm:g}mm{suffix}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        print(f"    → {out}")
        for a, v_vol, r_vol in vols:
            summary.append((size_mm, a, v_vol, r_vol))

    tag = f"layered dz={args.layer_dz:g}мм" if args.layered else "3D"
    print(f"\nСводная таблица объёмов ({tag}):")
    print("=" * 64)
    print(f"{'voxel,мм':>10}  {'α':>6}  {'V_voxel,м³':>14}  {'V_random,м³':>14}")
    print("-" * 64)
    for size_mm, a, v, r in summary:
        print(f"{size_mm:10.3f}  {a:6g}  {v:14.6f}  {r:14.6f}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
