"""Сравнение voxel vs random downsample.

Для каждого переданного размера вокселя (мм):
  1) voxel downsample облака → получает N точек,
  2) случайный downsample того же исходника до N точек (с фиксированным seed),
  3) сохраняет 3D-визуализацию (Plotly) в HTML — два облака бок-о-бок.

Использование:
    uv run python downsample_compare.py data/Yanco-1-1-1-b/1-1-1-b.pcd 5 10 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generate_cloud import load_real_cloud


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(pcd.voxel_down_sample(voxel_size=voxel_m).points)


def random_downsample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    n = min(n, len(points))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), n, replace=False)
    return points[idx]


def sor(points: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    if len(points) <= nb_neighbors:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_sor, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio, print_progress=False
    )
    return np.asarray(pcd_sor.points)


def scatter(points: np.ndarray, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=color, opacity=0.7),
        name=name,
    )


def make_figure(voxel_pts: np.ndarray, random_pts: np.ndarray,
                voxel_mm: float, seed: int, source_name: str) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            f"Voxel downsample, size={voxel_mm:g} мм, N={len(voxel_pts):,}",
            f"Random downsample, seed={seed}, N={len(random_pts):,}",
        ),
        horizontal_spacing=0.02,
    )
    fig.add_trace(scatter(voxel_pts, "voxel", "steelblue"), row=1, col=1)
    fig.add_trace(scatter(random_pts, "random", "orange"), row=1, col=2)

    scene_kw = dict(
        xaxis_title="X, м", yaxis_title="Y, м", zaxis_title="Z, м",
        aspectmode="data",
    )
    fig.update_layout(
        title=f"{source_name}: voxel {voxel_mm:g} мм vs random "
              f"({len(voxel_pts):,} точек)",
        scene=scene_kw, scene2=scene_kw,
        showlegend=False, height=700, margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cloud", help="Путь к облаку")
    p.add_argument("voxel_sizes_mm", type=float, nargs="*",
                   help="Размеры вокселей в мм (через пробел)")
    p.add_argument("--auto", action="store_true",
                   help="Добавить размер = среднее расстояние до ближайшего соседа")
    p.add_argument("--units", default="auto",
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sor", action="store_true",
                   help="Применить SOR после каждого downsample")
    p.add_argument("--sor-neighbors", type=int, default=20,
                   help="SOR nb_neighbors (default: 20)")
    p.add_argument("--sor-std-ratio", type=float, default=2.0,
                   help="SOR std_ratio (default: 2.0)")
    p.add_argument("--output-dir", default="results/downsample_compare")
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

    sor_on = args.sor
    for size_mm in sizes_mm:
        size_m = size_mm / 1000.0
        v_pts = voxel_downsample(pts, size_m)
        r_pts = random_downsample(pts, len(v_pts), args.seed)
        msg = (f"  voxel={size_mm:g} мм → {len(v_pts):,} точек "
               f"| random({args.seed}) → {len(r_pts):,}")
        if sor_on:
            n_v_before, n_r_before = len(v_pts), len(r_pts)
            v_pts = sor(v_pts, args.sor_neighbors, args.sor_std_ratio)
            r_pts = sor(r_pts, args.sor_neighbors, args.sor_std_ratio)
            msg += (f" | SOR(k={args.sor_neighbors},σ={args.sor_std_ratio}): "
                    f"voxel {n_v_before:,}→{len(v_pts):,}, "
                    f"random {n_r_before:,}→{len(r_pts):,}")
        print(msg)

        fig = make_figure(v_pts, r_pts, size_mm, args.seed, source_name)
        out = out_dir / f"{source_name}_voxel_{size_mm:g}mm.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        print(f"    → {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
