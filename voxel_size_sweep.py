"""Прогон voxel downsample для одного облака с шагом 1 мм от 3 до 200 мм.

Печатает таблицу (size_mm, n_points) и в конце ближайший размер к целевому
числу точек (по умолчанию 2000).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from generate_cloud import load_real_cloud


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cloud", help="Путь к .pcd/.ply/.xyz/.las/.npz")
    p.add_argument("--units", default="auto",
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--start-mm", type=float, default=3.0)
    p.add_argument("--stop-mm", type=float, default=200.0)
    p.add_argument("--step-mm", type=float, default=1.0)
    p.add_argument("--target", type=int, default=2000,
                   help="Целевое число точек (default: 2000)")
    args = p.parse_args()

    data = load_real_cloud(args.cloud, units=args.units)
    pts = np.asarray(data["all_pts_noisy"])
    print(f"Исходное облако: {len(pts):,} точек")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    sizes_mm = np.arange(args.start_mm, args.stop_mm + 1e-9, args.step_mm)
    results = []
    print(f"\n{'size_mm':>8}  {'n_points':>10}")
    print("-" * 22)
    for s_mm in sizes_mm:
        size_m = s_mm / 1000.0
        ds = pcd.voxel_down_sample(voxel_size=size_m)
        n = len(ds.points)
        results.append((s_mm, n))
        print(f"{s_mm:8.3f}  {n:10,}")

    arr = np.array(results)
    diffs = np.abs(arr[:, 1] - args.target)
    idx = int(np.argmin(diffs))
    best_mm, best_n = arr[idx]
    print("\n" + "=" * 50)
    print(f"Ближайшее к {args.target} точкам: "
          f"voxel_size = {best_mm:.3f} мм → {int(best_n):,} точек "
          f"(Δ = {int(diffs[idx])})")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
