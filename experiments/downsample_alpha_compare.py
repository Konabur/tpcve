"""Интерактивное HTML-сравнение alpha-shape поверх прореженного облака.

Для каждого размера вокселя строит два облака — voxel- и random-downsample той
же мощности — и накладывает на каждое alpha-shape (3D-меш или послойные кольца)
для набора значений alpha. Результат — Plotly-сетка (alpha × {voxel,random}),
сохраняется в HTML. Инструмент для глазной оценки, как способ прореживания и
alpha влияют на форму/объём.

Использование:
    uv run python experiments/downsample_alpha_compare.py data/Yanco-1-1-1-b/1-1-1-b.pcd \\
        --auto --alphas 5 10 20
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from tpcve.cloud.geometry import _compute_one, random_downsample, sor_np, voxel_downsample_np
from tpcve.cloud.generate_cloud import load_real_cloud
from tools.autoname import build_name, default_path


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
        tasks.append(((i, 1), voxel_pts, a, layered, layer_dz, True))
        tasks.append(((i, 2), random_pts, a, layered, layer_dz, True))

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
    p.add_argument("--output-dir", default=None,
                   help="Если не задан — results/downsample/downsample_alpha/<auto>/")
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
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        nn_dist, _ = tree.query(pts, k=2)
        mean_nn_mm = float(nn_dist[:, 1].mean()) * 1000
        print(f"  --auto: среднее nn-расстояние = {mean_nn_mm:.3f} мм")
        sizes_mm.append(mean_nn_mm)

    if not sizes_mm:
        raise SystemExit("Не задано ни одного размера (передайте voxel_sizes_mm или --auto)")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        extra: dict = {}
        if args.sor:
            extra["sor"] = args.sor_std_ratio
        name = build_name(
            source=args.cloud, source_kind="list",
            voxels_mm=args.voxel_sizes_mm,
            auto_voxel=args.auto,
            alphas=args.alphas,
            layered=args.layered,
            layer_dz_mm=args.layer_dz if args.layered else None,
            extra=extra,
        )
        out_dir = default_path("downsample_alpha", name, ext="")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []  # (size_mm, alpha, v_vol, r_vol)
    for size_mm in sizes_mm:
        size_m = size_mm / 1000.0
        v_pts = voxel_downsample_np(pts, size_m)
        r_pts = random_downsample(pts, len(v_pts), args.seed)
        msg = (f"  voxel={size_mm:g} мм → {len(v_pts):,} | "
               f"random({args.seed}) → {len(r_pts):,}")
        if args.sor:
            n_v0, n_r0 = len(v_pts), len(r_pts)
            v_pts = sor_np(v_pts, args.sor_neighbors, args.sor_std_ratio)
            r_pts = sor_np(r_pts, args.sor_neighbors, args.sor_std_ratio)
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
