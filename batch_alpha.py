"""Батч-версия downsample_alpha_compare.py: для каждого облака датасета
прогоняет voxel/random downsample и считает alpha-shape (3D или послойный)
для каждой комбинации (voxel_size, alpha). Результаты пишутся в CSV.

Источник входа — либо --list (с биомассой/метками), либо --input-dir
(рекурсивный glob по *.pcd).

Формат CSV (long): по одной строке на (file, voxel_mm, alpha):
    file, biomass, col3, col4, col5, voxel_mm, n_voxel, n_random,
    alpha, mode, layer_dz_mm, V_voxel, V_random, error
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from dotenv import load_dotenv
from tqdm import tqdm

from batch_process import (
    LABEL_COLS,
    InputItem,
    collect_inputs,
    load_done_files,
)
from downsample_alpha_compare import _compute_one
from downsample_compare import random_downsample, sor as apply_sor, voxel_downsample
from generate_cloud import load_real_cloud
from tools.autoname import build_name, default_path


BASE_COLUMNS = [
    "file", *LABEL_COLS,
    "voxel_mm",
    "alpha", "mode", "layer_dz_mm",
    "n_voxel", "V_voxel",
    "error",
]
RANDOM_COLUMNS = ["n_random", "V_random"]


def csv_columns(with_random: bool) -> list[str]:
    cols = list(BASE_COLUMNS)
    if with_random:
        cols += RANDOM_COLUMNS
    return cols


@dataclass
class BatchAlphaConfig:
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    output_csv: Path
    voxel_sizes_mm: list[float]
    auto_voxel: bool
    alphas: list[float]
    layered: bool
    layer_dz_mm: float
    with_random: bool
    units: str
    flip_z: bool
    sor_neighbors: int
    sor_std_ratio: float
    seed: int
    workers: int
    resume: bool
    limit: int | None
    analyze: bool = True
    plots: bool = True


def load_and_normalize(path: Path, units: str, flip_z: bool) -> np.ndarray:
    data = load_real_cloud(str(path), units=units, verbose=False)
    pts = np.asarray(data["all_pts_noisy"]).copy()
    if len(pts) == 0:
        return pts
    if flip_z:
        pts[:, 2] = -pts[:, 2]
    pts[:, 2] -= pts[:, 2].min()
    return pts


def auto_voxel_mm(points: np.ndarray) -> float:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    return float(nn.mean()) * 1000


def build_tasks(item: InputItem, points: np.ndarray, cfg: BatchAlphaConfig
                ) -> tuple[list[tuple], list[dict]]:
    """Возвращает (tasks_for_pool, row_skeletons).

    tasks: список (idx, points, alpha, layered, dz_m); idx = (rel_path, voxel_mm, alpha, kind)
    row_skeletons: список dict с предзаполненными мета-полями, ключи как у idx до alpha+kind.
    """
    sizes_mm = list(cfg.voxel_sizes_mm)
    if cfg.auto_voxel:
        sizes_mm.append(auto_voxel_mm(points))

    tasks = []
    rows: dict[tuple[str, float, float], dict] = {}
    for size_mm in sizes_mm:
        size_m = size_mm / 1000.0
        v_pts = voxel_downsample(points, size_m)
        if len(v_pts) == 0:
            continue
        v_pts = apply_sor(v_pts, cfg.sor_neighbors, cfg.sor_std_ratio)
        n_v = len(v_pts)

        if cfg.with_random:
            r_pts = random_downsample(points, n_v, cfg.seed)
            r_pts = apply_sor(r_pts, cfg.sor_neighbors, cfg.sor_std_ratio)
            n_r = len(r_pts)
        else:
            r_pts = None
            n_r = None

        for a in cfg.alphas:
            base = {
                "file": item.rel_path, **item.labels,
                "voxel_mm": size_mm, "n_voxel": n_v,
                "alpha": a,
                "mode": "layered" if cfg.layered else "3d",
                "layer_dz_mm": cfg.layer_dz_mm if cfg.layered else "",
                "V_voxel": "",
                "error": "",
            }
            if cfg.with_random:
                base["n_random"] = n_r
                base["V_random"] = ""
            key = (item.rel_path, size_mm, a)
            rows[key] = base
            dz_m = cfg.layer_dz_mm / 1000.0
            tasks.append(((key, "voxel"), v_pts, a, cfg.layered, dz_m))
            if cfg.with_random:
                tasks.append(((key, "random"), r_pts, a, cfg.layered, dz_m))
    return tasks, rows


def parse_args(argv: Iterable[str] | None = None) -> BatchAlphaConfig:
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
    src.add_argument("--list", dest="list_file")
    src.add_argument("--input-dir")
    p.add_argument("--base-dir",
                   default=os.getenv("TPCVE_BASE_DIR", "data"),
                   help="База для путей из --list "
                        "(env: TPCVE_BASE_DIR, default: data)")
    p.add_argument("--output-csv", default=None,
                   help="Если не задан — имя строится автоматически в "
                        "results/volume_csv/alpha/")
    p.add_argument("--voxel-sizes", default=None,
                   help="Размеры вокселей в мм через запятую")
    p.add_argument("--auto", action="store_true",
                   help="Добавить размер = среднее nn-расстояние (на каждое облако)")
    p.add_argument("--alphas", default="10,20",
                   help="Значения α через запятую (default: 10,20)")
    p.add_argument("--layered", action="store_true",
                   help="Послойный объём (2D × dz)")
    p.add_argument("--layer-dz", type=float, default=20.0,
                   help="Толщина слоя в мм (default: 20)")
    p.add_argument("--with-random", action="store_true",
                   help="Дополнительно считать random downsample той же N "
                        "(колонки n_random, V_random)")
    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower()
                   in ("1", "true", "yes"))
    p.add_argument("--sor-neighbors", type=int, default=20)
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO", "2.0")),
                   help="SOR std_ratio (default: 2.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="После batch вызвать analyze_correlation_alpha.py "
                        "(default: on; отключить --no-analyze)")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="При --analyze сохранить scatter-плоты в "
                        "results/regression_plots/alpha/<stem>/ "
                        "(default: on; отключить --no-plots)")
    a = p.parse_args(argv)

    voxel_sizes = ([float(x) for x in a.voxel_sizes.split(",")]
                   if a.voxel_sizes else [])
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    if not alphas:
        p.error("Нужен хотя бы один --alphas")
    if not voxel_sizes and not a.auto:
        p.error("Нужен --voxel-sizes или --auto")

    if a.output_csv is None:
        extra: dict = {}
        if abs(a.sor_std_ratio - 2.0) > 1e-9:
            extra["sor"] = a.sor_std_ratio
        if a.flip_z:
            extra["flipz"] = True
        if a.with_random:
            extra["rand"] = True
        name = build_name(
            source=a.list_file or a.input_dir,
            source_kind="list" if a.list_file else "dir",
            voxels_mm=voxel_sizes,
            auto_voxel=a.auto,
            alphas=alphas,
            layered=a.layered,
            layer_dz_mm=a.layer_dz if a.layered else None,
            extra=extra,
        )
        output_csv = default_path("volume_alpha", name, ".csv")
    else:
        output_csv = Path(a.output_csv)

    return BatchAlphaConfig(
        list_file=a.list_file, input_dir=a.input_dir,
        base_dir=Path(a.base_dir), output_csv=output_csv,
        voxel_sizes_mm=voxel_sizes, auto_voxel=a.auto,
        alphas=alphas,
        layered=a.layered, layer_dz_mm=a.layer_dz,
        with_random=a.with_random,
        units=a.units, flip_z=a.flip_z,
        sor_neighbors=a.sor_neighbors,
        sor_std_ratio=a.sor_std_ratio,
        seed=a.seed, workers=a.workers,
        resume=a.resume, limit=a.limit,
        analyze=a.analyze, plots=a.plots,
    )


def _row_key(row: dict) -> str:
    """Строковый ключ строки CSV для resume (file|voxel_mm|alpha)."""
    return f"{row['file']}|{row['voxel_mm']}|{row['alpha']}"


def load_done_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {_row_key(row) for row in reader if row.get("file")}


def process_batch(cfg: BatchAlphaConfig) -> int:
    # collect_inputs использует поля list_file/input_dir/base_dir/limit;
    # подменяем интерфейс через ad-hoc namespace.
    items_cfg = type("X", (), {
        "list_file": cfg.list_file, "input_dir": cfg.input_dir,
        "base_dir": cfg.base_dir, "limit": cfg.limit,
    })()
    items = collect_inputs(items_cfg)
    print(f"Файлов на входе: {len(items)}")

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(cfg.output_csv) if cfg.resume else set()
    file_exists = cfg.output_csv.exists()
    mode = "a" if (cfg.resume and file_exists) else "w"

    ex = ProcessPoolExecutor(max_workers=cfg.workers) if cfg.workers > 1 else None
    t0 = time.time()
    n_done = 0
    n_err = 0
    try:
        with open(cfg.output_csv, mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns(cfg.with_random),
                                    extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
                f.flush()

            file_bar = tqdm(items, unit="cloud", dynamic_ncols=True)
            for item in file_bar:
                file_bar.set_postfix_str(item.rel_path[-40:], refresh=False)
                if not item.full_path.exists():
                    writer.writerow({"file": item.rel_path, **item.labels,
                                     "error": f"not found: {item.full_path}"})
                    f.flush()
                    n_err += 1
                    continue

                try:
                    pts = load_and_normalize(item.full_path, cfg.units, cfg.flip_z)
                    if len(pts) == 0:
                        writer.writerow({"file": item.rel_path, **item.labels,
                                         "error": "empty cloud"})
                        f.flush()
                        n_err += 1
                        continue
                    tasks, rows = build_tasks(item, pts, cfg)
                except Exception as e:
                    writer.writerow({"file": item.rel_path, **item.labels,
                                     "error": f"{type(e).__name__}: {e}"})
                    f.flush()
                    n_err += 1
                    continue

                # фильтр resume
                if done_keys:
                    keep_keys = {k for k in rows
                                 if _row_key(rows[k]) not in done_keys}
                    tasks = [t for t in tasks if t[0][0] in keep_keys]
                    rows = {k: v for k, v in rows.items() if k in keep_keys}
                if not tasks:
                    continue

                # параллельный compute alpha-shape
                inner = tqdm(total=len(tasks), unit="task",
                             leave=False, dynamic_ncols=True,
                             desc=f"  α-shape ({item.rel_path[-25:]})")
                if ex is None:
                    for task in tasks:
                        (key, kind), _pts, _a, _l, _dz = task
                        idx, _, _, vol = _compute_one(task)
                        col = "V_voxel" if kind == "voxel" else "V_random"
                        rows[key][col] = vol
                        inner.update(1)
                else:
                    futs = {ex.submit(_compute_one, t): t[0] for t in tasks}
                    for fut in as_completed(futs):
                        try:
                            _idx, _kind, _payload, vol = fut.result()
                        except Exception as e:
                            key, kind = futs[fut]
                            rows[key]["error"] = f"{type(e).__name__}: {e}"
                            inner.update(1)
                            continue
                        key, kind = futs[fut]
                        col = "V_voxel" if kind == "voxel" else "V_random"
                        rows[key][col] = vol
                        inner.update(1)
                inner.close()

                # записать строки
                for key in sorted(rows):
                    writer.writerow(rows[key])
                f.flush()
                n_done += len(rows)
    finally:
        if ex is not None:
            ex.shutdown(wait=True)

    print(f"\nГотово за {time.time() - t0:.1f}s. Строк добавлено: {n_done} "
          f"(ошибок файлов: {n_err}). CSV: {cfg.output_csv}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    cfg = parse_args(argv)
    rc = process_batch(cfg)
    if cfg.analyze:
        import subprocess
        cmd = [sys.executable, "analyze_correlation_alpha.py", str(cfg.output_csv)]
        if cfg.plots:
            cmd.append("--plots-dir")
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
