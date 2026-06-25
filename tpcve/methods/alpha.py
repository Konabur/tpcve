"""Метод alpha: объём растительности через alpha-shape.

Растительность прореживается voxel-downsample до размера voxel_size, затем
считается alpha-shape объём — 3D-меш либо послойный (срезы по Z толщиной
layer_dz). Batch перебирает сетку (voxel_size × alpha[, layer_dz]); опционально
(--with-random) добавляет сравнение со случайным прореживанием той же мощности.

Тяжёлый этап: alpha-shape считается в пуле процессов (--workers).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import csv
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm

from tpcve.cloud.cloud_pipeline import PreprocessConfig, preprocess_cloud
from tpcve.cloud.geometry import _compute_one, random_downsample, voxel_downsample
from tools.autoname import build_name, default_path
from tpcve import core

NAME = "alpha"

BASE_COLUMNS = [
    "file", *core.LABEL_COLS,
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
class AlphaConfig:
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    output_csv: Path
    voxel_sizes_mm: list[float]
    auto_voxel: bool
    alphas: list[float]
    layer_dz_mm_list: list[float]
    with_random: bool
    seed: int
    workers: int
    resume: bool
    limit: int | None
    stage: str | None = None
    inner_progress: bool = True
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--voxel-sizes", default=None,
                   help="Размеры вокселей в мм через запятую")
    p.add_argument("--auto", action="store_true",
                   help="Добавить размер = среднее nn-расстояние (на каждое облако)")
    p.add_argument("--alphas", default="10,20",
                   help="Значения α через запятую (default: 10,20)")
    p.add_argument("--layer-dz", default=None,
                   help="Толщина слоя в мм через запятую (напр. 10,20,40). "
                        "Если передан — режим layered; иначе 3D alpha-shape.")
    p.add_argument("--with-random", action="store_true",
                   help="Дополнительно random downsample той же N "
                        "(колонки n_random, V_random)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--no-inner-progress", action="store_true",
                   help="Отключить вложенный tqdm-бар по α-shape задачам "
                        "(полезно при не-TTY выводе, напр. на Kaggle)")


def add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["voxel", "random", "both"],
                   default="voxel",
                   help="По какой колонке V строить регрессию (default: voxel)")


def auto_voxel_mm(points: np.ndarray) -> float:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    return float(nn.mean()) * 1000


def build_tasks(item: core.InputItem, points: np.ndarray, cfg: AlphaConfig,
                done_keys: set[str] | None = None
                ) -> tuple[list[tuple], dict]:
    """Возвращает (tasks_for_pool, rows). Layered: один таск на (size, dz, kind)
    с полным списком α (Delaunay-кэш на воркере). Non-layered: таск на каждый α."""
    done_keys = done_keys or set()
    sizes_mm = list(cfg.voxel_sizes_mm)
    if cfg.auto_voxel:
        sizes_mm.append(auto_voxel_mm(points))

    layered = bool(cfg.layer_dz_mm_list)
    dz_iter: list[float | None] = (list(cfg.layer_dz_mm_list)
                                   if layered else [None])

    tasks = []
    rows: dict[tuple[str, float, float, object], dict] = {}
    for size_mm in sizes_mm:
        size_m = size_mm / 1000.0
        v_pts = points if size_m <= 0 else voxel_downsample(points, size_m)
        if len(v_pts) == 0:
            continue
        n_v = len(v_pts)

        if cfg.with_random:
            r_pts = random_downsample(points, n_v, cfg.seed)
            n_r = len(r_pts)
        else:
            r_pts = None
            n_r = None

        for dz_mm in dz_iter:
            dz_label = dz_mm if layered else ""
            dz_m = (dz_mm / 1000.0) if layered else 0.0

            pending_alphas = []
            for a in cfg.alphas:
                key = (item.rel_path, size_mm, a, dz_label)
                row_key_str = (f"{item.rel_path}|{size_mm}|{a}|"
                               f"{dz_label}")
                if row_key_str in done_keys:
                    continue
                base = {
                    "file": item.rel_path, **item.labels,
                    "voxel_mm": size_mm, "n_voxel": n_v,
                    "alpha": a,
                    "mode": "layered" if layered else "3d",
                    "layer_dz_mm": dz_label,
                    "V_voxel": "",
                    "error": "",
                }
                if cfg.with_random:
                    base["n_random"] = n_r
                    base["V_random"] = ""
                rows[key] = base
                pending_alphas.append(a)

            if not pending_alphas:
                continue

            if layered:
                group_idx = (item.rel_path, size_mm, dz_label)
                tasks.append(((group_idx, "voxel"),
                              v_pts, pending_alphas, True, dz_m))
                if cfg.with_random:
                    tasks.append(((group_idx, "random"),
                                  r_pts, pending_alphas, True, dz_m))
            else:
                for a in pending_alphas:
                    key = (item.rel_path, size_mm, a, dz_label)
                    tasks.append(((key, "voxel"), v_pts, a, False, dz_m))
                    if cfg.with_random:
                        tasks.append(((key, "random"), r_pts, a, False, dz_m))
    return tasks, rows


def _row_key(row: dict) -> str:
    """Строковый ключ строки CSV для resume (file|voxel_mm|alpha|dz)."""
    return (f"{row['file']}|{row['voxel_mm']}|{row['alpha']}|"
            f"{row.get('layer_dz_mm', '')}")


def process_batch(cfg: AlphaConfig, *,
                  items: list[core.InputItem] | None = None,
                  csv_path: Path | None = None,
                  label: str = "train") -> int:
    if items is None:
        items = core.collect_for(cfg, None)
    if csv_path is None:
        csv_path = cfg.output_csv
    print(f"[{label}] файлов на входе: {len(items)} -> {csv_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = core.load_done_keys(csv_path, _row_key) if cfg.resume else set()
    file_exists = csv_path.exists()
    mode = "a" if (cfg.resume and file_exists) else "w"

    ex = ProcessPoolExecutor(max_workers=cfg.workers) if cfg.workers > 1 else None
    t0 = time.time()
    n_done = 0
    n_err = 0
    try:
        with open(csv_path, mode, encoding="utf-8", newline="") as f:
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
                    res = preprocess_cloud(str(item.full_path), cfg.preprocess)
                    pts = res.vegetation
                    if len(pts) == 0:
                        writer.writerow({"file": item.rel_path, **item.labels,
                                         "error": "empty cloud"})
                        f.flush()
                        n_err += 1
                        continue
                    tasks, rows = build_tasks(item, pts, cfg,
                                              done_keys=done_keys)
                except Exception as e:
                    writer.writerow({"file": item.rel_path, **item.labels,
                                     "error": f"{type(e).__name__}: {e}"})
                    f.flush()
                    n_err += 1
                    continue

                if not tasks:
                    continue

                def _apply_result(idx_payload, vol_or_dict):
                    key_or_group, kind = idx_payload
                    col = "V_voxel" if kind == "voxel" else "V_random"
                    if len(key_or_group) == 4:
                        rows[key_or_group][col] = vol_or_dict
                        return
                    rel, size_mm, dz_label = key_or_group
                    for a, vol in vol_or_dict.items():
                        row_key = (rel, size_mm, a, dz_label)
                        if row_key in rows:
                            rows[row_key][col] = vol

                def _apply_error(idx_payload, err_msg):
                    key_or_group, _ = idx_payload
                    if len(key_or_group) == 4:
                        rows[key_or_group]["error"] = err_msg
                        return
                    rel, size_mm, dz_label = key_or_group
                    for a in cfg.alphas:
                        row_key = (rel, size_mm, a, dz_label)
                        if row_key in rows:
                            rows[row_key]["error"] = err_msg

                inner = tqdm(total=len(tasks), unit="task",
                             leave=False, dynamic_ncols=True,
                             disable=not cfg.inner_progress,
                             desc=f"  α-shape ({item.rel_path[-25:]})")
                if ex is None:
                    for task in tasks:
                        try:
                            _idx, _kind, _payload, vol = _compute_one(task)
                        except Exception as e:
                            _apply_error(task[0], f"{type(e).__name__}: {e}")
                            inner.update(1)
                            continue
                        _apply_result(task[0], vol)
                        inner.update(1)
                else:
                    futs = {ex.submit(_compute_one, t): t[0] for t in tasks}
                    for fut in as_completed(futs):
                        try:
                            _idx, _kind, _payload, vol = fut.result()
                        except Exception as e:
                            _apply_error(futs[fut], f"{type(e).__name__}: {e}")
                            inner.update(1)
                            continue
                        _apply_result(futs[fut], vol)
                        inner.update(1)
                inner.close()

                for key in sorted(rows):
                    writer.writerow(rows[key])
                f.flush()
                n_done += len(rows)
    finally:
        if ex is not None:
            ex.shutdown(wait=True)

    print(f"\nГотово за {time.time() - t0:.1f}s. Строк добавлено: {n_done} "
          f"(ошибок файлов: {n_err}). CSV: {csv_path}")
    return 0


def run_batch(argv=None) -> Path:
    core.load_env_from_argv(argv)
    p = argparse.ArgumentParser(description=__doc__)
    core.add_common_batch_args(p)
    add_batch_args(p)
    a, _ = p.parse_known_args(argv)  # known_args: при --method a,b чужие флаги игнор

    voxel_sizes = ([float(x) for x in a.voxel_sizes.split(",")]
                   if a.voxel_sizes else [])
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    if not alphas:
        p.error("Нужен хотя бы один --alphas")
    if not voxel_sizes and not a.auto:
        p.error("Нужен --voxel-sizes или --auto")

    layer_dz_list = ([float(x) for x in a.layer_dz.split(",") if x.strip()]
                     if a.layer_dz else [])
    layered = bool(layer_dz_list)

    if a.output_csv is None:
        extra = core.autoname_extra_from_args(a)
        if a.with_random:
            extra["rand"] = True
        name = build_name(
            source=a.list_file or a.input_dir,
            source_kind="list" if a.list_file else "dir",
            voxels_mm=voxel_sizes,
            auto_voxel=a.auto,
            alphas=alphas,
            layered=layered,
            layer_dz_mm=layer_dz_list if layered else None,
            extra=extra,
        )
        output_csv = default_path("volume_csv", name, subfolder=NAME)
    else:
        output_csv = Path(a.output_csv)

    cfg = AlphaConfig(
        list_file=a.list_file, input_dir=a.input_dir,
        base_dir=Path(a.base_dir), output_csv=output_csv,
        voxel_sizes_mm=voxel_sizes, auto_voxel=a.auto,
        alphas=alphas,
        layer_dz_mm_list=layer_dz_list,
        with_random=a.with_random,
        seed=a.seed, workers=a.workers,
        resume=a.resume, limit=a.limit,
        stage=a.stage,
        inner_progress=not a.no_inner_progress,
        preprocess=core.preprocess_config_from_args(a),
    )
    process_batch(cfg, csv_path=output_csv, label="train")
    if a.list_test:
        test_csv = output_csv.with_name(output_csv.stem + "_test"
                                        + output_csv.suffix)
        items_test = core.collect_for(cfg, a.list_test)
        process_batch(cfg, items=items_test, csv_path=test_csv, label="test")
    return output_csv


def run_analyze(argv=None) -> int:
    p = core.build_analyze_parser(__doc__)
    add_analyze_args(p)
    args, _ = p.parse_known_args(argv)  # known_args: терпим к чужим флагам при мульти-методе

    value_cols = (["V_voxel", "V_random"] if args.source == "both"
                  else [f"V_{args.source}"])

    def prep_df(df):
        df["layer_dz_mm"] = df["layer_dz_mm"].fillna("").astype(str)

    def label_fn(meta, vc):
        mode = meta["mode"]
        suffix = f"_dz{meta['layer_dz_mm']}" if mode == "layered" else ""
        return (f"v{float(meta['voxel_mm']):g}_a{float(meta['alpha']):g}_"
                f"{mode}{suffix}")

    return core.run_long_analyze(
        args, value_cols=value_cols,
        group_cols=["voxel_mm", "alpha", "mode", "layer_dz_mm"],
        label_fn=label_fn, prep_df=prep_df,
        subfolder=NAME)


def main(argv=None) -> int:
    return core.standard_main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
