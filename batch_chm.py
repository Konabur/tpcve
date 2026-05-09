"""Батч-версия для CHM (Canopy Height Model): для каждого облака датасета
проецирует точки растительности на XY-сетку, в каждой ячейке берёт
percentile Z, V = Σ h × cell_area. Sweep по (cell_size_mm × percentile).

Источник входа — либо --list (с биомассой/метками), либо --input-dir
(рекурсивный glob по *.pcd).

Формат CSV (long): по одной строке на (file, cell_size_mm, percentile):
    file, biomass, col3, col4, col5, cell_size_mm, percentile,
    n_cells, V_chm, error
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from batch_process import LABEL_COLS, InputItem, collect_inputs
from generate_cloud import load_real_cloud
from tools.autoname import build_name, default_path

GROUND_HEIGHT_THRESHOLD_M = 0.04

COLUMNS = [
    "file", *LABEL_COLS,
    "cell_size_mm", "percentile",
    "n_cells", "V_chm", "error",
]


@dataclass
class BatchChmConfig:
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    output_csv: Path
    cell_sizes_mm: list[float]
    percentiles: list[float]
    units: str
    flip_z: bool
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


def chm_volume(points: np.ndarray, cell_size_m: float,
               percentile: float) -> tuple[float, int]:
    if len(points) == 0:
        return 0.0, 0
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ix = np.floor((x - x.min()) / cell_size_m).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_size_m).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    order = np.argsort(key, kind="stable")
    key_s, z_s = key[order], z[order]
    uniq, starts = np.unique(key_s, return_index=True)
    ends = np.r_[starts[1:], len(key_s)]
    heights = np.fromiter(
        (np.percentile(z_s[s:e], percentile)
         for s, e in zip(starts, ends)),
        dtype=float, count=len(uniq),
    )
    area = cell_size_m ** 2
    return float(heights.sum() * area), len(uniq)


def parse_args(argv: Iterable[str] | None = None) -> BatchChmConfig:
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
                        "results/volume_csv/chm/")
    p.add_argument("--cell-sizes", required=True,
                   help="Размеры ячейки в мм через запятую (обязательно)")
    p.add_argument("--percentiles", default="95",
                   help="Percentile Z в ячейке через запятую (default: 95)")
    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower()
                   in ("1", "true", "yes"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="После batch вызвать analyze_correlation_chm.py "
                        "(default: on; отключить --no-analyze)")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="При --analyze сохранить scatter-плоты в "
                        "results/regression_plots/chm/<stem>/ "
                        "(default: on; отключить --no-plots)")
    a = p.parse_args(argv)

    cell_sizes = [float(x) for x in a.cell_sizes.split(",") if x.strip()]
    percentiles = [float(x) for x in a.percentiles.split(",") if x.strip()]
    if not cell_sizes:
        p.error("Нужен хотя бы один --cell-sizes")
    if not percentiles:
        p.error("Нужен хотя бы один --percentiles")
    for q in percentiles:
        if not 0 < q <= 100:
            p.error(f"percentile должен быть в (0, 100]: {q}")

    if a.output_csv is None:
        extra: dict = {}
        if a.flip_z:
            extra["flipz"] = True
        name = build_name(
            source=a.list_file or a.input_dir,
            source_kind="list" if a.list_file else "dir",
            cell_sizes_mm=cell_sizes,
            percentiles=percentiles,
            extra=extra,
        )
        output_csv = default_path("volume_chm", name, ".csv")
    else:
        output_csv = Path(a.output_csv)

    return BatchChmConfig(
        list_file=a.list_file, input_dir=a.input_dir,
        base_dir=Path(a.base_dir), output_csv=output_csv,
        cell_sizes_mm=cell_sizes, percentiles=percentiles,
        units=a.units, flip_z=a.flip_z,
        resume=a.resume, limit=a.limit,
        analyze=a.analyze, plots=a.plots,
    )


def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['cell_size_mm']}|{row['percentile']}"


def load_done_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {_row_key(row) for row in reader if row.get("file")}


def process_batch(cfg: BatchChmConfig) -> int:
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

    t0 = time.time()
    n_done = 0
    n_err = 0
    with open(cfg.output_csv, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
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
            except Exception as e:
                writer.writerow({"file": item.rel_path, **item.labels,
                                 "error": f"{type(e).__name__}: {e}"})
                f.flush()
                n_err += 1
                continue
            if len(pts) == 0:
                writer.writerow({"file": item.rel_path, **item.labels,
                                 "error": "empty cloud"})
                f.flush()
                n_err += 1
                continue

            veg = pts[pts[:, 2] > GROUND_HEIGHT_THRESHOLD_M]

            for cell_mm in cfg.cell_sizes_mm:
                cell_m = cell_mm / 1000.0
                for q in cfg.percentiles:
                    base = {
                        "file": item.rel_path, **item.labels,
                        "cell_size_mm": cell_mm, "percentile": q,
                        "n_cells": "", "V_chm": "", "error": "",
                    }
                    if _row_key(base) in done_keys:
                        continue
                    try:
                        v, n_cells = chm_volume(veg, cell_m, q)
                        base["V_chm"] = v
                        base["n_cells"] = n_cells
                    except Exception as e:
                        base["error"] = f"{type(e).__name__}: {e}"
                    writer.writerow(base)
                    n_done += 1
            f.flush()

    print(f"\nГотово за {time.time() - t0:.1f}s. Строк добавлено: {n_done} "
          f"(ошибок файлов: {n_err}). CSV: {cfg.output_csv}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    cfg = parse_args(argv)
    rc = process_batch(cfg)
    if cfg.analyze:
        import subprocess
        cmd = [sys.executable, "analyze_correlation_chm.py", str(cfg.output_csv)]
        if cfg.plots:
            cmd.append("--plots-dir")
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
