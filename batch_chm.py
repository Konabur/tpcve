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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from batch_process import LABEL_COLS, InputItem, collect_inputs
from cloud_pipeline import PreprocessConfig, preprocess_cloud
from tools.autoname import build_name, default_path

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
    resume: bool
    limit: int | None
    list_test: str | None = None
    output_csv_test: Path | None = None
    analyze: bool = True
    plots: bool = True
    top: int | None = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def _bin_and_sort(points: np.ndarray, cell_size_m: float):
    """Сгруппировать точки по XY-ячейкам и отсортировать Z в каждой группе.

    Возвращает (z_sorted, starts, n_cells) или None, если точек нет.
    """
    if len(points) == 0:
        return None
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ix = np.floor((x - x.min()) / cell_size_m).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_size_m).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    order = np.lexsort((z, key))
    key_s = key[order]
    z_s = z[order]
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(key_s)) + 1, [len(key_s)])
    )
    starts = boundaries[:-1]
    ends = boundaries[1:]
    return z_s, starts, ends


def chm_volume_from_bins(z_s: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                         cell_size_m: float, percentile: float
                         ) -> tuple[float, int]:
    lengths = ends - starts
    idx_f = (percentile / 100.0) * (lengths - 1)
    idx_lo = np.floor(idx_f).astype(np.int64)
    idx_hi = np.minimum(idx_lo + 1, lengths - 1)
    frac = idx_f - idx_lo
    lo = z_s[starts + idx_lo]
    hi = z_s[starts + idx_hi]
    heights = lo + frac * (hi - lo)
    return float(heights.sum() * cell_size_m * cell_size_m), int(len(starts))


def chm_volume(points: np.ndarray, cell_size_m: float,
               percentile: float) -> tuple[float, int]:
    binned = _bin_and_sort(points, cell_size_m)
    if binned is None:
        return 0.0, 0
    z_s, starts, ends = binned
    return chm_volume_from_bins(z_s, starts, ends, cell_size_m, percentile)


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
    p.add_argument("--list-test", default=None,
                   help="Опциональный второй --list-файл для теста (held-out); "
                        "пишется в <output>_test.csv той же структуры.")
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
    p.add_argument("--downsample", type=float,
                   default=float(os.getenv("TPCVE_DOWNSAMPLE", "0") or 0))
    p.add_argument("--sor-neighbors", type=int, default=20)
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO", "2.0")),
                   help="SOR std_ratio (default: 2.0)")
    p.add_argument("--min-range", type=float,
                   default=float(os.getenv("TPCVE_MIN_RANGE", "0") or 0))
    p.add_argument("--height-threshold", type=float, default=0.04,
                   help="Порог по Z для отделения земли от растительности (m)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=None,
                   help="Передать --top N в analyze_correlation_chm.py")
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
        if abs(a.sor_std_ratio - 2.0) > 1e-9:
            extra["sor"] = a.sor_std_ratio
        if a.flip_z:
            extra["flipz"] = True
        if a.downsample > 0:
            extra["ds"] = a.downsample
        if a.min_range > 0:
            extra["r"] = a.min_range
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

    output_csv_test = (output_csv.with_name(output_csv.stem + "_test"
                                          + output_csv.suffix)
                      if a.list_test else None)

    return BatchChmConfig(
        list_file=a.list_file, input_dir=a.input_dir,
        base_dir=Path(a.base_dir), output_csv=output_csv,
        cell_sizes_mm=cell_sizes, percentiles=percentiles,
        resume=a.resume, limit=a.limit,
        list_test=a.list_test, output_csv_test=output_csv_test,
        analyze=a.analyze, plots=a.plots, top=a.top,
        preprocess=PreprocessConfig(
            units=a.units,
            flip_z=a.flip_z,
            downsample=a.downsample,
            sor_std_ratio=a.sor_std_ratio,
            sor_neighbors=a.sor_neighbors,
            min_range=a.min_range,
            height_threshold=a.height_threshold,
            verbose=a.verbose,
        ),
    )


def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['cell_size_mm']}|{row['percentile']}"


def load_done_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {_row_key(row) for row in reader if row.get("file")}


def _collect(cfg: BatchChmConfig, list_file: str | None) -> list:
    items_cfg = type("X", (), {
        "list_file": list_file if list_file is not None else cfg.list_file,
        "input_dir": None if list_file is not None else cfg.input_dir,
        "base_dir": cfg.base_dir, "limit": cfg.limit,
    })()
    return collect_inputs(items_cfg)


def process_batch(cfg: BatchChmConfig, *, items: list | None = None,
                  csv_path: Path | None = None,
                  label: str = "train") -> int:
    if items is None:
        items = _collect(cfg, None)
    if csv_path is None:
        csv_path = cfg.output_csv
    print(f"[{label}] файлов на входе: {len(items)} -> {csv_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(csv_path) if cfg.resume else set()
    file_exists = csv_path.exists()
    mode = "a" if (cfg.resume and file_exists) else "w"

    t0 = time.time()
    n_done = 0
    n_err = 0
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
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
                res = preprocess_cloud(str(item.full_path), cfg.preprocess)
            except Exception as e:
                writer.writerow({"file": item.rel_path, **item.labels,
                                 "error": f"{type(e).__name__}: {e}"})
                f.flush()
                n_err += 1
                continue
            veg = res.vegetation
            if len(veg) == 0:
                writer.writerow({"file": item.rel_path, **item.labels,
                                 "error": "empty cloud"})
                f.flush()
                n_err += 1
                continue

            for cell_mm in cfg.cell_sizes_mm:
                cell_m = cell_mm / 1000.0
                # биннинг считаем один раз на cell_size, percentile
                # переиспользует z_sorted/starts/ends
                try:
                    binned = _bin_and_sort(veg, cell_m)
                except Exception as e:
                    binned = e
                for q in cfg.percentiles:
                    base = {
                        "file": item.rel_path, **item.labels,
                        "cell_size_mm": cell_mm, "percentile": q,
                        "n_cells": "", "V_chm": "", "error": "",
                    }
                    if _row_key(base) in done_keys:
                        continue
                    if isinstance(binned, Exception):
                        base["error"] = f"{type(binned).__name__}: {binned}"
                    elif binned is None:
                        base["V_chm"] = 0.0
                        base["n_cells"] = 0
                    else:
                        try:
                            z_s, starts, ends = binned
                            v, n_cells = chm_volume_from_bins(
                                z_s, starts, ends, cell_m, q)
                            base["V_chm"] = v
                            base["n_cells"] = n_cells
                        except Exception as e:
                            base["error"] = f"{type(e).__name__}: {e}"
                    writer.writerow(base)
                    n_done += 1
            f.flush()

    print(f"\nГотово за {time.time() - t0:.1f}s. Строк добавлено: {n_done} "
          f"(ошибок файлов: {n_err}). CSV: {csv_path}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    cfg = parse_args(argv)
    rc = process_batch(cfg, csv_path=cfg.output_csv, label="train")
    if cfg.list_test and cfg.output_csv_test is not None:
        items_test = _collect(cfg, cfg.list_test)
        process_batch(cfg, items=items_test,
                      csv_path=cfg.output_csv_test, label="test")
    if cfg.analyze:
        import subprocess
        cmd = [sys.executable, "analyze_correlation_chm.py",
               str(cfg.output_csv)]
        if cfg.output_csv_test is not None:
            cmd += ["--test-csv", str(cfg.output_csv_test)]
        if cfg.plots:
            cmd.append("--plots-dir")
        if cfg.top is not None:
            cmd += ["--top", str(cfg.top)]
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
