"""Метод chm (Canopy Height Model): объём полога через сетку высот.

Точки растительности проецируются на XY-сетку с ячейкой cell_size; в каждой
ячейке берётся percentile значения Z (высота полога), объём = Σ height·cell_area.
Batch перебирает сетку (cell_size_mm × percentile).

Публичная функция:
    chm_volume(points, cell_size_m, percentile) -> (volume_m3, n_cells)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.autoname import build_name, default_path
import core as common

NAME = "chm"
COLUMNS = ["file", *common.LABEL_COLS, "cell_size_mm", "percentile",
           "n_cells", "V_chm", "error"]


def add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cell-sizes", required=True,
                   help="Размеры ячейки в мм через запятую (обязательно)")
    p.add_argument("--percentiles", default="95",
                   help="Percentile Z в ячейке через запятую (default: 95)")


def add_analyze_args(p: argparse.ArgumentParser) -> None:
    pass


# --- объём CHM ---

def _bin_and_sort(points: np.ndarray, cell_size_m: float):
    """Сгруппировать точки по XY-ячейкам и отсортировать Z в каждой группе.

    Возвращает (z_sorted, starts, ends) или None, если точек нет.
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


# --- batch ---

def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['cell_size_mm']}|{row['percentile']}"


def _make_compute_rows(cell_sizes_mm, percentiles):
    def compute_rows(item, res, done_keys) -> list[dict]:
        veg = res.vegetation
        if len(veg) == 0:
            return [{"file": item.rel_path, **item.labels, "error": "empty cloud"}]
        out = []
        for cell_mm in cell_sizes_mm:
            cell_m = cell_mm / 1000.0
            try:
                binned = _bin_and_sort(veg, cell_m)
            except Exception as e:
                binned = e
            for q in percentiles:
                base = {"file": item.rel_path, **item.labels,
                        "cell_size_mm": cell_mm, "percentile": q,
                        "n_cells": "", "V_chm": "", "error": ""}
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
                        v, n_cells = chm_volume_from_bins(z_s, starts, ends,
                                                          cell_m, q)
                        base["V_chm"] = v
                        base["n_cells"] = n_cells
                    except Exception as e:
                        base["error"] = f"{type(e).__name__}: {e}"
                out.append(base)
        return out
    return compute_rows


def run_batch(argv=None) -> Path:
    common.load_env_from_argv(argv)
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_batch_args(p)
    add_batch_args(p)
    a, _ = p.parse_known_args(argv)  # known_args: при --method a,b чужие флаги игнор

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
        name = build_name(source=a.list_file or a.input_dir,
                          source_kind="list" if a.list_file else "dir",
                          cell_sizes_mm=cell_sizes, percentiles=percentiles,
                          extra=common.autoname_extra_from_args(a))
        output_csv = default_path("volume_csv", name, subfolder=NAME)
    else:
        output_csv = Path(a.output_csv)

    spec = common.LongBatchSpec(columns=COLUMNS, row_key=_row_key,
                                error_rows=common.simple_error_rows,
                                compute_rows=_make_compute_rows(cell_sizes, percentiles))
    return common.run_batch_train_test(spec, a, output_csv)


def _label(meta, vc):
    return f"c{float(meta['cell_size_mm']):g}_p{float(meta['percentile']):g}"


def run_analyze(argv=None) -> int:
    p = common.build_analyze_parser(__doc__)
    add_analyze_args(p)
    args, _ = p.parse_known_args(argv)
    return common.run_long_analyze(
        args, value_cols=["V_chm"],
        group_cols=["cell_size_mm", "percentile"], label_fn=_label,
        subfolder=NAME)


def main(argv=None) -> int:
    return common.standard_main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
