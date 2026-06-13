"""Метод voxel: объём вокселизации растительности → биомасса. Sweep по voxel_size.

Long-формат: строка на (file, voxel_mm). Объём через volume_methods.voxel_volume.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.autoname import build_name, default_path
from volume_methods import DEFAULT_VOXEL_SIZES, voxel_volume
from methods import _common as C

NAME = "voxel"
COLUMNS = ["file", *C.LABEL_COLS, "voxel_mm", "n_vegetation", "V_voxel", "error"]


def add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--voxel-sizes", default=None,
                   help=f"Размеры вокселей в мм через запятую "
                        f"(default: {[v * 1000 for v in DEFAULT_VOXEL_SIZES]})")


def add_analyze_args(p: argparse.ArgumentParser) -> None:
    pass


def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['voxel_mm']}"


def _error_rows(item: C.InputItem, msg: str) -> list[dict]:
    return [{"file": item.rel_path, **item.labels, "error": msg}]


def _make_compute_rows(sizes_mm):
    def compute_rows(item, res, done_keys) -> list[dict]:
        veg = res.vegetation
        if len(veg) == 0:
            return [{"file": item.rel_path, **item.labels, "error": "empty cloud"}]
        out = []
        for size_mm in sizes_mm:
            vol, _ = voxel_volume(veg, size_mm / 1000.0)
            row = {"file": item.rel_path, **item.labels,
                   "voxel_mm": size_mm, "n_vegetation": int(len(veg)),
                   "V_voxel": vol, "error": ""}
            if _row_key(row) in done_keys:
                continue
            out.append(row)
        return out
    return compute_rows


def run_batch(argv=None) -> Path:
    C.load_env_from_argv(argv)
    p = argparse.ArgumentParser(description=__doc__)
    C.add_common_batch_args(p)
    add_batch_args(p)
    a, _ = p.parse_known_args(argv)  # known_args: при --method a,b чужие флаги игнор

    sizes_mm = ([float(x) for x in a.voxel_sizes.split(",") if x.strip()]
                if a.voxel_sizes else [v * 1000 for v in DEFAULT_VOXEL_SIZES])
    voxels_token = ([float(x) for x in a.voxel_sizes.split(",") if x.strip()]
                    if a.voxel_sizes else None)  # quirk: дефолт без токена в имени

    if a.output_csv is None:
        name = build_name(source=a.list_file or a.input_dir,
                          source_kind="list" if a.list_file else "dir",
                          voxels_mm=voxels_token,
                          extra=C.autoname_extra_from_args(a))
        output_csv = default_path("volume_voxel", name, ".csv")
    else:
        output_csv = Path(a.output_csv)

    cfg = type("Cfg", (), {"list_file": a.list_file, "input_dir": a.input_dir,
                           "base_dir": Path(a.base_dir), "limit": a.limit})()
    spec = C.LongBatchSpec(columns=COLUMNS, row_key=_row_key,
                           error_rows=_error_rows,
                           compute_rows=_make_compute_rows(sizes_mm))
    pre = C.preprocess_config_from_args(a)
    C.run_long_batch(spec, items=C.collect_for(cfg, None), csv_path=output_csv,
                     resume=a.resume, preprocess=pre, label="train")
    if a.list_test:
        test_csv = output_csv.with_name(output_csv.stem + "_test"
                                        + output_csv.suffix)
        C.run_long_batch(spec, items=C.collect_for(cfg, a.list_test),
                         csv_path=test_csv, resume=a.resume, preprocess=pre,
                         label="test")
    return output_csv


def run_analyze(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--test-csv", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--plots-dir", nargs="?", const="__auto__", default=None)
    p.add_argument("--target", default="biomass")
    p.add_argument("--top", type=int, default=None)
    add_analyze_args(p)
    args, _ = p.parse_known_args(argv)  # known_args: терпим к чужим флагам при мульти-методе
    return C.run_long_analyze(args, value_cols=["V_voxel"],
                              group_cols=["voxel_mm"],
                              label_fn=lambda meta, vc: f"v{float(meta['voxel_mm']):g}",
                              kind_regression="regression_voxel",
                              kind_plots="regression_plots_voxel")


def main(argv=None) -> int:
    csv_path = run_batch(argv)
    C.chain_analyze(sys.modules[__name__], csv_path, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
