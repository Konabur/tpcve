"""Метод voxel: объём вокселизации растительности → биомасса. Sweep по voxel_size.

Long-формат: строка на (file, voxel_mm). Объём через volume_methods.voxel_volume.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.autoname import build_name, default_path
from volume_methods import DEFAULT_VOXEL_SIZES, voxel_volume
import core as common

NAME = "voxel"
COLUMNS = ["file", *common.LABEL_COLS, "voxel_mm", "n_vegetation", "V_voxel", "error"]


def add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--voxel-sizes", default=None,
                   help=f"Размеры вокселей в мм через запятую "
                        f"(default: {[v * 1000 for v in DEFAULT_VOXEL_SIZES]})")


def add_analyze_args(p: argparse.ArgumentParser) -> None:
    pass


def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['voxel_mm']}"


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
    common.load_env_from_argv(argv)
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_batch_args(p)
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
                          extra=common.autoname_extra_from_args(a))
        output_csv = default_path("volume_csv", name, subfolder=NAME)
    else:
        output_csv = Path(a.output_csv)

    spec = common.LongBatchSpec(columns=COLUMNS, row_key=_row_key,
                                error_rows=common.simple_error_rows,
                                compute_rows=_make_compute_rows(sizes_mm))
    return common.run_batch_train_test(spec, a, output_csv)


def _label(meta, vc):
    return f"v{float(meta['voxel_mm']):g}"


def run_analyze(argv=None) -> int:
    p = common.build_analyze_parser(__doc__)
    add_analyze_args(p)
    args, _ = p.parse_known_args(argv)
    return common.run_long_analyze(args, value_cols=["V_voxel"],
                                   group_cols=["voxel_mm"], label_fn=_label,
                                   subfolder=NAME)


def main(argv=None) -> int:
    return common.standard_main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
