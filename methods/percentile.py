"""Метод percentile: глобальный перцентиль высоты растительности → биомасса.

Один скаляр h_p = np.percentile(veg[:, 2], q) на (cloud, percentile). Без объёма
и сетки.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.autoname import build_name, default_path
import core

NAME = "percentile"
COLUMNS = ["file", *core.LABEL_COLS, "percentile", "n_veg", "h_p", "error"]


def add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--percentiles", default="95",
                   help="Percentile Z растительности через запятую (default: 95)")


def add_analyze_args(p: argparse.ArgumentParser) -> None:
    pass


def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['percentile']}"


def _make_compute_rows(percentiles):
    def compute_rows(item, res, done_keys) -> list[dict]:
        veg = res.vegetation
        if len(veg) == 0:
            return [{"file": item.rel_path, **item.labels, "error": "empty cloud"}]
        z = veg[:, 2]
        n_veg = int(len(z))
        out = []
        for q in percentiles:
            base = {"file": item.rel_path, **item.labels, "percentile": q,
                    "n_veg": n_veg, "h_p": "", "error": ""}
            if _row_key(base) in done_keys:
                continue
            try:
                base["h_p"] = float(np.percentile(z, q))
            except Exception as e:
                base["error"] = f"{type(e).__name__}: {e}"
            out.append(base)
        return out
    return compute_rows


def run_batch(argv=None) -> Path:
    core.load_env_from_argv(argv)
    p = argparse.ArgumentParser(description=__doc__)
    core.add_common_batch_args(p)
    add_batch_args(p)
    a, _ = p.parse_known_args(argv)  # known_args: при --method a,b чужие флаги игнор

    percentiles = [float(x) for x in a.percentiles.split(",") if x.strip()]
    if not percentiles:
        p.error("Нужен хотя бы один --percentiles")
    for q in percentiles:
        if not 0 < q <= 100:
            p.error(f"percentile должен быть в (0, 100]: {q}")

    if a.output_csv is None:
        name = build_name(source=a.list_file or a.input_dir,
                          source_kind="list" if a.list_file else "dir",
                          percentiles=percentiles,
                          extra=core.autoname_extra_from_args(a))
        output_csv = default_path("volume_csv", name, subfolder=NAME)
    else:
        output_csv = Path(a.output_csv)

    spec = core.LongBatchSpec(columns=COLUMNS, row_key=_row_key,
                                error_rows=core.simple_error_rows,
                                compute_rows=_make_compute_rows(percentiles))
    return core.run_batch_train_test(spec, a, output_csv)


def _label(meta, vc):
    return f"p{float(meta['percentile']):g}"


def run_analyze(argv=None) -> int:
    p = core.build_analyze_parser(__doc__)
    add_analyze_args(p)
    args, _ = p.parse_known_args(argv)
    return core.run_long_analyze(args, value_cols=["h_p"],
                                   group_cols=["percentile"], label_fn=_label,
                                   subfolder=NAME)


def main(argv=None) -> int:
    return core.standard_main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
