"""Метод count: число точек (raw / pre) → биомасса.

Две строки на облако: source="raw" (n_input до обработки) и source="pre"
(число точек растительности после preprocess+классификации). Без объёма/сетки.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.autoname import build_name, default_path
from methods import _common as common

NAME = "count"
SOURCES = ("raw", "pre")
COLUMNS = ["file", *common.LABEL_COLS, "source", "n_points", "error"]


def add_batch_args(p: argparse.ArgumentParser) -> None:
    pass  # у count нет своих sweep-флагов


def add_analyze_args(p: argparse.ArgumentParser) -> None:
    pass


def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['source']}"


def _error_rows(item: common.InputItem, msg: str) -> list[dict]:
    return [{"file": item.rel_path, **item.labels, "source": s,
             "n_points": "", "error": msg} for s in SOURCES]


def _compute_rows(item, res, done_keys) -> list[dict]:
    counts = {"raw": int(res.n_input), "pre": int(len(res.vegetation))}
    out = []
    for source in SOURCES:
        row = {"file": item.rel_path, **item.labels, "source": source,
               "n_points": counts[source], "error": ""}
        if _row_key(row) in done_keys:
            continue
        out.append(row)
    return out


def run_batch(argv=None) -> Path:
    common.load_env_from_argv(argv)
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_batch_args(p)
    add_batch_args(p)
    a, _ = p.parse_known_args(argv)  # known_args: при --method a,b чужие флаги игнор

    if a.output_csv is None:
        name = build_name(source=a.list_file or a.input_dir,
                          source_kind="list" if a.list_file else "dir",
                          extra=common.autoname_extra_from_args(a))
        output_csv = default_path("volume_count", name, ".csv")
    else:
        output_csv = Path(a.output_csv)

    cfg = type("Cfg", (), {"list_file": a.list_file, "input_dir": a.input_dir,
                           "base_dir": Path(a.base_dir), "limit": a.limit})()
    spec = common.LongBatchSpec(columns=COLUMNS, row_key=_row_key,
                           error_rows=_error_rows, compute_rows=_compute_rows)
    pre = common.preprocess_config_from_args(a)
    common.run_long_batch(spec, items=common.collect_for(cfg, None), csv_path=output_csv,
                     resume=a.resume, preprocess=pre, label="train")
    if a.list_test:
        test_csv = output_csv.with_name(output_csv.stem + "_test"
                                        + output_csv.suffix)
        common.run_long_batch(spec, items=common.collect_for(cfg, a.list_test),
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
    return common.run_long_analyze(args, value_cols=["n_points"],
                              group_cols=["source"],
                              label_fn=lambda meta, vc: str(meta["source"]),
                              kind_regression="regression_count",
                              kind_plots="regression_plots_count")


def main(argv=None) -> int:
    csv_path = run_batch(argv)
    common.chain_analyze(sys.modules[__name__], csv_path, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
