"""Метод percentile: глобальный перцентиль высоты растительности → биомасса.

Один скаляр h_p = np.percentile(veg[:, 2], q) на (cloud, percentile). Без объёма
и сетки. Подпапки results/.../height/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.autoname import build_name, default_path
from methods import _common as common

NAME = "percentile"
COLUMNS = ["file", *common.LABEL_COLS, "percentile", "n_veg", "h_p", "error"]


def add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--percentiles", default="95",
                   help="Percentile Z растительности через запятую (default: 95)")


def add_analyze_args(p: argparse.ArgumentParser) -> None:
    pass


def _row_key(row: dict) -> str:
    return f"{row['file']}|{row['percentile']}"


def _error_rows(item: common.InputItem, msg: str) -> list[dict]:
    return [{"file": item.rel_path, **item.labels, "error": msg}]


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
    common.load_env_from_argv(argv)
    p = argparse.ArgumentParser(description=__doc__)
    common.add_common_batch_args(p)
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
                          extra=common.autoname_extra_from_args(a))
        output_csv = default_path("volume_csv", name, subfolder=NAME)
    else:
        output_csv = Path(a.output_csv)

    cfg = type("Cfg", (), {"list_file": a.list_file, "input_dir": a.input_dir,
                           "base_dir": Path(a.base_dir), "limit": a.limit})()
    spec = common.LongBatchSpec(columns=COLUMNS, row_key=_row_key,
                           error_rows=_error_rows,
                           compute_rows=_make_compute_rows(percentiles))
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
    return common.run_long_analyze(args, value_cols=["h_p"],
                              group_cols=["percentile"],
                              label_fn=lambda meta, vc: f"p{float(meta['percentile']):g}",
                              subfolder=NAME)


def main(argv=None) -> int:
    csv_path = run_batch(argv)
    common.chain_analyze(sys.modules[__name__], csv_path, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
