"""Батч-версия для «простого перцентиля высоты» (без объёма): для каждого
облака считает один скаляр h_p = np.percentile(veg[:, 2], q) — глобальный
перцентиль Z по точкам растительности (после SOR + height_threshold split).

Источник входа — либо --list (с биомассой/метками), либо --input-dir
(рекурсивный glob по *.pcd).

Формат CSV (long): по одной строке на (file, percentile):
    file, biomass, col3, col4, col5, percentile, n_veg, h_p, error
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

from batch_process import LABEL_COLS, collect_inputs
from cloud_pipeline import PreprocessConfig, preprocess_cloud
from tools.autoname import build_name, default_path

COLUMNS = [
    "file", *LABEL_COLS,
    "percentile",
    "n_veg", "h_p", "error",
]


@dataclass
class BatchPercentileConfig:
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    output_csv: Path
    percentiles: list[float]
    resume: bool
    limit: int | None
    list_test: str | None = None
    output_csv_test: Path | None = None
    analyze: bool = True
    plots: bool = True
    top: int | None = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def parse_args(argv: Iterable[str] | None = None) -> BatchPercentileConfig:
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
                        "results/volume_csv/height/")
    p.add_argument("--percentiles", default="95",
                   help="Percentile Z растительности через запятую (default: 95)")
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
                   help="Передать --top N в analyze_correlation_percentile.py")
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="После batch вызвать analyze_correlation_percentile.py "
                        "(default: on; отключить --no-analyze)")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="При --analyze сохранить scatter-плоты в "
                        "results/regression_plots/height/<stem>/ "
                        "(default: on; отключить --no-plots)")
    a = p.parse_args(argv)

    percentiles = [float(x) for x in a.percentiles.split(",") if x.strip()]
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
            percentiles=percentiles,
            extra=extra,
        )
        output_csv = default_path("volume_height", name, ".csv")
    else:
        output_csv = Path(a.output_csv)

    output_csv_test = (output_csv.with_name(output_csv.stem + "_test"
                                          + output_csv.suffix)
                      if a.list_test else None)

    return BatchPercentileConfig(
        list_file=a.list_file, input_dir=a.input_dir,
        base_dir=Path(a.base_dir), output_csv=output_csv,
        percentiles=percentiles,
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
    return f"{row['file']}|{row['percentile']}"


def load_done_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {_row_key(row) for row in reader if row.get("file")}


def _collect(cfg: BatchPercentileConfig, list_file: str | None) -> list:
    items_cfg = type("X", (), {
        "list_file": list_file if list_file is not None else cfg.list_file,
        "input_dir": None if list_file is not None else cfg.input_dir,
        "base_dir": cfg.base_dir, "limit": cfg.limit,
    })()
    return collect_inputs(items_cfg)


def process_batch(cfg: BatchPercentileConfig, *, items: list | None = None,
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

            z = veg[:, 2]
            n_veg = int(len(z))
            for q in cfg.percentiles:
                base = {
                    "file": item.rel_path, **item.labels,
                    "percentile": q,
                    "n_veg": n_veg, "h_p": "", "error": "",
                }
                if _row_key(base) in done_keys:
                    continue
                try:
                    base["h_p"] = float(np.percentile(z, q))
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
        cmd = [sys.executable, "analyze_correlation_percentile.py",
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
