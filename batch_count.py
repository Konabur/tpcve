"""Батч-версия для «просто количество точек → биомасса» (без объёма,
без высоты, без сетки). Для каждого облака пишет две строки:

    source = "raw" — n_points = число точек в исходном файле (без обработки)
    source = "pre" — n_points = число точек после preprocess
                     (units → flip-z → min-range → downsample → SOR)

Источник входа — либо --list (с биомассой/метками), либо --input-dir
(рекурсивный glob по *.pcd).

Формат CSV (long): по одной строке на (file, source):
    file, biomass, col3, col4, col5, source, n_points, error
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

from dotenv import load_dotenv
from tqdm import tqdm

from batch_process import LABEL_COLS, collect_inputs
from cloud_pipeline import PreprocessConfig, preprocess_cloud
from tools.autoname import build_name, default_path

COLUMNS = [
    "file", *LABEL_COLS,
    "source", "n_points", "error",
]

SOURCES = ("raw", "pre")


@dataclass
class BatchCountConfig:
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    output_csv: Path
    resume: bool
    limit: int | None
    list_test: str | None = None
    output_csv_test: Path | None = None
    analyze: bool = True
    plots: bool = True
    top: int | None = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def parse_args(argv: Iterable[str] | None = None) -> BatchCountConfig:
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
                   help="Опциональный второй --list-файл для теста (held-out).")
    p.add_argument("--base-dir",
                   default=os.getenv("TPCVE_BASE_DIR", "data"),
                   help="База для путей из --list "
                        "(env: TPCVE_BASE_DIR, default: data)")
    p.add_argument("--output-csv", default=None,
                   help="Если не задан — имя строится автоматически в "
                        "results/volume_csv/count/")
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
                   help="Не влияет на n_points, но прокидывается в "
                        "PreprocessConfig для единообразия (m)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=None,
                   help="Передать --top N в analyze_correlation_count.py")
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="После batch вызвать analyze_correlation_count.py "
                        "(default: on; отключить --no-analyze)")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="При --analyze сохранить scatter-плоты в "
                        "results/regression_plots/count/<stem>/ "
                        "(default: on; отключить --no-plots)")
    a = p.parse_args(argv)

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
            extra=extra,
        )
        output_csv = default_path("volume_count", name, ".csv")
    else:
        output_csv = Path(a.output_csv)

    output_csv_test = (output_csv.with_name(output_csv.stem + "_test"
                                          + output_csv.suffix)
                      if a.list_test else None)

    return BatchCountConfig(
        list_file=a.list_file, input_dir=a.input_dir,
        base_dir=Path(a.base_dir), output_csv=output_csv,
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
    return f"{row['file']}|{row['source']}"


def load_done_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {_row_key(row) for row in reader if row.get("file")}


def _collect(cfg: BatchCountConfig, list_file: str | None) -> list:
    items_cfg = type("X", (), {
        "list_file": list_file if list_file is not None else cfg.list_file,
        "input_dir": None if list_file is not None else cfg.input_dir,
        "base_dir": cfg.base_dir, "limit": cfg.limit,
    })()
    return collect_inputs(items_cfg)


def process_batch(cfg: BatchCountConfig, *, items: list | None = None,
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
                for source in SOURCES:
                    writer.writerow({
                        "file": item.rel_path, **item.labels,
                        "source": source, "n_points": "",
                        "error": f"not found: {item.full_path}",
                    })
                f.flush()
                n_err += 1
                continue

            try:
                res = preprocess_cloud(str(item.full_path), cfg.preprocess)
            except Exception as e:
                for source in SOURCES:
                    writer.writerow({
                        "file": item.rel_path, **item.labels,
                        "source": source, "n_points": "",
                        "error": f"{type(e).__name__}: {e}",
                    })
                f.flush()
                n_err += 1
                continue

            counts = {"raw": int(res.n_input), "pre": int(res.n_after_sor)}
            for source in SOURCES:
                row = {
                    "file": item.rel_path, **item.labels,
                    "source": source, "n_points": counts[source],
                    "error": "",
                }
                if _row_key(row) in done_keys:
                    continue
                writer.writerow(row)
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
        cmd = [sys.executable, "analyze_correlation_count.py",
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
