"""Batch-обработка датасета: для каждого .pcd файла считает выбранные методы
оценки объёма и пишет результаты в CSV (одна строка на файл).

Источник входа — либо --list (с биомассой/метками), либо --input-dir
(рекурсивный glob по *.pcd).
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

from cloud_pipeline import PreprocessConfig, preprocess_cloud
from tools.autoname import build_name, default_path
from volume_methods import (
    DEFAULT_ALPHAS,
    DEFAULT_VOXEL_SIZES,
    METHODS,
    method_columns,
    run_method,
)

# Ре-экспорт io для обратной совместимости (потребители: batch_alpha/chm/count/
# percentile, visualize_methods). __all__ помечает имена как публичные → без F401.
from methods._common import (
    InputItem,
    LABEL_COLS,
    collect_inputs,
    load_done_files,
    parse_list_line,
)

__all__ = [
    "InputItem",
    "LABEL_COLS",
    "collect_inputs",
    "load_done_files",
    "parse_list_line",
]


@dataclass
class BatchConfig:
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    output_csv: Path
    methods: list[str]
    voxel_sizes: list[float]
    alphas: list[float]
    resume: bool
    limit: int | None
    list_test: str | None = None
    output_csv_test: Path | None = None
    analyze: bool = False
    plots: bool = False
    top: int | None = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def build_columns(cfg: BatchConfig) -> list[str]:
    cols = ["file", *LABEL_COLS, "n_input", "n_after_sor", "n_vegetation"]
    for m in cfg.methods:
        cols.extend(method_columns(m, voxel_sizes=cfg.voxel_sizes,
                                   alphas=cfg.alphas))
    cols.append("error")
    return cols


def process_one(item: InputItem, cfg: BatchConfig) -> dict:
    row = {"file": item.rel_path, **item.labels}
    if not item.full_path.exists():
        row["error"] = f"file not found: {item.full_path}"
        return row
    try:
        res = preprocess_cloud(str(item.full_path), cfg.preprocess)
        row["n_input"] = res.n_input
        row["n_after_sor"] = res.n_after_sor
        row["n_vegetation"] = len(res.vegetation)
        if len(res.vegetation) == 0:
            row["error"] = "no vegetation points after filtering"
            return row
        for m in cfg.methods:
            row.update(run_method(m, res.vegetation,
                                  voxel_sizes=cfg.voxel_sizes,
                                  alphas=cfg.alphas))
        row["error"] = ""
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def parse_args(argv: Iterable[str] | None = None) -> BatchConfig:
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
    src.add_argument("--list", dest="list_file",
                     help="Текстовый список 'path biomass c3 c4 c5'")
    src.add_argument("--input-dir",
                     help="Папка с .pcd (обходится рекурсивно)")
    p.add_argument("--list-test", default=None,
                   help="Опциональный второй --list-файл для теста (held-out); "
                        "пишется в <output>_test.csv той же структуры.")
    p.add_argument("--base-dir",
                   default=os.getenv("TPCVE_BASE_DIR", "data"),
                   help="База для путей из --list "
                        "(env: TPCVE_BASE_DIR, default: data)")
    p.add_argument("--output-csv", default=None,
                   help="Если не задан — имя строится автоматически в "
                        "results/volume_csv/voxel/")
    p.add_argument("--methods", default="voxel",
                   help=f"Список через запятую из {sorted(METHODS)}")
    p.add_argument("--voxel-sizes", default=None,
                   help="Размеры вокселей в мм через запятую "
                        f"(default: {DEFAULT_VOXEL_SIZES})")
    p.add_argument("--alphas", default=None,
                   help=f"Alpha через запятую (default: {DEFAULT_ALPHAS})")
    p.add_argument("--resume", action="store_true",
                   help="Пропускать файлы, уже записанные в CSV")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=None,
                   help="Передать --top N в analyze_correlation.py")
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="После batch вызвать analyze_correlation.py "
                        "(default: on; отключить --no-analyze)")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="При --analyze сохранить scatter-плоты в "
                        "results/regression_plots/voxel/<stem>/ "
                        "(default: on; отключить --no-plots)")
    # препроцессинг
    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower()
                   in ("1", "true", "yes"))
    p.add_argument("--downsample", type=float,
                   default=float(os.getenv("TPCVE_DOWNSAMPLE", "0") or 0))
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO", "2.0")))
    p.add_argument("--min-range", type=float,
                   default=float(os.getenv("TPCVE_MIN_RANGE", "0") or 0))
    p.add_argument("--height-threshold", type=float, default=0.04)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        p.error(f"Неизвестные методы: {bad}. Доступно: {sorted(METHODS)}")

    voxel_sizes_mm = ([float(x) for x in a.voxel_sizes.split(",")]
                      if a.voxel_sizes else None)
    voxel_sizes = (DEFAULT_VOXEL_SIZES if voxel_sizes_mm is None
                   else [v / 1000 for v in voxel_sizes_mm])
    alphas_user = ([float(x) for x in a.alphas.split(",")]
                   if a.alphas else None)
    alphas = DEFAULT_ALPHAS if alphas_user is None else alphas_user

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
        method_uses_alpha = any(m in {"alpha", "hull_alpha"} for m in methods)
        name = build_name(
            source=a.list_file or a.input_dir,
            source_kind="list" if a.list_file else "dir",
            voxels_mm=voxel_sizes_mm,
            alphas=alphas_user if method_uses_alpha else None,
            extra=extra,
        )
        output_csv = default_path("volume_voxel", name, ".csv")
    else:
        output_csv = Path(a.output_csv)

    output_csv_test = (output_csv.with_name(output_csv.stem + "_test"
                                          + output_csv.suffix)
                      if a.list_test else None)

    return BatchConfig(
        list_file=a.list_file,
        input_dir=a.input_dir,
        base_dir=Path(a.base_dir),
        output_csv=output_csv,
        methods=methods,
        voxel_sizes=voxel_sizes,
        alphas=alphas,
        resume=a.resume,
        limit=a.limit,
        list_test=a.list_test,
        output_csv_test=output_csv_test,
        analyze=a.analyze,
        plots=a.plots,
        top=a.top,
        preprocess=PreprocessConfig(
            units=a.units,
            flip_z=a.flip_z,
            downsample=a.downsample,
            sor_std_ratio=a.sor_std_ratio,
            min_range=a.min_range,
            height_threshold=a.height_threshold,
            verbose=a.verbose,
        ),
    )


def write_csv(items: list[InputItem], csv_path: Path, cfg: BatchConfig,
              columns: list[str], label: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_files(csv_path) if cfg.resume else set()
    file_exists = csv_path.exists()
    mode = "a" if (cfg.resume and file_exists) else "w"
    pending = [it for it in items if it.rel_path not in done]
    print(f"[{label}] входов: {len(items)}; к обработке: {len(pending)} "
          f"(пропущено по resume: {len(items) - len(pending)}) -> {csv_path}")
    t0 = time.time()
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            f.flush()
        bar = tqdm(pending, unit="cloud", dynamic_ncols=True, desc=label)
        n_err = 0
        for item in bar:
            bar.set_postfix_str(item.rel_path[-40:], refresh=False)
            row = process_one(item, cfg)
            writer.writerow(row)
            f.flush()
            if row.get("error"):
                n_err += 1
                tqdm.write(f"ERR [{label}] {item.rel_path}: {row['error']}")
            bar.set_postfix(err=n_err, refresh=False)
    print(f"[{label}] готово за {time.time() - t0:.1f}s.")


def main(argv: Iterable[str] | None = None) -> int:
    cfg = parse_args(argv)

    columns = build_columns(cfg)
    print(f"Методы: {cfg.methods} | колонок: {len(columns)}")

    items = collect_inputs(cfg)
    write_csv(items, cfg.output_csv, cfg, columns, "train")

    if cfg.list_test and cfg.output_csv_test is not None:
        items_test = collect_inputs(cfg, list_file=cfg.list_test)
        write_csv(items_test, cfg.output_csv_test, cfg, columns, "test")

    if cfg.analyze:
        import subprocess
        cmd = [sys.executable, "analyze_correlation.py", str(cfg.output_csv)]
        if cfg.output_csv_test is not None:
            cmd += ["--test-csv", str(cfg.output_csv_test)]
        if cfg.plots:
            cmd.append("--plots-dir")
        if cfg.top is not None:
            cmd += ["--top", str(cfg.top)]
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
