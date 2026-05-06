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
from volume_methods import (
    DEFAULT_ALPHAS,
    DEFAULT_VOXEL_SIZES,
    METHODS,
    method_columns,
    run_method,
)

LABEL_COLS = ["biomass", "col3", "col4", "col5"]


@dataclass
class InputItem:
    rel_path: str
    full_path: Path
    labels: dict  # {biomass, col3, col4, col5}


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
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def parse_list_line(line: str) -> tuple[str, dict]:
    """`<path> <biomass> <c3> <c4> <c5>` — путь может содержать пробелы.

    Берём последние 4 токена как метки, всё перед ними — это путь.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Ожидалось >=5 токенов, получено {len(parts)}: {line!r}")
    *path_parts, biomass, c3, c4, c5 = parts
    rel_path = " ".join(path_parts)
    return rel_path, {
        "biomass": biomass,
        "col3": c3,
        "col4": c4,
        "col5": c5,
    }


def collect_inputs(cfg: BatchConfig) -> list[InputItem]:
    items: list[InputItem] = []

    if cfg.list_file:
        with open(cfg.list_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                rel, labels = parse_list_line(line)
                full = cfg.base_dir / rel.lstrip("/\\")
                items.append(InputItem(rel, full, labels))
    elif cfg.input_dir:
        root = Path(cfg.input_dir)
        for f in sorted(root.rglob("*.pcd")):
            rel = str(f.relative_to(root))
            items.append(InputItem(rel, f, {k: "" for k in LABEL_COLS}))
    else:
        raise ValueError("Нужен --list или --input-dir")

    if cfg.limit:
        items = items[: cfg.limit]
    return items


def build_columns(cfg: BatchConfig) -> list[str]:
    cols = ["file", *LABEL_COLS, "n_input", "n_after_sor", "n_vegetation"]
    for m in cfg.methods:
        cols.extend(method_columns(m, voxel_sizes=cfg.voxel_sizes,
                                   alphas=cfg.alphas))
    cols.append("error")
    return cols


def load_done_files(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["file"] for row in reader if row.get("file")}


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
    p.add_argument("--base-dir", default="data",
                   help="База для путей из --list (default: data)")
    p.add_argument("--output-csv", default="results/batch.csv")
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
    # препроцессинг
    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower()
                   in ("1", "true", "yes"))
    p.add_argument("--downsample", type=float,
                   default=float(os.getenv("TPCVE_DOWNSAMPLE", "0") or 0))
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO", "1.5")))
    p.add_argument("--min-range", type=float,
                   default=float(os.getenv("TPCVE_MIN_RANGE", "0") or 0))
    p.add_argument("--height-threshold", type=float, default=0.04)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        p.error(f"Неизвестные методы: {bad}. Доступно: {sorted(METHODS)}")

    voxel_sizes = (DEFAULT_VOXEL_SIZES if not a.voxel_sizes
                   else [float(x) / 1000 for x in a.voxel_sizes.split(",")])
    alphas = (DEFAULT_ALPHAS if not a.alphas
              else [float(x) for x in a.alphas.split(",")])

    return BatchConfig(
        list_file=a.list_file,
        input_dir=a.input_dir,
        base_dir=Path(a.base_dir),
        output_csv=Path(a.output_csv),
        methods=methods,
        voxel_sizes=voxel_sizes,
        alphas=alphas,
        resume=a.resume,
        limit=a.limit,
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


def main(argv: Iterable[str] | None = None) -> int:
    cfg = parse_args(argv)

    items = collect_inputs(cfg)
    columns = build_columns(cfg)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_files(cfg.output_csv) if cfg.resume else set()
    file_exists = cfg.output_csv.exists()
    mode = "a" if (cfg.resume and file_exists) else "w"

    pending = [it for it in items if it.rel_path not in done]
    print(f"Всего входов: {len(items)}; к обработке: {len(pending)} "
          f"(пропущено по resume: {len(items) - len(pending)})")
    print(f"Методы: {cfg.methods} | колонок: {len(columns)} | CSV: {cfg.output_csv}")

    t0 = time.time()
    with open(cfg.output_csv, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            f.flush()

        bar = tqdm(pending, unit="cloud", dynamic_ncols=True)
        n_err = 0
        for item in bar:
            bar.set_postfix_str(item.rel_path[-40:], refresh=False)
            row = process_one(item, cfg)
            writer.writerow(row)
            f.flush()
            if row.get("error"):
                n_err += 1
                tqdm.write(f"ERR {item.rel_path}: {row['error']}")
            bar.set_postfix(err=n_err, refresh=False)

    print(f"\nГотово за {time.time() - t0:.1f}s. CSV: {cfg.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
