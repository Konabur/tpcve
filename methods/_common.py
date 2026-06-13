"""Общий код batch/analyze: io, цикл long-batch, общий long-analyze."""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from dotenv import load_dotenv
from tqdm import tqdm

from cloud_pipeline import PreprocessConfig, preprocess_cloud

LABEL_COLS = ["biomass", "col3", "col4", "col5"]


@dataclass
class InputItem:
    rel_path: str
    full_path: Path
    labels: dict


def parse_list_line(line: str) -> tuple[str, dict]:
    """`<path> <biomass> <c3> <c4> <c5>` — путь может содержать пробелы."""
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Ожидалось >=5 токенов, получено {len(parts)}: {line!r}")
    *path_parts, biomass, c3, c4, c5 = parts
    rel_path = " ".join(path_parts)
    return rel_path, {"biomass": biomass, "col3": c3, "col4": c4, "col5": c5}


def collect_inputs(cfg, *, list_file: str | None = None) -> list[InputItem]:
    """list_file override позволяет переиспользовать конфиг для test-прохода."""
    items: list[InputItem] = []
    src_list = list_file if list_file is not None else cfg.list_file
    if src_list:
        with open(src_list, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                rel, labels = parse_list_line(line)
                full = cfg.base_dir / rel.lstrip("/\\")
                items.append(InputItem(rel, full, labels))
    elif cfg.input_dir and list_file is None:
        root = Path(cfg.input_dir)
        for f in sorted(root.rglob("*.pcd")):
            rel = str(f.relative_to(root))
            items.append(InputItem(rel, f, {k: "" for k in LABEL_COLS}))
    else:
        raise ValueError("Нужен --list или --input-dir")
    if cfg.limit:
        items = items[: cfg.limit]
    return items


def collect_for(cfg, list_file: str | None) -> list[InputItem]:
    """Обёртка для test-прохода: строит временный cfg-объект (как старый _collect)."""
    items_cfg = type("X", (), {
        "list_file": list_file if list_file is not None else cfg.list_file,
        "input_dir": None if list_file is not None else cfg.input_dir,
        "base_dir": cfg.base_dir, "limit": cfg.limit,
    })()
    return collect_inputs(items_cfg)


def load_done_files(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {row["file"] for row in csv.DictReader(f) if row.get("file")}


def load_done_keys(csv_path: Path, key_fn: Callable[[dict], str]) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {key_fn(row) for row in csv.DictReader(f) if row.get("file")}
