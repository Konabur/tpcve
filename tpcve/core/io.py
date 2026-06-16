"""Сбор и парсинг входных данных: --list / --input-dir, выбор облака из списка."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

LABEL_COLS = ["biomass", "col3", "col4", "col5"]

# Расширения облаков — по ним отделяем путь (может содержать пробелы) от меток.
CLOUD_EXTS = (".npz", ".las", ".laz", ".pcd", ".ply", ".xyz", ".pts", ".db3")


@dataclass
class InputItem:
    rel_path: str
    full_path: Path
    labels: dict


@dataclass
class BatchCfg:
    """Минимальный конфиг для collect_inputs/collect_for (заменяет type('Cfg', …))."""
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    limit: int | None = None
    stage: str | None = None


def parse_list_line(line: str) -> tuple[str, dict]:
    """`<path> <biomass> [<c3> <c4> <c5>]` — путь может содержать пробелы.

    Путь отделяется от меток по расширению облака (`CLOUD_EXTS`): всё до токена с
    расширением включительно — путь, остальное — метки. `biomass` обязателен;
    `col3..col5` опциональны (отсутствующие заполняются "").
    """
    parts = line.strip().split()
    ext_idx = next((i for i, t in enumerate(parts)
                    if t.lower().endswith(CLOUD_EXTS)), None)
    if ext_idx is None:
        raise ValueError(
            f"Не найден путь к облаку ({'/'.join(CLOUD_EXTS)}): {line!r}")
    rel_path = " ".join(parts[:ext_idx + 1])
    labels = parts[ext_idx + 1:]
    if not labels:
        raise ValueError(f"Отсутствует biomass после пути: {line!r}")
    biomass, *extra = labels
    extra = (extra + ["", "", ""])[:3]
    return rel_path, {"biomass": biomass, "col3": extra[0],
                      "col4": extra[1], "col5": extra[2]}


def collect_inputs(cfg, *, list_file: str | None = None) -> list[InputItem]:
    """list_file override позволяет переиспользовать конфиг для test-прохода."""
    items: list[InputItem] = []
    stage = getattr(cfg, "stage", None)
    src_list = list_file if list_file is not None else cfg.list_file
    if src_list:
        with open(src_list, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                rel, labels = parse_list_line(line)
                if stage is not None and stage_from_path(rel) != stage:
                    continue
                full = cfg.base_dir / rel.lstrip("/\\")
                items.append(InputItem(rel, full, labels))
    elif cfg.input_dir and list_file is None:
        root = Path(cfg.input_dir)
        for f in sorted(root.rglob("*.pcd")):
            rel = str(f.relative_to(root))
            if stage is not None and stage_from_path(rel) != stage:
                continue
            items.append(InputItem(rel, f, {k: "" for k in LABEL_COLS}))
    else:
        raise ValueError("Нужен --list или --input-dir")
    if cfg.limit:
        items = items[: cfg.limit]
    return items


def collect_for(cfg, list_file: str | None) -> list[InputItem]:
    """Обёртка для test-прохода: строит временный cfg-объект."""
    return collect_inputs(BatchCfg(
        list_file=list_file if list_file is not None else cfg.list_file,
        input_dir=None if list_file is not None else cfg.input_dir,
        base_dir=cfg.base_dir, limit=cfg.limit,
        stage=getattr(cfg, "stage", None),
    ))


def load_done_keys(csv_path: Path, key_fn: Callable[[dict], str]) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {key_fn(row) for row in csv.DictReader(f) if row.get("file")}


# ---------------------------------------------------------------------------
# Выбор облака из списка (по стадии роста / медиане биомассы)

# Стадия роста определяется по подстроке в пути облака (дата съёмки).
STAGE_TOKENS = {"Z31": "0828", "Z65": "1002"}


def stage_from_path(path: str) -> str | None:
    for stage, tok in STAGE_TOKENS.items():
        if tok in path:
            return stage
    return None


def pick_median_biomass(list_path: str, base_dir: Path,
                        stage: str | None = None
                        ) -> tuple[Path, float, str | None]:
    """Из --list-файла выбрать облако с медианной биомассой.

    Возвращает (полный путь, биомасса, стадия). При stage != None берутся только
    облака этой стадии. Пустой результат → SystemExit.
    """
    rows: list[tuple[Path, float, str | None]] = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                rel, labels = parse_list_line(line)
                bm = float(labels["biomass"])
            except (ValueError, KeyError):
                continue
            st = stage_from_path(rel)
            if stage is not None and st != stage:
                continue
            rows.append((base_dir / rel.lstrip("/\\"), bm, st))
    if not rows:
        msg = f"В {list_path} не нашлось валидных строк с биомассой"
        if stage is not None:
            msg += f" для стадии {stage} (подстрока '{STAGE_TOKENS[stage]}')"
        raise SystemExit(msg)
    rows.sort(key=lambda r: r[1])
    return rows[len(rows) // 2]
