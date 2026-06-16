"""Общий цикл long-batch (chm/count/percentile/voxel) + чейнинг analyze."""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from cloud_pipeline import PreprocessConfig, preprocess_cloud

from core.args import add_common_batch_args, preprocess_config_from_args
from core.io import BatchCfg, InputItem, collect_for, load_done_keys


def simple_error_rows(item: InputItem, msg: str) -> list[dict]:
    """Стандартная строка ошибки: одна запись file+labels+error.

    Используют voxel/percentile/chm. count переопределяет (две строки на source).
    """
    return [{"file": item.rel_path, **item.labels, "error": msg}]


@dataclass
class LongBatchSpec:
    columns: list[str]
    row_key: Callable[[dict], str]
    error_rows: Callable[[InputItem, str], list[dict]]
    compute_rows: Callable[[InputItem, object, set], list[dict]]


def run_long_batch(spec: LongBatchSpec, *, items: list[InputItem],
                   csv_path: Path, resume: bool, preprocess: PreprocessConfig,
                   label: str = "train") -> int:
    print(f"[{label}] файлов на входе: {len(items)} -> {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(csv_path, spec.row_key) if resume else set()
    mode = "a" if (resume and csv_path.exists()) else "w"
    t0 = time.time()
    n_done = n_err = 0
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=spec.columns,
                                extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            f.flush()
        bar = tqdm(items, unit="cloud", dynamic_ncols=True)
        for item in bar:
            bar.set_postfix_str(item.rel_path[-40:], refresh=False)
            if not item.full_path.exists():
                for r in spec.error_rows(item, f"not found: {item.full_path}"):
                    writer.writerow(r)
                f.flush()
                n_err += 1
                continue
            try:
                res = preprocess_cloud(str(item.full_path), preprocess)
            except Exception as e:
                for r in spec.error_rows(item, f"{type(e).__name__}: {e}"):
                    writer.writerow(r)
                f.flush()
                n_err += 1
                continue
            rows = spec.compute_rows(item, res, done_keys)
            for r in rows:
                writer.writerow(r)
                n_done += 1
            f.flush()
    print(f"\nГотово за {time.time() - t0:.1f}s. Строк добавлено: {n_done} "
          f"(ошибок файлов: {n_err}). CSV: {csv_path}")
    return 0


def run_batch_train_test(spec: LongBatchSpec, a, output_csv: Path) -> Path:
    """Общий хвост batch: train-проход + (опц.) test-проход. Возвращает output_csv.

    `a` — распарсенные общие batch-аргументы (add_common_batch_args). Sweep-парсинг
    и автоназвание остаются в методе; сюда приходит уже готовый spec и output_csv.
    """
    cfg = BatchCfg(a.list_file, a.input_dir, Path(a.base_dir), a.limit)
    pre = preprocess_config_from_args(a)
    run_long_batch(spec, items=collect_for(cfg, None), csv_path=output_csv,
                   resume=a.resume, preprocess=pre, label="train")
    if a.list_test:
        test_csv = output_csv.with_name(output_csv.stem + "_test"
                                        + output_csv.suffix)
        run_long_batch(spec, items=collect_for(cfg, a.list_test),
                       csv_path=test_csv, resume=a.resume, preprocess=pre,
                       label="test")
    return output_csv


def chain_analyze(mod, output_csv: Path, argv: Iterable[str] | None) -> None:
    """Запустить analyze метода сразу после batch, если включён --analyze.

    Аргументы для analyze выводятся из общих batch-флагов: путь к train-CSV,
    --test-csv (если был --list-test), --plots-dir (если --plots), --top. Вызывается
    и при прямом запуске метода (python -m methods.<name>), и диспетчером batch.py."""
    p = argparse.ArgumentParser(add_help=False)
    add_common_batch_args(p)
    mod.add_batch_args(p)
    a, _ = p.parse_known_args(argv)
    if not a.analyze:
        return
    an = [str(output_csv)]
    if a.list_test:
        an += ["--test-csv", str(output_csv.with_name(
            output_csv.stem + "_test" + output_csv.suffix))]
    if a.plots:
        an.append("--plots-dir")
    if a.top is not None:
        an += ["--top", str(a.top)]
    print(f"\n>>> analyze {getattr(mod, 'NAME', '?')}: {' '.join(an)}")
    mod.run_analyze(an)


def standard_main(module, argv=None) -> int:
    """Единый main: batch метода, затем chain_analyze. Для всех методов одинаков."""
    csv_path = module.run_batch(argv)
    chain_analyze(module, csv_path, argv)
    return 0
