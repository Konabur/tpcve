"""Единый batch-этап: оценка признаков биомассы по облакам точек.

Предсказание биомассы пшеницы по облаку точек: каждый метод считает признак
(voxel/alpha объём, CHM, число точек, высотный перцентиль) для последующей
регрессии biomass ~ признак. Метод выбирается через --method.

    uv run python batch.py --method alpha --list f.txt --alphas 10,20 --layer-dz 20
    uv run python batch.py --method voxel,chm --input-dir data/...
    uv run python batch.py --method chm --list f.txt --cell-sizes 20,50 --percentiles 95

Метод-специфичные флаги (--alphas, --cell-sizes, --percentiles, --layer-dz, …)
смотри в `batch.py --method <method> --help`.
"""
from __future__ import annotations

import argparse
import sys

from tpcve.methods import METHODS, load
from tpcve import core


def _build_pre_parser() -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)  # help обрабатываем вручную: см. main()
    pre.add_argument("-h", "--help", action="store_true")
    pre.add_argument("--method", help=f"CSV-список из {sorted(METHODS)}")
    return pre


def _print_method_help(name: str) -> None:
    """Полная справка метода: общие batch-флаги + метод-специфичные."""
    mod = load(name)
    p = argparse.ArgumentParser(
        prog=f"batch.py --method {name}",
        description=getattr(mod, "__doc__", None),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    core.add_common_batch_args(p)
    mod.add_batch_args(p)
    mod.add_analyze_args(p)
    p.print_help()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = _build_pre_parser()
    pre_args, rest = pre.parse_known_args(argv)

    if pre_args.help:
        # `--method <m> --help` -> справка метода(ов); без метода -> общая справка
        names = [m.strip() for m in (pre_args.method or "").split(",")
                 if m.strip() and m.strip() in METHODS]
        if names:
            for name in names:
                _print_method_help(name)
                print()
        else:
            pre.print_help()
        return 0

    if not pre_args.method:
        pre.error(f"--method обязателен. Доступно: {sorted(METHODS)}")
    names = [m.strip() for m in pre_args.method.split(",") if m.strip()]
    bad = [m for m in names if m not in METHODS]
    if bad:
        pre.error(f"Неизвестные методы: {bad}. Доступно: {sorted(METHODS)}")

    # rest содержит общие + метод-специфичные флаги; каждый run_batch/run_analyze
    # парсит их через parse_known_args, игнорируя чужие.
    for name in names:
        mod = load(name)
        print(f"\n=== batch: {name} ===")
        csv_path = mod.run_batch(rest)
        core.chain_analyze(mod, csv_path, rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
