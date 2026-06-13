"""Единый analyze-этап: регрессия biomass ~ признак по CSV из batch.py.

    uv run python analyze.py --method alpha --input-csv results/volume_csv/alpha/x.csv
    uv run python analyze.py --method chm        # автопоиск свежайшего CSV
    uv run python analyze.py --method voxel,chm --input-csv a.csv,b.csv

Метод-специфичные analyze-флаги (напр. alpha --source) — в `python -m methods.<m> --help`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from methods import METHODS, load

# NAME -> подпапка results/volume_csv для автопоиска (percentile живёт в height/)
_VOLUME_SUBDIR = {"voxel": "voxel", "alpha": "alpha", "chm": "chm",
                  "count": "count", "percentile": "height"}


def _latest_csv(name: str) -> Path:
    d = Path("results") / "volume_csv" / _VOLUME_SUBDIR[name]
    cands = [c for c in d.glob("*.csv") if not c.stem.endswith("_test")]
    if not cands:
        raise SystemExit(f"Нет CSV в {d} — укажи --input-csv")
    return max(cands, key=lambda p: p.stat().st_mtime)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pre.add_argument("--method", help=f"CSV-список из {sorted(METHODS)}")
    pre.add_argument("--input-csv", default=None,
                     help="CSV-список путей в порядке методов; "
                          "иначе автопоиск свежайшего")
    pre_args, rest = pre.parse_known_args(argv)  # --help здесь покажет общую справку
    if not pre_args.method:
        pre.error(f"--method обязателен. Доступно: {sorted(METHODS)}")
    names = [m.strip() for m in pre_args.method.split(",") if m.strip()]
    bad = [m for m in names if m not in METHODS]
    if bad:
        pre.error(f"Неизвестные методы: {bad}. Доступно: {sorted(METHODS)}")
    inputs = ([p.strip() for p in pre_args.input_csv.split(",")]
              if pre_args.input_csv else [None] * len(names))
    if len(inputs) != len(names):
        pre.error("--input-csv должен содержать столько путей, сколько методов")

    for name, csv in zip(names, inputs):
        mod = load(name)
        path = Path(csv) if csv else _latest_csv(name)
        print(f"\n=== analyze: {name} <- {path} ===")
        mod.run_analyze([str(path), *rest])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
