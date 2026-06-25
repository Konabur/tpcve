#!/usr/bin/env python3
"""tpcve — batch + analyze + predict + plots (локальный запуск).

Запуск методов (voxel/chm/alpha/percentile/count) по dataset-спискам,
вместе и раздельно по стадиям роста --stage Z31/Z65.

Каждый запуск: batch (признак по облакам) → analyze (регрессия biomass ~ признак).
После — predict-демо + сводные графики.

Требования: uv sync (или pip install -e .).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Полный пайплайн batch + analyze + predict + plots")
    p.add_argument("--stages", default=None,
                   help="Стадии через запятую: combined, Z31, Z65 — каждая запускается отдельно (по умолчанию combined,Z31,Z65)")
    p.add_argument("--workers", type=int, default=2,
                   help="Число воркеров для batch.py (default: 2)")
    return p.parse_args(argv)


def _setup() -> tuple[Path, Path, Path, Path]:
    REPO_DIR = Path.cwd().resolve()
    print("REPO_DIR:", REPO_DIR)

    try:
        import tpcve  # noqa: F401
    except ImportError:
        print("tpcve не импортируется — uv sync или pip install -e .")
        raise SystemExit(1)

    DATA = Path(os.getenv("TPCVE_DATA_DIR",
                          "datasets/yanco-2019-wheat-pcd/data/Yanco_TC_2019_HI-pcd"))
    TRAIN_LIST = DATA.parent / "train_list.txt"
    TEST_LIST = DATA.parent / "test_list.txt"

    for p in (TRAIN_LIST, TEST_LIST):
        assert p.exists(), f"нет файла: {p}"

    return REPO_DIR, DATA, TRAIN_LIST, TEST_LIST


def _run_batch(method, stage, train_list, test_list, data, repo_dir, workers, *method_flags, **kwargs):
    """Запустить batch.py для одного метода/стадии (train+test, analyze в цепочке)."""
    cmd = [
        sys.executable, "batch.py",
        "--method", method,
        "--list", str(train_list),
        "--list-test", str(test_list),
        "--base-dir", str(data),
        "--workers", str(workers),
        *method_flags,
    ]
    if stage is not None:
        cmd.extend(["--stage", stage])
    print("\n$ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=repo_dir, **kwargs)
    if p.returncode:
        raise SystemExit(f"batch завершился с кодом {p.returncode}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.stages is None:
        stages = [None, "Z31", "Z65"]
    else:
        stages = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
        stages = [None if s == "COMBINED" else s for s in stages]

    repo_dir, data, train_list, test_list = _setup()
    results_dir = repo_dir / "results"

    def r(m, s, *f):
        _run_batch(m, s, train_list, test_list, data, repo_dir, args.workers, *f)

    # 4. Воксельный метод
    voxel_cfg = "--voxel-sizes", "10,15,20,25,30,35,40,45"
    for s in stages:
        r("voxel", s, *voxel_cfg)

    # 5. CHM (объём по сетке высот)
    chm_cfg = "--cell-sizes", "10,15,20,25", "--percentiles", "1,5,10,50,75,95,99"
    for s in stages:
        r("chm", s, *chm_cfg)

    # 6. Послойная альфа-форма
    alpha_cfg = (
        "--voxel-sizes", "10,20,30,40,50",
        "--alphas", "20,30,40",
        "--layer-dz", "10,20,30,40,50,60",
        "--no-inner-progress",
    )
    for s in stages:
        r("alpha", s, *alpha_cfg)

    # 7. Бейзлайны
    for s in stages:
        r("count", s)
    for s in stages:
        r("percentile", s, "--percentiles", "1,5,10,50,75,95,99")

    # 8. Результаты
    print("\nregression CSV:")
    for p in sorted(results_dir.glob("regression_csv/**/*.csv")):
        print(" ", p.relative_to(results_dir))
    print("\nsummary plots:")
    for p in sorted(results_dir.glob("figures/summary/*.png")):
        print(" ", p.relative_to(results_dir))

    # 9. Предсказание биомассы (демо)
    reg = results_dir / "regression_csv"

    def pick_csv(method, stage):
        cands = [p for p in (reg / method).glob("*.csv")
                 if not p.stem.endswith("_test")
                 and (f"_{stage}_" in f"_{p.stem}_" if stage
                      else ("_Z31_" not in p.stem and "_Z65_" not in p.stem))]
        assert len(cands) == 1, f"{method}/{stage}: ожидал 1 CSV, нашёл {len(cands)}: {cands}"
        return cands[0]

    for stage in stages:
        label = stage or "combined"
        print("\n" + "=" * 70)
        print(f"=== predict: stage={label} ===")
        print("=" * 70)
        cmd = [
            sys.executable, "scripts/predict_biomass.py",
            "--list", str(test_list),
            "--base-dir", str(data),
            "--voxel-csv",  str(pick_csv("voxel", stage)),
            "--alpha-csv",  str(pick_csv("alpha", stage)),
            "--chm-csv",    str(pick_csv("chm", stage)),
            "--height-csv", str(pick_csv("percentile", stage)),
            "--count-csv",  str(pick_csv("count", stage)),
        ]
        print("\n$ " + " ".join(cmd) + "\n")
        p = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, end="")
        if p.returncode:
            raise SystemExit(f"predict (stage={stage}) завершился с кодом {p.returncode}")

    # 10. Сводные графики
    summary_dir = results_dir / "figures" / "summary"
    stages_str = ",".join(s or "combined" for s in stages)
    subprocess.run([
        sys.executable, "scripts/plot_summary.py",
        "--results-dir", str(results_dir),
        "--output-dir", str(summary_dir),
        "--stages", stages_str,
    ], cwd=repo_dir, check=True)

    for s in [s or "combined" for s in stages]:
        png = summary_dir / f"r2_stage_{s}.png"
        if png.exists():
            print(f"\nStage: {s} -> {png}")

    best_dir = summary_dir / "best_fits"
    if best_dir.is_dir():
        for png in sorted(best_dir.glob("*.png")):
            print(f"  {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
