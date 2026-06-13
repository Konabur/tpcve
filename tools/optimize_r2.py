"""Bayesian-оптимизатор R² для batch_process / batch_alpha / batch_chm.

Для выбранного метода ищет натуральные параметры в заданных пользователем
границах через Optuna TPE и максимизирует max(linear_r2, power_r2, huber_r2)
по выборке (biomass ~ V) на --list-датасете.

Поиск sample-efficient (TPE, Bergstra et al. 2011; Akiba et al., KDD 2019),
seed фиксируется, study персистится в SQLite — прогон полностью воспроизводим
и поддерживает resume.

Примеры:
    uv run python tools/optimize_r2.py --method voxel --list data/train.txt \\
        --voxel-size 3 30 --n-trials 50

    uv run python tools/optimize_r2.py --method alpha --list data/train.txt \\
        --voxel-size 3 15 --alpha 5 50 --layer-dz 10 100 --n-trials 80

    uv run python tools/optimize_r2.py --method chm --list data/train.txt \\
        --cell-size 10 200 --percentile 50 99 --n-trials 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# tools/ — это пакет, но скрипт может запускаться напрямую → подложим repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.regression import fit_all  # noqa: E402


METHODS = ("voxel", "alpha", "chm")

# Baseline-параметры для первого trial — берутся из текущих дефолтов проекта,
# чтобы в логе был честный "до/после". Если значение вне переданных границ —
# зажимается к ближайшей границе (см. _clamp_to_bounds).
BASELINE_DEFAULTS: dict[str, dict[str, int]] = {
    "voxel": {"voxel_size": 10},
    "alpha": {"voxel_size": 5, "alpha": 20, "layer_dz": 30},
    "chm":   {"cell_size": 50, "percentile": 95},
}

REQUIRED_BOUNDS: dict[str, tuple[str, ...]] = {
    "voxel": ("voxel_size",),
    "alpha": ("voxel_size", "alpha", "layer_dz"),
    "chm":   ("cell_size", "percentile"),
}


# ---------- evaluation per method ----------

def _max_r2(x: np.ndarray, y: np.ndarray) -> float:
    """max R² по трём моделям (linear/power/huber); -inf если фит невозможен."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() < 3:
        return float("-inf")
    res = fit_all(x[mask], y[mask])
    if res is None:
        return float("-inf")
    r2s = [f["r2"] for f in res["all"].values() if np.isfinite(f["r2"])]
    return float(max(r2s)) if r2s else float("-inf")


def _eval_voxel(params: dict, list_file: str, base_dir: str,
                trial_csv: Path, extra_argv: list[str]) -> float:
    from batch import main as batch_main
    argv = ["--method", "voxel", "--list", list_file, "--base-dir", base_dir,
            "--voxel-sizes", str(params["voxel_size"]),
            "--output-csv", str(trial_csv),
            "--no-analyze", "--no-plots", *extra_argv]
    batch_main(argv)
    df = pd.read_csv(trial_csv)
    x = pd.to_numeric(df["V_voxel"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["biomass"], errors="coerce").to_numpy()
    return _max_r2(x, y)


def _eval_alpha(params: dict, list_file: str, base_dir: str,
                trial_csv: Path, extra_argv: list[str]) -> float:
    from batch import main as batch_main
    argv = ["--method", "alpha", "--list", list_file, "--base-dir", base_dir,
            "--voxel-sizes", str(params["voxel_size"]),
            "--alphas", str(params["alpha"]),
            "--layer-dz", str(params["layer_dz"]),
            "--output-csv", str(trial_csv),
            "--no-analyze", "--no-plots", *extra_argv]
    batch_main(argv)
    df = pd.read_csv(trial_csv)
    x = pd.to_numeric(df["V_voxel"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["biomass"], errors="coerce").to_numpy()
    return _max_r2(x, y)


def _eval_chm(params: dict, list_file: str, base_dir: str,
              trial_csv: Path, extra_argv: list[str]) -> float:
    from batch import main as batch_main
    argv = ["--method", "chm", "--list", list_file, "--base-dir", base_dir,
            "--cell-sizes", str(params["cell_size"]),
            "--percentiles", str(params["percentile"]),
            "--output-csv", str(trial_csv),
            "--no-analyze", "--no-plots", *extra_argv]
    batch_main(argv)
    df = pd.read_csv(trial_csv)
    x = pd.to_numeric(df["V_chm"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["biomass"], errors="coerce").to_numpy()
    return _max_r2(x, y)


EVALUATORS: dict[str, Callable] = {
    "voxel": _eval_voxel,
    "alpha": _eval_alpha,
    "chm":   _eval_chm,
}


# ---------- optuna glue ----------

def _suggest(trial, method: str, bounds: dict[str, tuple[int, int]]) -> dict:
    return {name: trial.suggest_int(name, lo, hi)
            for name, (lo, hi) in bounds.items()}


def _clamp_to_bounds(values: dict, bounds: dict[str, tuple[int, int]]) -> dict:
    return {k: max(bounds[k][0], min(bounds[k][1], v)) for k, v in values.items()}


def _parse_bounds(args, method: str, parser) -> dict[str, tuple[int, int]]:
    bounds: dict[str, tuple[int, int]] = {}
    for name in REQUIRED_BOUNDS[method]:
        cli = "--" + name.replace("_", "-")
        v = getattr(args, name)
        if v is None:
            parser.error(f"{cli} обязателен для --method {method}")
        lo, hi = v
        if lo >= hi:
            parser.error(f"{cli}: LO ({lo}) должен быть < HI ({hi})")
        if lo <= 0:
            parser.error(f"{cli}: натуральные числа, LO должно быть > 0")
        bounds[name] = (lo, hi)
    return bounds


def _save_visualizations(study, out_root: Path) -> None:
    """Optuna built-in matplotlib backend — не требует kaleido/plotly export."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as ovm

    n_trials = len([t for t in study.trials
                    if t.state.name == "COMPLETE"
                    and t.value is not None
                    and np.isfinite(t.value)])
    n_params = len(study.best_params) if study.best_trial else 0

    def _save(fig_factory, fname: str) -> None:
        try:
            fig_factory()
            plt.tight_layout()
            plt.savefig(out_root / fname, dpi=130)
        except Exception as e:
            print(f"[viz] {fname} пропущен: {type(e).__name__}: {e}",
                  file=sys.stderr)
        finally:
            plt.close("all")

    _save(lambda: ovm.plot_optimization_history(study), "convergence.png")
    if n_trials >= 5 and n_params >= 2:
        _save(lambda: ovm.plot_param_importances(study), "param_importance.png")
    if n_params >= 1:
        _save(lambda: ovm.plot_slice(study), "slice.png")


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.env_file and os.path.exists(pre_args.env_file):
        load_dotenv(pre_args.env_file, override=True)
    elif os.path.exists(".env"):
        load_dotenv(".env", override=True)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env-file", default=None,
                   help="Путь к .env (default: .env в cwd)")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--list", dest="list_file", required=True,
                   help="Список 'path biomass c3 c4 c5' (как у batch_*)")
    p.add_argument("--base-dir",
                   default=os.getenv("TPCVE_BASE_DIR", "data"),
                   help="База для путей из --list "
                        "(env: TPCVE_BASE_DIR, default: data)")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--timeout", type=float, default=None,
                   help="Лимит времени всей оптимизации в секундах")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed TPE-сэмплера (default: 42)")
    p.add_argument("--study-name", default=None)
    p.add_argument("--output-dir", default=None,
                   help="default: results/optimize/<method>/<study_name>/")
    p.add_argument("--no-baseline", action="store_true",
                   help="Не запускать baseline-trial с дефолтами проекта")
    p.add_argument("--resume", action="store_true",
                   help="Загрузить существующий study.db и продолжить "
                        "(default: каждый запуск — fresh study)")
    p.add_argument("--patience", type=int, default=3,
                   help="Ранняя остановка: остановить, если best R² не вырос "
                        "за последние N trial-ов (default: 3; 0 — выкл)")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="Минимальный прирост R², считающийся улучшением "
                        "(default: 0.001)")

    # границы (натуральные мм / %)
    p.add_argument("--voxel-size", type=int, nargs=2, metavar=("LO", "HI"),
                   help="мм; используется в --method voxel|alpha")
    p.add_argument("--alpha", type=int, nargs=2, metavar=("LO", "HI"),
                   help="безразмерно; для --method alpha")
    p.add_argument("--layer-dz", type=int, nargs=2, metavar=("LO", "HI"),
                   help="мм; для --method alpha (2D layered)")
    p.add_argument("--cell-size", type=int, nargs=2, metavar=("LO", "HI"),
                   help="мм; для --method chm")
    p.add_argument("--percentile", type=int, nargs=2, metavar=("LO", "HI"),
                   help="%%; для --method chm")

    # переброс препроцессинга в batch
    p.add_argument("--sor-std-ratio", type=float, default=None)
    p.add_argument("--units", default=None,
                   choices=[None, "auto", "m", "cm", "mm"])

    args = p.parse_args(argv)

    bounds = _parse_bounds(args, args.method, p)

    extra_argv: list[str] = []
    if args.sor_std_ratio is not None:
        extra_argv += ["--sor-std-ratio", str(args.sor_std_ratio)]
    if args.units is not None:
        extra_argv += ["--units", args.units]

    study_name = args.study_name or (
        f"{args.method}_"
        + "_".join(f"{k}{lo}-{hi}" for k, (lo, hi) in bounds.items())
    )
    out_root = Path(args.output_dir
                    or f"results/optimize/{args.method}/{study_name}")
    out_root.mkdir(parents=True, exist_ok=True)
    trials_dir = out_root / "trials"
    trials_dir.mkdir(exist_ok=True)
    db_path = out_root / "study.db"
    if not args.resume and db_path.exists():
        db_path.unlink()
        print(f"[fresh] удалён прежний {db_path} (используй --resume чтобы "
              f"продолжить)")
    storage_url = f"sqlite:///{db_path.as_posix()}"

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    if not args.no_baseline and len(study.trials) == 0:
        baseline = _clamp_to_bounds(BASELINE_DEFAULTS[args.method], bounds)
        study.enqueue_trial(baseline)
        print(f"Baseline (enqueued): {baseline}")

    evaluator = EVALUATORS[args.method]
    # кеш дубликатов: TPE на маленьких дискретных пространствах часто
    # повторяет одну и ту же точку — не пересчитываем batch.
    cache: dict[tuple, float] = {}
    for t in study.trials:
        if t.value is not None and np.isfinite(t.value):
            cache[tuple(sorted(t.params.items()))] = t.value

    def objective(trial) -> float:
        params = _suggest(trial, args.method, bounds)
        key = tuple(sorted(params.items()))
        if key in cache:
            r2 = cache[key]
            trial.set_user_attr("duplicate", True)
            trial.set_user_attr("r2_max", r2)
            print(f"[trial {trial.number:>3}] {params} -> R²={r2:.4f} (cached)")
            return r2
        trial_csv = trials_dir / f"trial_{trial.number:04d}.csv"
        try:
            r2 = evaluator(params, args.list_file, args.base_dir,
                          trial_csv, extra_argv)
        except Exception as e:
            print(f"[trial {trial.number}] FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return float("-inf")
        cache[key] = r2
        for k, v in params.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("r2_max", r2)
        print(f"[trial {trial.number:>3}] {params} -> R²={r2:.4f}")
        return r2

    print(f"\n=== Optuna study: {study_name} ===")
    print(f"storage: {storage_url}")
    print(f"bounds : {bounds}")
    print(f"n_trials={args.n_trials} timeout={args.timeout} seed={args.seed} "
          f"patience={args.patience} min_delta={args.min_delta}\n")

    callbacks = []
    if args.patience and args.patience > 0:
        # patience считается ТОЛЬКО по уникальным точкам — дубликаты TPE
        # на дискретной сетке не должны "съедать" терпение.
        state = {"best": float("-inf"), "best_uniq": -1, "uniq": 0}

        def early_stop(st, trial):
            v = trial.value
            if v is None or not np.isfinite(v):
                return
            if trial.user_attrs.get("duplicate"):
                return
            state["uniq"] += 1
            if v > state["best"] + args.min_delta:
                state["best"] = v
                state["best_uniq"] = state["uniq"]
                return
            stagnant = state["uniq"] - state["best_uniq"]
            if state["best_uniq"] >= 0 and stagnant >= args.patience:
                print(f"\n[early stop] нет улучшения >{args.min_delta} "
                      f"за {stagnant} уникальных trial-ов "
                      f"(best={state['best']:.4f}); останавливаюсь.")
                st.stop()
        callbacks.append(early_stop)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   show_progress_bar=False, callbacks=callbacks)

    if study.best_trial is None:
        print("Нет успешных trial-ов.", file=sys.stderr)
        return 1

    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"Лучший R² = {best.value:.4f} на trial #{best.number}")
    print(f"Параметры : {best.params}")
    print("=" * 70)

    # Артефакты
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_root / "trials.csv", index=False)
    with open(out_root / "best.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": args.method,
            "best_r2": best.value,
            "best_params": best.params,
            "best_trial_number": best.number,
            "n_trials_total": len(study.trials),
            "seed": args.seed,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "bounds": {k: list(v) for k, v in bounds.items()},
            "list_file": args.list_file,
        }, f, indent=2, ensure_ascii=False)

    _save_visualizations(study, out_root)

    print(f"\nАртефакты: {out_root}/")
    print(f"  - study.db          (resume + воспроизводимость)")
    print(f"  - trials.csv        (полный лог trial-ов)")
    print(f"  - best.json         (лучшие параметры + R²)")
    print(f"  - convergence.png   (кривая сходимости)")
    print(f"  - param_importance.png (fANOVA, если ≥2 параметров)")
    print(f"  - slice.png         (slice-plot по каждому параметру)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
