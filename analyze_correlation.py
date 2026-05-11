"""Регрессия biomass ~ volume для каждой колонки-метода в CSV.

Фитит linear / power / huber и для каждого метода выбирает лучшую по R².

Использование:
    uv run python analyze_correlation.py results/batch.csv
    uv run python analyze_correlation.py results/batch.csv --plots-dir results/regression
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.autoname import default_path
from tools.regression import compute_metrics, fit_all, flatten_for_csv, plot_fits

NON_METHOD_COLS = {"file", "biomass", "col3", "col4", "col5",
                   "n_input", "n_after_sor", "n_vegetation", "error"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="CSV из batch_process.py")
    p.add_argument("--test-csv", default=None,
                   help="CSV с тестовой выборкой (та же структура, "
                        "что у train); для каждой модели считается R²/RMSE/"
                        "bias на test и сохраняется в *_regression_test.csv.")
    p.add_argument("--output", default=None,
                   help="Куда сохранить CSV с результатами регрессии "
                        "(default: results/regression_csv/voxel/<stem>_regression.csv)")
    p.add_argument("--plots-dir", nargs="?", const="__auto__", default=None,
                   help="Если указано — сохранить scatter+линии для каждого метода. "
                        "Без значения: results/regression_plots/voxel/<stem>/")
    p.add_argument("--target", default="biomass",
                   help="Целевая колонка (default: biomass)")
    p.add_argument("--top", type=int, default=None,
                   help="Показать только top-N методов по R²")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Нет колонки {args.target!r} в CSV")

    method_cols = [c for c in df.columns if c not in NON_METHOD_COLS]
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    df_test = None
    if args.test_csv:
        df_test = pd.read_csv(args.test_csv)
        if args.target not in df_test.columns:
            raise SystemExit(
                f"Нет колонки {args.target!r} в test CSV")
        df_test[args.target] = pd.to_numeric(df_test[args.target],
                                            errors="coerce")

    rows = []
    fit_cache: dict[str, dict] = {}
    for col in method_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        mask = s.notna() & df[args.target].notna() & (s > 0)
        if mask.sum() < 3:
            continue
        x = s[mask].to_numpy()
        y = df.loc[mask, args.target].to_numpy()
        result = fit_all(x, y)
        if result is None:
            continue
        fit_cache[col] = result
        rows.append({"method": col, **flatten_for_csv(result)})

    if not rows:
        raise SystemExit("Нет валидных данных для регрессии")

    res = pd.DataFrame(rows)
    res["_max_r2"] = res[["linear_r2", "power_r2", "huber_r2"]].max(axis=1)
    res = res.sort_values("_max_r2", ascending=False).drop(columns="_max_r2").reset_index(drop=True)
    if args.top:
        res = res.head(args.top)

    print(f"\nЦель: {args.target} ~ volume   (всего методов: {len(res)})")
    print("=" * 130)
    show_cols = ["method", "best_model",
                 "linear_r2", "linear_rmse_pct",
                 "power_r2",  "power_rmse_pct",  "power_b",
                 "huber_r2",  "huber_rmse_pct"]
    print(res[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("=" * 130)

    stem = Path(args.csv).stem
    if args.output:
        out_csv = Path(args.output)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = default_path("regression_voxel", stem + "_regression", ".csv")
    res.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")

    test_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if df_test is not None:
        test_rows = []
        for col in res["method"]:
            result = fit_cache.get(col)
            if result is None or col not in df_test.columns:
                continue
            s = pd.to_numeric(df_test[col], errors="coerce")
            mask = s.notna() & df_test[args.target].notna() & (s > 0)
            if mask.sum() == 0:
                continue
            xv = s[mask].to_numpy()
            yv = df_test.loc[mask, args.target].to_numpy()
            test_data[col] = (xv, yv)
            row = {"method": col, "n_test": int(mask.sum())}
            for name, f in result["all"].items():
                m = compute_metrics(yv, f["predict"](xv))
                row[f"{name}_test_r2"] = m["r2"]
                row[f"{name}_test_rmse"] = m["rmse"]
                row[f"{name}_test_rmse_pct"] = m["rmse_pct"]
                row[f"{name}_test_bias"] = m["bias"]
            test_rows.append(row)
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            print(f"\nTest (n_test до {test_df['n_test'].max()}):")
            print("=" * 130)
            show = ["method", "n_test",
                    "linear_test_r2", "linear_test_rmse_pct", "linear_test_bias",
                    "power_test_r2",  "power_test_rmse_pct",  "power_test_bias",
                    "huber_test_r2",  "huber_test_rmse_pct",  "huber_test_bias"]
            print(test_df[show].to_string(index=False,
                                         float_format=lambda v: f"{v:.4g}"))
            print("=" * 130)
            test_out = out_csv.with_name(out_csv.stem + "_test" + out_csv.suffix)
            test_df.to_csv(test_out, index=False)
            print(f"Test regression: {test_out}")

    if args.plots_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.plots_dir == "__auto__":
            out = default_path("regression_plots_voxel", stem, ext="")
        else:
            out = Path(args.plots_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in res.itertuples():
            col = r.method
            result = fit_cache.get(col)
            if result is None:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            mask = s.notna() & df[args.target].notna() & (s > 0)
            x = s[mask].to_numpy()
            y = df.loc[mask, args.target].to_numpy()

            xv, yv = test_data.get(col, (None, None))
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            plot_fits(ax, x, y, result,
                      xlabel=f"{col} (м³)",
                      ylabel=args.target,
                      title=f"{args.target} ~ {col}",
                      x_test=xv, y_test=yv)
            fig.tight_layout()
            fig.savefig(out / f"{col}.png", dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
