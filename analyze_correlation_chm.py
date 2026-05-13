"""Регрессия biomass ~ V_chm для long-формата CSV из batch_chm.py.

Группирует строки по (cell_size_mm, percentile). Для каждой группы фитит
linear / power / huber и выбирает лучшую модель по R².

Использование:
    uv run python analyze_correlation_chm.py results/volume_csv/chm/<name>.csv
    uv run python analyze_correlation_chm.py results/volume_csv/chm/<name>.csv \\
        --plots-dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.autoname import default_path
from tools.regression import (bootstrap_test_r2_ci, compute_metrics, fit_all,
                              flatten_for_csv, plot_fits)

GROUP_COLS = ["cell_size_mm", "percentile"]


def label_for(row) -> str:
    return f"c{float(row['cell_size_mm']):g}_p{float(row['percentile']):g}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="CSV из batch_chm.py")
    p.add_argument("--test-csv", default=None,
                   help="Опциональный CSV с тестовой выборкой.")
    p.add_argument("--target", default="biomass")
    p.add_argument("--output", default=None,
                   help="Куда сохранить CSV с результатами регрессии "
                        "(default: results/regression_csv/chm/<stem>_regression.csv)")
    p.add_argument("--plots-dir", nargs="?", const="__auto__", default=None,
                   help="Если указано — сохранить scatter+линии для каждой группы. "
                        "Без значения: results/regression_plots/chm/<stem>/")
    p.add_argument("--top", type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Нет колонки {args.target!r} в CSV")
    if "V_chm" not in df.columns:
        raise SystemExit("Нет колонки V_chm — это не batch_chm CSV?")

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    df_test = None
    test_groups: dict[tuple, "pd.DataFrame"] = {}
    if args.test_csv:
        df_test = pd.read_csv(args.test_csv)
        df_test[args.target] = pd.to_numeric(df_test[args.target],
                                            errors="coerce")
        for keys, grp in df_test.groupby(GROUP_COLS, dropna=False):
            test_groups[tuple(keys)] = grp

    rows = []
    fit_cache: dict[str, dict] = {}
    for keys, grp in df.groupby(GROUP_COLS, dropna=False):
        meta = dict(zip(GROUP_COLS, keys))
        s = pd.to_numeric(grp["V_chm"], errors="coerce")
        y = pd.to_numeric(grp[args.target], errors="coerce")
        mask = s.notna() & y.notna() & (s > 0)
        if mask.sum() < 3:
            continue
        x_arr = s[mask].to_numpy()
        y_arr = y[mask].to_numpy()
        result = fit_all(x_arr, y_arr)
        if result is None:
            continue
        method = label_for(meta)
        fit_cache[method] = result
        rows.append({
            "method": method,
            "source": "V_chm",
            **meta,
            **flatten_for_csv(result),
        })

    if not rows:
        raise SystemExit("Нет валидных данных для регрессии")

    res_df = pd.DataFrame(rows)
    res_df["_max_r2"] = res_df[["linear_r2", "power_r2", "huber_r2"]].max(axis=1)
    res_df = (res_df.sort_values("_max_r2", ascending=False)
              .drop(columns="_max_r2")
              .reset_index(drop=True))
    if args.top:
        res_df = res_df.head(args.top)

    print(f"\nЦель: {args.target} ~ V_chm   (групп: {len(res_df)})")
    print("=" * 140)
    show_cols = ["method", "source", "best_model",
                 "linear_r2", "linear_rmse_pct",
                 "power_r2",  "power_rmse_pct",  "power_b",
                 "huber_r2",  "huber_rmse_pct"]
    print(res_df[show_cols].to_string(index=False,
                                      float_format=lambda v: f"{v:.4g}"))
    print("=" * 140)

    stem = Path(args.csv).stem
    if args.output:
        out_csv = Path(args.output)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = default_path("regression_chm", stem + "_regression", ".csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")

    test_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if df_test is not None:
        test_rows = []
        for r in res_df.itertuples():
            result = fit_cache.get(r.method)
            if result is None:
                continue
            grp = test_groups.get((r.cell_size_mm, r.percentile))
            if grp is None or "V_chm" not in grp.columns:
                continue
            s = pd.to_numeric(grp["V_chm"], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            if mask.sum() == 0:
                continue
            xv = s[mask].to_numpy()
            yv = y[mask].to_numpy()
            test_xy[r.method] = (xv, yv)
            row = {"method": r.method, "source": "V_chm",
                   "cell_size_mm": r.cell_size_mm,
                   "percentile": r.percentile,
                   "n_test": int(mask.sum())}
            for name, f in result["all"].items():
                m = compute_metrics(yv, f["predict"](xv))
                row[f"{name}_test_r2"] = m["r2"]
                row[f"{name}_test_rmse"] = m["rmse"]
                row[f"{name}_test_rmse_pct"] = m["rmse_pct"]
                row[f"{name}_test_bias"] = m["bias"]
                ci_lo, ci_hi = bootstrap_test_r2_ci(f["predict"], xv, yv)
                row[f"{name}_test_r2_ci_lo"] = ci_lo
                row[f"{name}_test_r2_ci_hi"] = ci_hi
            test_rows.append(row)
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            print(f"\nTest (групп с test: {len(test_df)}):")
            print("=" * 140)
            disp = test_df[["method", "source", "n_test"]].copy()
            for name in ("linear", "power", "huber"):
                disp[f"{name}_R2[CI95]"] = [
                    f"{a:.3f}[{b:.2f},{c:.2f}]" for a, b, c in zip(
                        test_df[f"{name}_test_r2"],
                        test_df[f"{name}_test_r2_ci_lo"],
                        test_df[f"{name}_test_r2_ci_hi"])]
                disp[f"{name}_RMSE%"] = test_df[f"{name}_test_rmse_pct"].round(1)
            print(disp.to_string(index=False))
            print("=" * 140)
            test_out = out_csv.with_name(out_csv.stem + "_test" + out_csv.suffix)
            test_df.to_csv(test_out, index=False)
            print(f"Test regression: {test_out}")

    if args.plots_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.plots_dir == "__auto__":
            out = default_path("regression_plots_chm", stem, ext="")
        else:
            out = Path(args.plots_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in res_df.itertuples():
            result = fit_cache.get(r.method)
            if result is None:
                continue
            grp = df[(df["cell_size_mm"] == r.cell_size_mm)
                     & (df["percentile"] == r.percentile)]
            s = pd.to_numeric(grp["V_chm"], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            x = s[mask].to_numpy()
            yv = y[mask].to_numpy()

            xv_test, yv_test = test_xy.get(r.method, (None, None))
            fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))
            plot_fits(ax, x, yv, result,
                      xlabel="V_chm (м³)",
                      ylabel=args.target,
                      title=f"{args.target} ~ {r.method}",
                      x_test=xv_test, y_test=yv_test)
            best = result["best"]
            y_pred_tr = best["predict"](x)
            ax_r.scatter(y_pred_tr, yv - y_pred_tr, s=20, alpha=0.6,
                         color="#444", label=f"train (n={len(x)})")
            if xv_test is not None and len(xv_test) > 0:
                y_pred_te = best["predict"](xv_test)
                ax_r.scatter(y_pred_te, yv_test - y_pred_te, s=28, alpha=0.85,
                             color="#d62728", marker="^",
                             label=f"test (n={len(xv_test)})")
            ax_r.axhline(0, color="k", lw=0.8, ls="--")
            ax_r.set_xlabel(f"fitted {args.target}")
            ax_r.set_ylabel("residual (y − ŷ)")
            ax_r.set_title(f"Residuals [{result['best_model']}]")
            ax_r.legend(loc="best", fontsize=8)
            ax_r.grid(alpha=0.3)
            fig.tight_layout()
            fname = f"{r.method}.png".replace("/", "_")
            fig.savefig(out / fname, dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
