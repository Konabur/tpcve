"""Регрессия biomass ~ V для long-формата CSV из batch_alpha.py.

Группирует строки по (voxel_mm, alpha, mode, layer_dz_mm). Для каждой группы
фитит linear / power / huber и выбирает лучшую модель по R².

Использование:
    uv run python analyze_correlation_alpha.py results/batch_alpha.csv
    uv run python analyze_correlation_alpha.py results/batch_alpha.csv \\
        --plots-dir results/regression_alpha
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.autoname import default_path
from tools.regression import compute_metrics, fit_all, flatten_for_csv, plot_fits

GROUP_COLS = ["voxel_mm", "alpha", "mode", "layer_dz_mm"]


def label_for(row) -> str:
    mode = row["mode"]
    suffix = f"_dz{row['layer_dz_mm']}" if mode == "layered" else ""
    return f"v{float(row['voxel_mm']):g}_a{float(row['alpha']):g}_{mode}{suffix}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="CSV из batch_alpha.py")
    p.add_argument("--test-csv", default=None,
                   help="Опциональный CSV с тестовой выборкой.")
    p.add_argument("--target", default="biomass")
    p.add_argument("--output", default=None,
                   help="Куда сохранить CSV с результатами регрессии "
                        "(default: results/regression_csv/alpha/<stem>_regression.csv)")
    p.add_argument("--plots-dir", nargs="?", const="__auto__", default=None,
                   help="Если указано — сохранить scatter+линии для каждой группы. "
                        "Без значения: results/regression_plots/alpha/<stem>/")
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--source", choices=["voxel", "random", "both"],
                   default="voxel",
                   help="По какой колонке V строить регрессию (default: voxel)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Нет колонки {args.target!r} в CSV")
    if "V_voxel" not in df.columns:
        raise SystemExit("Нет колонки V_voxel — это не batch_alpha CSV?")

    # Сделать layer_dz_mm устойчивым к NaN (для группировки)
    df["layer_dz_mm"] = df["layer_dz_mm"].fillna("").astype(str)
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    df_test = None
    test_groups: dict[tuple, "pd.DataFrame"] = {}
    if args.test_csv:
        df_test = pd.read_csv(args.test_csv)
        df_test["layer_dz_mm"] = df_test["layer_dz_mm"].fillna("").astype(str)
        df_test[args.target] = pd.to_numeric(df_test[args.target],
                                            errors="coerce")
        for keys, grp in df_test.groupby(GROUP_COLS, dropna=False):
            test_groups[tuple(keys)] = grp

    sources = (["V_voxel", "V_random"] if args.source == "both"
               else [f"V_{args.source}"])
    sources = [c for c in sources if c in df.columns]

    rows = []
    fit_cache: dict[tuple, dict] = {}
    for keys, grp in df.groupby(GROUP_COLS, dropna=False):
        meta = dict(zip(GROUP_COLS, keys))
        for vc in sources:
            s = pd.to_numeric(grp[vc], errors="coerce")
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
            fit_cache[(method, vc)] = result
            rows.append({
                "method": method,
                "source": vc,
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

    print(f"\nЦель: {args.target} ~ V   (групп: {len(res_df)})")
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
        out_csv = default_path("regression_alpha", stem + "_regression", ".csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")

    test_xy: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    if df_test is not None:
        test_rows = []
        for r in res_df.itertuples():
            result = fit_cache.get((r.method, r.source))
            if result is None:
                continue
            key = (r.voxel_mm, r.alpha, r.mode, r.layer_dz_mm)
            grp = test_groups.get(key)
            if grp is None or r.source not in grp.columns:
                continue
            s = pd.to_numeric(grp[r.source], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            if mask.sum() == 0:
                continue
            xv = s[mask].to_numpy()
            yv = y[mask].to_numpy()
            test_xy[(r.method, r.source)] = (xv, yv)
            row = {"method": r.method, "source": r.source,
                   "voxel_mm": r.voxel_mm, "alpha": r.alpha,
                   "mode": r.mode, "layer_dz_mm": r.layer_dz_mm,
                   "n_test": int(mask.sum())}
            for name, f in result["all"].items():
                m = compute_metrics(yv, f["predict"](xv))
                row[f"{name}_test_r2"] = m["r2"]
                row[f"{name}_test_rmse"] = m["rmse"]
                row[f"{name}_test_rmse_pct"] = m["rmse_pct"]
                row[f"{name}_test_bias"] = m["bias"]
            test_rows.append(row)
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            print(f"\nTest (групп с test: {len(test_df)}):")
            print("=" * 140)
            show = ["method", "source", "n_test",
                    "linear_test_r2", "linear_test_rmse_pct", "linear_test_bias",
                    "power_test_r2",  "power_test_rmse_pct",  "power_test_bias",
                    "huber_test_r2",  "huber_test_rmse_pct",  "huber_test_bias"]
            print(test_df[show].to_string(index=False,
                                         float_format=lambda v: f"{v:.4g}"))
            print("=" * 140)
            test_out = out_csv.with_name(out_csv.stem + "_test" + out_csv.suffix)
            test_df.to_csv(test_out, index=False)
            print(f"Test regression: {test_out}")

    if args.plots_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.plots_dir == "__auto__":
            out = default_path("regression_plots_alpha", stem, ext="")
        else:
            out = Path(args.plots_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in res_df.itertuples():
            result = fit_cache.get((r.method, r.source))
            if result is None:
                continue
            grp = df[(df["voxel_mm"] == r.voxel_mm)
                     & (df["alpha"] == r.alpha)
                     & (df["mode"] == r.mode)
                     & (df["layer_dz_mm"] == r.layer_dz_mm)]
            s = pd.to_numeric(grp[r.source], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            x = s[mask].to_numpy()
            yv = y[mask].to_numpy()

            xv_test, yv_test = test_xy.get((r.method, r.source), (None, None))
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            plot_fits(ax, x, yv, result,
                      xlabel=f"{r.source} (м³)",
                      ylabel=args.target,
                      title=f"{args.target} ~ {r.method} ({r.source})",
                      x_test=xv_test, y_test=yv_test)
            fig.tight_layout()
            fname = f"{r.method}__{r.source}.png".replace("/", "_")
            fig.savefig(out / fname, dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
