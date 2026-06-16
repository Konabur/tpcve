"""Общий long-analyze: регрессия biomass ~ x по группам строк CSV.

Единая схема выходного regression-CSV: [method, *group_cols, x_col,
<статистики fit_all>].
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def _as_tuple(keys) -> tuple:
    return keys if isinstance(keys, tuple) else (keys,)


def run_long_analyze(args, *, value_cols: list[str], group_cols: list[str],
                     label_fn: Callable[[dict, str], str],
                     subfolder: str,
                     prep_df: Callable | None = None) -> int:
    """Единый analyze для всех методов.

    - `value_cols`: x-колонки (обычно одна; alpha может дать [V_voxel, V_random]).
    - `group_cols`: колонки группировки (свои у каждого метода).
    - `label_fn(meta, vc) -> str`: человекочитаемая метка группы (колонка `method`).
    - `prep_df(df)`: опц. in-place подготовка (напр. alpha: fillna layer_dz_mm).
    """
    import pandas as pd

    from tools.autoname import default_path
    from tools.regression import (bootstrap_test_r2_ci, compute_metrics,
                                   fit_all, flatten_for_csv, plot_fits)

    df = pd.read_csv(args.csv)
    if prep_df is not None:
        prep_df(df)
    if args.target not in df.columns:
        raise SystemExit(f"Нет колонки {args.target!r} в CSV")
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    vcs = [c for c in value_cols if c in df.columns]
    if not vcs:
        raise SystemExit(f"Нет ни одной value-колонки {value_cols} в CSV")

    df_test = None
    test_groups: dict[tuple, "pd.DataFrame"] = {}
    if args.test_csv:
        df_test = pd.read_csv(args.test_csv)
        if prep_df is not None:
            prep_df(df_test)
        df_test[args.target] = pd.to_numeric(df_test[args.target], errors="coerce")
        for keys, grp in df_test.groupby(group_cols, dropna=False):
            test_groups[_as_tuple(keys)] = grp

    rows = []
    fit_cache: dict[tuple, dict] = {}
    meta_cache: dict[tuple, dict] = {}
    for keys, grp in df.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, _as_tuple(keys)))
        for vc in vcs:
            s = pd.to_numeric(grp[vc], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            if mask.sum() < 3:
                continue
            result = fit_all(s[mask].to_numpy(), y[mask].to_numpy())
            if result is None:
                continue
            label = label_fn(meta, vc)
            key = (_as_tuple(keys), vc)
            fit_cache[key] = result
            meta_cache[key] = meta
            rows.append({"method": label, **meta, "x_col": vc,
                         **flatten_for_csv(result)})

    if not rows:
        raise SystemExit("Нет валидных данных для регрессии")

    res_df = pd.DataFrame(rows)
    res_df["_max_r2"] = res_df[["linear_r2", "power_r2", "huber_r2"]].max(axis=1)
    res_df = (res_df.sort_values("_max_r2", ascending=False)
              .drop(columns="_max_r2").reset_index(drop=True))
    if args.top:
        res_df = res_df.head(args.top)

    print(f"\nЦель: {args.target} ~ x   (групп: {len(res_df)})")
    print("=" * 140)
    show_cols = ["method", "x_col", "best_model",
                 "linear_r2", "linear_rmse_pct",
                 "power_r2", "power_rmse_pct", "power_b",
                 "huber_r2", "huber_rmse_pct"]
    print(res_df[show_cols].to_string(index=False,
                                      float_format=lambda v: f"{v:.4g}"))
    print("=" * 140)

    stem = Path(args.csv).stem
    if args.output:
        out_csv = Path(args.output)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = default_path("regression_csv", stem + "_regression",
                               subfolder=subfolder)
    res_df.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")

    test_xy: dict[tuple, tuple] = {}
    if df_test is not None:
        test_rows = []
        for r in res_df.itertuples(index=False):
            meta = {c: getattr(r, c) for c in group_cols}
            vc = r.x_col
            key = (tuple(meta[c] for c in group_cols), vc)
            result = fit_cache.get(key)
            grp = test_groups.get(tuple(meta[c] for c in group_cols))
            if result is None or grp is None or vc not in grp.columns:
                continue
            s = pd.to_numeric(grp[vc], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            if mask.sum() == 0:
                continue
            xv, yv = s[mask].to_numpy(), y[mask].to_numpy()
            test_xy[key] = (xv, yv)
            row = {"method": r.method, **meta, "x_col": vc,
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
            disp = test_df[["method", "x_col", "n_test"]].copy()
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

        out = (default_path("regression_plots", stem, ext="", subfolder=subfolder)
               if args.plots_dir == "__auto__" else Path(args.plots_dir))
        out.mkdir(parents=True, exist_ok=True)
        for r in res_df.itertuples(index=False):
            meta = {c: getattr(r, c) for c in group_cols}
            vc = r.x_col
            key = (tuple(meta[c] for c in group_cols), vc)
            result = fit_cache.get(key)
            if result is None:
                continue
            sub = df
            for c in group_cols:
                sub = sub[sub[c] == meta[c]]
            s = pd.to_numeric(sub[vc], errors="coerce")
            y = pd.to_numeric(sub[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            x, yv = s[mask].to_numpy(), y[mask].to_numpy()
            xv_test, yv_test = test_xy.get(key, (None, None))
            fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))
            plot_fits(ax, x, yv, result, xlabel=str(vc), ylabel=args.target,
                      title=f"{args.target} ~ {r.method} [{vc}]",
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
            fname = f"{r.method}__{vc}.png".replace("/", "_")
            fig.savefig(out / fname, dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0
