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
from tools.regression import fit_all, flatten_for_csv, plot_fits

GROUP_COLS = ["cell_size_mm", "percentile"]


def label_for(row) -> str:
    return f"c{float(row['cell_size_mm']):g}_p{float(row['percentile']):g}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="CSV из batch_chm.py")
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

            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            plot_fits(ax, x, yv, result,
                      xlabel="V_chm (м³)",
                      ylabel=args.target,
                      title=f"{args.target} ~ {r.method}")
            fig.tight_layout()
            fname = f"{r.method}.png".replace("/", "_")
            fig.savefig(out / fname, dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
