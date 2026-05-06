"""Линейная регрессия biomass ~ volume для каждой колонки-метода в CSV.

Использование:
    uv run python analyze_correlation.py results/batch.csv
    uv run python analyze_correlation.py results/batch.csv --plots-dir results/regression
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NON_METHOD_COLS = {"file", "biomass", "col3", "col4", "col5",
                   "n_input", "n_after_sor", "n_vegetation", "error"}


def fit_column(x: np.ndarray, y: np.ndarray) -> dict:
    res = stats.linregress(x, y)
    return {
        "n": len(x),
        "slope": res.slope,
        "intercept": res.intercept,
        "r2": res.rvalue ** 2,
        "r": res.rvalue,
        "p_value": res.pvalue,
        "stderr": res.stderr,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="CSV из batch_process.py")
    p.add_argument("--plots-dir", default=None,
                   help="Если указано — сохранить scatter+линию для каждого метода")
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

    rows = []
    for col in method_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        mask = s.notna() & df[args.target].notna() & (s > 0)
        if mask.sum() < 3:
            continue
        x = s[mask].to_numpy()
        y = df.loc[mask, args.target].to_numpy()
        rows.append({"method": col, **fit_column(x, y)})

    if not rows:
        raise SystemExit("Нет валидных данных для регрессии")

    res = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    if args.top:
        res = res.head(args.top)

    print(f"\nЦель: {args.target} ~ volume   (всего методов: {len(res)})")
    print("=" * 90)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("=" * 90)

    out_csv = Path(args.csv).with_name(Path(args.csv).stem + "_regression.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")

    if args.plots_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(args.plots_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in res.itertuples():
            col = r.method
            s = pd.to_numeric(df[col], errors="coerce")
            mask = s.notna() & df[args.target].notna() & (s > 0)
            x = s[mask].to_numpy()
            y = df.loc[mask, args.target].to_numpy()

            xs = np.linspace(x.min(), x.max(), 100)
            ys = r.slope * xs + r.intercept

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(x, y, s=20, alpha=0.6)
            ax.plot(xs, ys, "r-", lw=1.5,
                    label=f"y={r.slope:.3g}·x+{r.intercept:.3g}\n"
                          f"R²={r.r2:.3f}, p={r.p_value:.2g}, n={r.n}")
            ax.set_xlabel(f"{col} (м³)")
            ax.set_ylabel(args.target)
            ax.set_title(f"{args.target} ~ {col}")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"{col}.png", dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
