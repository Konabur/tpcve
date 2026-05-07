"""Линейная регрессия biomass ~ V для long-формата CSV из batch_alpha.py.

Группирует строки по (voxel_mm, alpha, mode, layer_dz_mm) — каждая комбинация
параметров считается отдельным «методом», в нём по всем файлам строится
регрессия biomass ~ V_voxel (и V_random, если колонка есть).

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
from scipy import stats

GROUP_COLS = ["voxel_mm", "alpha", "mode", "layer_dz_mm"]


def fit(x: np.ndarray, y: np.ndarray) -> dict:
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


def fit_group(group: pd.DataFrame, vol_col: str, target: str) -> dict | None:
    s = pd.to_numeric(group[vol_col], errors="coerce")
    y = pd.to_numeric(group[target], errors="coerce")
    mask = s.notna() & y.notna() & (s > 0)
    if mask.sum() < 3:
        return None
    return fit(s[mask].to_numpy(), y[mask].to_numpy())


def label_for(row) -> str:
    mode = row["mode"]
    suffix = f"_dz{row['layer_dz_mm']}" if mode == "layered" else ""
    return f"v{float(row['voxel_mm']):g}_a{float(row['alpha']):g}_{mode}{suffix}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="CSV из batch_alpha.py")
    p.add_argument("--target", default="biomass")
    p.add_argument("--plots-dir", default=None,
                   help="Если указано — сохранить scatter+линию для каждой группы")
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

    sources = (["V_voxel", "V_random"] if args.source == "both"
               else [f"V_{args.source}"])
    sources = [c for c in sources if c in df.columns]

    rows = []
    for keys, grp in df.groupby(GROUP_COLS, dropna=False):
        meta = dict(zip(GROUP_COLS, keys))
        for vc in sources:
            res = fit_group(grp, vc, args.target)
            if res is None:
                continue
            rows.append({
                "method": label_for(meta),
                "source": vc,
                **meta,
                **res,
            })

    if not rows:
        raise SystemExit("Нет валидных данных для регрессии")

    res_df = (pd.DataFrame(rows)
              .sort_values("r2", ascending=False)
              .reset_index(drop=True))
    if args.top:
        res_df = res_df.head(args.top)

    print(f"\nЦель: {args.target} ~ V   (групп: {len(res_df)})")
    print("=" * 110)
    show_cols = ["method", "source", "n", "slope", "intercept",
                 "r", "r2", "p_value"]
    print(res_df[show_cols].to_string(index=False,
                                      float_format=lambda v: f"{v:.4g}"))
    print("=" * 110)

    out_csv = Path(args.csv).with_name(Path(args.csv).stem + "_regression.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")

    if args.plots_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(args.plots_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in res_df.itertuples():
            grp = df[(df["voxel_mm"] == r.voxel_mm)
                     & (df["alpha"] == r.alpha)
                     & (df["mode"] == r.mode)
                     & (df["layer_dz_mm"] == r.layer_dz_mm)]
            s = pd.to_numeric(grp[r.source], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            x = s[mask].to_numpy()
            yv = y[mask].to_numpy()

            xs = np.linspace(x.min(), x.max(), 100)
            ys = r.slope * xs + r.intercept

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(x, yv, s=20, alpha=0.6)
            ax.plot(xs, ys, "r-", lw=1.5,
                    label=f"y={r.slope:.3g}·x+{r.intercept:.3g}\n"
                          f"R²={r.r2:.3f}, p={r.p_value:.2g}, n={r.n}")
            ax.set_xlabel(f"{r.source} (м³)")
            ax.set_ylabel(args.target)
            ax.set_title(f"{args.target} ~ {r.method} ({r.source})")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fname = f"{r.method}__{r.source}.png".replace("/", "_")
            fig.savefig(out / fname, dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
