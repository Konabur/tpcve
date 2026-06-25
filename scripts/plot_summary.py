#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["voxel", "alpha", "chm", "percentile", "count"]
VOLUME_METHODS = {"voxel", "alpha", "chm"}

COLOR_VOLUME = "#1f77b4"
COLOR_OTHER = "#999999"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate summary plots from regression CSVs")
    p.add_argument("--results-dir", default="results",
                    help="Root results directory (default: results)")
    p.add_argument("--stages", default="combined,Z31,Z65",
                    help="Comma-separated stage list (default: combined,Z31,Z65)")
    p.add_argument("--output-dir", default="results/figures/summary",
                    help="Output directory for plots (default: results/figures/summary)")
    return p.parse_args(argv)


def _pick_csv(method_dir: Path, stage: str, test: bool) -> Path | None:
    cands = []
    for p in method_dir.glob("*.csv"):
        stem = p.stem
        is_test = stem.endswith("_test")
        if test != is_test:
            continue
        clean = stem[:-5] if is_test else stem
        if stage == "combined":
            if "_Z31_" in clean or "_Z65_" in clean:
                continue
        else:
            if f"_{stage.upper()}_" not in f"_{clean}_":
                continue
        cands.append(p)
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def _load_best(method_dir: Path, stage: str) -> dict | None:
    if not method_dir.is_dir():
        return None
    train_csv = _pick_csv(method_dir, stage, test=False)
    if train_csv is None:
        return None
    df = pd.read_csv(train_csv)
    if df.empty:
        return None

    rank_cols = [c for c in df.columns if c.endswith("_r2") and not c.startswith("best")]
    if rank_cols:
        df = df.assign(_max_r2=df[rank_cols].max(axis=1))
        df = df.sort_values("_max_r2", ascending=False).drop(columns="_max_r2").reset_index(drop=True)

    best_row = df.iloc[0]
    method = method_dir.name
    param_label = str(best_row.get("method", ""))
    x_col = str(best_row.get("x_col", ""))
    best_model = str(best_row.get("best_model", "linear"))
    r2 = float(best_row.get(f"{best_model}_r2", float("nan")))
    if not np.isfinite(r2):
        return None

    ci_lo, ci_hi = float("nan"), float("nan")
    test_csv = _pick_csv(method_dir, stage, test=True)
    if test_csv is None:
        print(f"  [warn] {method}/{stage}: нет _test.csv — CI не будут отображены")
    else:
        tdf = pd.read_csv(test_csv)
        if not tdf.empty and 'x_col' in tdf.columns:
            match = tdf[tdf["x_col"] == x_col]
            if not match.empty:
                row = match.iloc[0]
                ci_lo = float(row.get(f"{best_model}_test_r2_ci_lo", float("nan")))
                ci_hi = float(row.get(f"{best_model}_test_r2_ci_hi", float("nan")))

    return {
        "method": method,
        "param_label": param_label,
        "r2": r2,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "x_col": x_col,
    }


def _plot_stage(data: list[dict], stage: str, out_dir: Path) -> None:
    data = sorted(data, key=lambda d: -d["r2"])
    methods_raw = [d["method"] for d in data]
    r2_vals = [d["r2"] for d in data]
    ci_lo = [d["ci_lo"] for d in data]
    ci_hi = [d["ci_hi"] for d in data]
    methods_label = []
    for d in data:
        label = d["param_label"]
        if label:
            label = label.replace("_layered", "").replace("layered_", "")
            methods_label.append(f"{d['method']}\n{label}")
        else:
            methods_label.append(d["method"])

    fig, ax = plt.subplots(figsize=(6, 1.8 + 0.35 * len(data)))
    y_pos = range(len(data))

    for i, (r2, y, method) in enumerate(zip(r2_vals, y_pos, methods_raw)):
        c = COLOR_VOLUME if method in VOLUME_METHODS else COLOR_OTHER
        lo, hi = ci_lo[i], ci_hi[i]
        has_ci = np.isfinite(lo) and np.isfinite(hi) and lo < hi
        if has_ci:
            ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.plot(r2, y, color=c, marker="o", markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(methods_label, fontsize=10)
    ax.set_xlabel("$R^2$ & 95% CI", fontsize=11)
    ax.set_title(f"Stage: {stage}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.6, len(data) - 0.6)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for i, (r2, y) in enumerate(zip(r2_vals, y_pos)):
        ax.text(r2, y - 0.18, f"{r2:.3f}", ha="center", va="bottom", fontsize=9)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=COLOR_VOLUME, marker="o", markersize=7,
               markeredgecolor="white", markeredgewidth=1.2, lw=2,
               label="Volume methods"),
        Line2D([0], [0], color=COLOR_OTHER, marker="o", markersize=7,
               markeredgecolor="white", markeredgewidth=1.2, lw=2,
               label="Baseline methods"),
    ], fontsize=8, loc="upper left")

    fig.tight_layout()
    fname = f"r2_stage_{stage}.png"
    fig.savefig(out_dir / fname, dpi=130)
    plt.close(fig)


def _copy_best_fits(csv_root: Path, best_dir: Path, stages: list[str]) -> None:
    """Copy the best-fit regression PNG per method from regression_plots."""
    plots_root = csv_root.parent / "regression_plots"
    if not plots_root.is_dir():
        return
    best_dir.mkdir(parents=True, exist_ok=True)
    for method_dir in csv_root.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for stage in stages:
            info = _load_best(method_dir, stage)
            if info is None or not info["x_col"]:
                continue
            png_name = f"{method}__{info['x_col']}.png".replace("/", "_")
            src = plots_root / method / png_name
            if src.exists():
                dst = best_dir / f"{method}__{stage}__{info['x_col']}.png"
                shutil.copy2(src, dst)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    csv_root = results_dir / "regression_csv"

    for stage in stages:
        stage_data = []
        for method in METHODS:
            row = _load_best(csv_root / method, stage)
            if row is not None:
                stage_data.append(row)
        if not stage_data:
            print(f"[{stage}] нет данных")
            continue
        _plot_stage(stage_data, stage, out_dir)
        print(f"[{stage}] график сохранён")

    _copy_best_fits(csv_root, out_dir / "best_fits", stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
