"""Регрессионные модели для analyze_correlation*.py.

Все фиттеры возвращают одинаковую схему dict с метриками на исходной шкале,
чтобы можно было честно сравнивать R²/RMSE между моделями.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.linear_model import HuberRegressor

ModelFit = dict


def compute_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"rmse": float("nan"), "rmse_pct": float("nan"),
                "bias": float("nan"), "r2": float("nan"), "n": 0}
    y = y[mask]
    y_pred = y_pred[mask]
    resid = y - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    y_mean = float(np.mean(y))
    rmse_pct = rmse / y_mean * 100 if y_mean > 0 else float("nan")
    bias = float(np.mean(resid))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "rmse_pct": rmse_pct, "bias": bias, "r2": r2,
            "n": int(mask.sum())}


def _metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    m = compute_metrics(y, y_pred)
    return {"rmse": m["rmse"], "rmse_pct": m["rmse_pct"],
            "bias": m["bias"], "r2": m["r2"]}


def fit_linear(x: np.ndarray, y: np.ndarray) -> ModelFit | None:
    if len(x) < 3:
        return None
    res = stats.linregress(x, y)
    slope = float(res.slope)
    intercept = float(res.intercept)
    predict = lambda xs: slope * xs + intercept
    m = _metrics(y, predict(x))
    return {
        "model": "linear",
        "n": len(x),
        "slope": slope,
        "intercept": intercept,
        "a": float("nan"),
        "b": float("nan"),
        "r": float(res.rvalue),
        "r2": m["r2"],
        "p_value": float(res.pvalue),
        "stderr": float(res.stderr),
        "rmse": m["rmse"],
        "rmse_pct": m["rmse_pct"],
        "bias": m["bias"],
        "predict": predict,
    }


def fit_power(x: np.ndarray, y: np.ndarray) -> ModelFit | None:
    """y = a · x^b — фит МНК в логарифмических координатах."""
    mask = (x > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    xm = x[mask]
    ym = y[mask]
    b, log_a = np.polyfit(np.log(xm), np.log(ym), 1)
    a = float(np.exp(log_a))
    b = float(b)
    predict = lambda xs: np.where(xs > 0, a * np.power(np.maximum(xs, 1e-12), b), np.nan)
    m = _metrics(y, predict(x))
    log_r = float(np.corrcoef(np.log(xm), np.log(ym))[0, 1])
    return {
        "model": "power",
        "n": int(mask.sum()),
        "slope": float("nan"),
        "intercept": float("nan"),
        "a": a,
        "b": b,
        "r": log_r,
        "r2": m["r2"],
        "p_value": float("nan"),
        "stderr": float("nan"),
        "rmse": m["rmse"],
        "rmse_pct": m["rmse_pct"],
        "bias": m["bias"],
        "predict": predict,
    }


def fit_huber(x: np.ndarray, y: np.ndarray) -> ModelFit | None:
    if len(x) < 3:
        return None
    try:
        model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=200)
        model.fit(x.reshape(-1, 1), y)
    except (ValueError, RuntimeError):
        return None
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    predict = lambda xs: slope * xs + intercept
    m = _metrics(y, predict(x))
    return {
        "model": "huber",
        "n": len(x),
        "slope": slope,
        "intercept": intercept,
        "a": float("nan"),
        "b": float("nan"),
        "r": float("nan"),
        "r2": m["r2"],
        "p_value": float("nan"),
        "stderr": float("nan"),
        "rmse": m["rmse"],
        "rmse_pct": m["rmse_pct"],
        "bias": m["bias"],
        "predict": predict,
    }


MODELS: dict[str, Callable[[np.ndarray, np.ndarray], ModelFit | None]] = {
    "linear": fit_linear,
    "power": fit_power,
    "huber": fit_huber,
}
TIE_BREAK = ["linear", "power", "huber"]


def fit_all(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Фитит все модели и выбирает лучшую по R².

    Возвращает None если ни одна модель не дала валидного фита.
    """
    fits: dict[str, ModelFit] = {}
    for name, fn in MODELS.items():
        f = fn(x, y)
        if f is not None and np.isfinite(f["r2"]):
            fits[name] = f
    if not fits:
        return None
    best_name = max(
        fits,
        key=lambda k: (fits[k]["r2"], -TIE_BREAK.index(k)),
    )
    return {
        "best_model": best_name,
        "best": fits[best_name],
        "all": fits,
    }


_COMMON_FIELDS = ("n", "r2", "rmse", "rmse_pct", "bias")
_MODEL_FIELDS: dict[str, tuple[str, ...]] = {
    "linear": _COMMON_FIELDS + ("slope", "intercept", "r", "p_value", "stderr"),
    "power":  _COMMON_FIELDS + ("a", "b", "r"),
    "huber":  _COMMON_FIELDS + ("slope", "intercept"),
}


def flatten_for_csv(result: dict) -> dict:
    """Разворачивает fit_all-результат в плоский dict для DataFrame.

    Все коэффициенты и метрики хранятся per-model: `<model>_<field>` — только
    осмысленные для данной модели поля. Топ-уровневая `best_model` — просто
    подсказка (по R²), а не источник дублирующих коэффициентов: пользователь
    сам выбирает модель из таблицы.
    """
    out = {"best_model": result["best_model"]}
    for name, fields in _MODEL_FIELDS.items():
        f = result["all"].get(name)
        for k in fields:
            out[f"{name}_{k}"] = f[k] if f is not None else float("nan")
    return out


_COLORS = {"linear": "#1f77b4", "power": "#2ca02c", "huber": "#ff7f0e"}


def plot_fits(ax, x: np.ndarray, y: np.ndarray, result: dict,
              xlabel: str, ylabel: str, title: str,
              x_test: np.ndarray | None = None,
              y_test: np.ndarray | None = None) -> None:
    """Scatter + три кривые на одном Axes; лучшая выделена жирной линией.

    Если переданы x_test/y_test — нарисовать их как test-scatter и дописать
    test-R² в подписи моделей.
    """
    has_test = (x_test is not None and y_test is not None
               and len(x_test) > 0 and len(y_test) > 0)
    ax.scatter(x, y, s=20, alpha=0.6, color="#444", label="train")
    if has_test:
        ax.scatter(x_test, y_test, s=28, alpha=0.85,
                   color="#d62728", marker="^", label="test")
    if x.size == 0:
        return
    x_max = float(np.max(x))
    if has_test:
        x_max = max(x_max, float(np.max(x_test)))
    x_min = float(np.min(x))
    if has_test:
        x_min = min(x_min, float(np.min(x_test)))
    xs = np.linspace(x_min, x_max, 200)
    best_name = result["best_model"]
    for name, f in result["all"].items():
        is_best = name == best_name
        ys = f["predict"](xs)
        if name == "linear":
            eq = f"slope={f['slope']:.3g}, b0={f['intercept']:.3g}"
        elif name == "power":
            eq = f"a={f['a']:.3g}, b={f['b']:.3g}"
        else:
            eq = f"slope={f['slope']:.3g}, b0={f['intercept']:.3g}"
        label = (f"{name}{' *' if is_best else ''}: "
                 f"R²={f['r2']:.3f}, RMSE%={f['rmse_pct']:.1f}")
        if has_test:
            vm = compute_metrics(np.asarray(y_test),
                                 f["predict"](np.asarray(x_test)))
            label += f"\n  test R²={vm['r2']:.3f}, RMSE%={vm['rmse_pct']:.1f}"
        label += f"\n  {eq}"
        ax.plot(
            xs, ys,
            color=_COLORS[name],
            lw=2.2 if is_best else 1.0,
            ls="-" if is_best else "--",
            label=label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    n_suffix = f", n={result['best']['n']}"
    if has_test:
        n_suffix += f", n_test={len(x_test)}"
    ax.set_title(f"{title}  [best: {best_name}{n_suffix}]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
