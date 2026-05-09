"""Автогенерация имён выходных файлов из параметров запуска.

build_name() собирает имя из stem источника + sweep-параметров + extras.
default_path() возвращает путь в нужной подпапке results/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

RESULTS_ROOT = Path("results")

KIND_DIRS: dict[str, Path] = {
    "volume_voxel":           RESULTS_ROOT / "volume_csv" / "voxel",
    "volume_alpha":           RESULTS_ROOT / "volume_csv" / "alpha",
    "regression_voxel":       RESULTS_ROOT / "regression_csv" / "voxel",
    "regression_alpha":       RESULTS_ROOT / "regression_csv" / "alpha",
    "regression_plots_voxel": RESULTS_ROOT / "regression_plots" / "voxel",
    "regression_plots_alpha": RESULTS_ROOT / "regression_plots" / "alpha",
    "downsample_compare":     RESULTS_ROOT / "downsample" / "downsample_compare",
    "downsample_alpha":       RESULTS_ROOT / "downsample" / "downsample_alpha",
}


def _fmt_num(x) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def _list_token(prefix: str, values: Iterable | None) -> str:
    if not values:
        return ""
    return prefix + "_".join(_fmt_num(v) for v in values)


def _source_stem(source: str | None, source_kind: str) -> str:
    if not source:
        return ""
    p = Path(source)
    return p.stem if source_kind == "list" else p.name


def build_name(
    *,
    source: str | None = None,
    source_kind: str = "list",  # "list" -> stem (без .txt), "dir" -> name папки
    voxels_mm: Iterable | None = None,
    auto_voxel: bool = False,
    alphas: Iterable | None = None,
    layered: bool = False,
    layer_dz_mm: float | None = None,
    extra: dict | None = None,
) -> str:
    """Собрать имя файла из параметров.

    Формат: <source>[_v<sizes>][_vauto][_a<alphas>][_layered][_dz<dz>][_<k><v>...]
    """
    parts: list[str] = []

    stem = _source_stem(source, source_kind)
    if stem:
        parts.append(stem)

    v_tok = _list_token("v", list(voxels_mm) if voxels_mm else [])
    if v_tok:
        parts.append(v_tok)
    if auto_voxel:
        parts.append("vauto")

    a_tok = _list_token("a", list(alphas) if alphas else [])
    if a_tok:
        parts.append(a_tok)

    if layered:
        parts.append("layered")
        if layer_dz_mm is not None:
            parts.append(f"dz{_fmt_num(layer_dz_mm)}")

    if extra:
        for k, v in extra.items():
            if v is None or v is False or v == "":
                continue
            if v is True:
                parts.append(str(k))
            elif isinstance(v, (int, float)):
                parts.append(f"{k}{_fmt_num(v)}")
            else:
                parts.append(f"{k}{v}")

    return "_".join(parts) or "run"


def default_path(kind: str, name: str, ext: str = ".csv") -> Path:
    if kind not in KIND_DIRS:
        raise ValueError(
            f"Unknown kind: {kind!r}; available: {sorted(KIND_DIRS)}"
        )
    d = KIND_DIRS[kind]
    d.mkdir(parents=True, exist_ok=True)
    if ext and not ext.startswith("."):
        ext = "." + ext
    return d / (name + ext)
