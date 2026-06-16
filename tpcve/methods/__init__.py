"""Реестр методов оценки признаков биомассы. Ленивый импорт по имени."""
from __future__ import annotations

import importlib
from types import ModuleType

# name -> import-path. Импорт модуля (а с ним open3d/scipy) происходит
# только при load(), чтобы `batch.py --help` был лёгким.
METHODS: dict[str, str] = {
    "voxel": "tpcve.methods.voxel",
    "alpha": "tpcve.methods.alpha",
    "chm": "tpcve.methods.chm",
    "count": "tpcve.methods.count",
    "percentile": "tpcve.methods.percentile",
}


def load(name: str) -> ModuleType:
    if name not in METHODS:
        raise KeyError(f"Неизвестный метод: {name!r}. Доступно: {sorted(METHODS)}")
    return importlib.import_module(METHODS[name])
