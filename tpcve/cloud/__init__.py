"""Облака точек как библиотека: генерация/загрузка, геометрия, объёмные методы.

Модули:
- ``generate_cloud`` — генерация синтетики, load/save, загрузка реальных сканов.
- ``geometry``       — downsample, SOR, alpha-shape, послойная триангуляция.
- ``volume_methods`` — воксель/hull/alpha объёмы (библиотека для методов).
- ``cloud_pipeline`` — единый preprocess (downsample/min-range/SOR/классификация).
"""
