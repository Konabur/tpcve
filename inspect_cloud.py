#!/usr/bin/env python3
"""
Инспектор облака точек - показывает метаданные и статистику.
Использование: python inspect_cloud.py <путь_к_файлу.pcd>
"""

import sys
import numpy as np
import open3d as o3d


def inspect_cloud(filepath):
    print(f"Загрузка: {filepath}\n")

    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)

    print("=" * 60)
    print("ОСНОВНАЯ ИНФОРМАЦИЯ")
    print("=" * 60)
    print(f"Количество точек: {len(points):,}")
    print(f"Есть цвета: {pcd.has_colors()}")
    print(f"Есть нормали: {pcd.has_normals()}")

    if len(points) == 0:
        print("\nОблако пустое!")
        return

    print("\n" + "=" * 60)
    print("КООРДИНАТЫ")
    print("=" * 60)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    ranges = maxs - mins

    print(f"{'':8} {'X':>12} {'Y':>12} {'Z':>12}")
    print("-" * 60)
    print(f"{'Min':8} {mins[0]:12.6f} {mins[1]:12.6f} {mins[2]:12.6f}")
    print(f"{'Max':8} {maxs[0]:12.6f} {maxs[1]:12.6f} {maxs[2]:12.6f}")
    print(f"{'Mean':8} {means[0]:12.6f} {means[1]:12.6f} {means[2]:12.6f}")
    print(f"{'Std':8} {stds[0]:12.6f} {stds[1]:12.6f} {stds[2]:12.6f}")
    print(f"{'Range':8} {ranges[0]:12.6f} {ranges[1]:12.6f} {ranges[2]:12.6f}")

    print("\n" + "=" * 60)
    print("МАСШТАБ И ЕДИНИЦЫ")
    print("=" * 60)

    # Определяем вероятные единицы измерения по масштабу
    max_range = ranges.max()
    if max_range < 1:
        likely_units = "метры (м)"
    elif max_range < 100:
        likely_units = "метры (м) или дециметры (дм)"
    elif max_range < 10000:
        likely_units = "сантиметры (см) или миллиметры (мм)"
    else:
        likely_units = "миллиметры (мм) или микрометры (мкм)"

    print(f"Максимальный размер: {max_range:.6f}")
    print(f"Вероятные единицы: {likely_units}")

    # Проверяем плотность точек
    volume = ranges[0] * ranges[1] * ranges[2]
    if volume > 0:
        density = len(points) / volume
        print(f"Плотность: {density:.2f} точек/ед³")

    print("\n" + "=" * 60)
    print("СТАТИСТИКА РАССТОЯНИЙ")
    print("=" * 60)

    # Расстояния до ближайшего соседа (на полном облаке)
    nearest_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    if len(nearest_dists) > 0:
        print(f"Среднее расстояние до ближайшего соседа: {nearest_dists.mean():.6f}")
        print(f"Медианное расстояние: {np.median(nearest_dists):.6f}")
        print(f"Min расстояние: {nearest_dists.min():.6f}")
        print(f"Max расстояние: {nearest_dists.max():.6f}")

    if pcd.has_colors():
        print("\n" + "=" * 60)
        print("ЦВЕТА")
        print("=" * 60)
        colors = np.asarray(pcd.colors)
        print(f"Диапазон RGB: [{colors.min():.3f}, {colors.max():.3f}]")
        print(f"Средний цвет: R={colors[:, 0].mean():.3f} G={colors[:, 1].mean():.3f} B={colors[:, 2].mean():.3f}")

    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 60)

    if max_range > 100:
        print("⚠ Координаты имеют большой масштаб!")
        print(f"  Для работы в метрах умножьте координаты на: {1.0/max_range:.6f}")
        print(f"  Или разделите на: {max_range:.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python inspect_cloud.py <путь_к_файлу.pcd>")
        sys.exit(1)

    inspect_cloud(sys.argv[1])
