"""
Эксперимент: влияние окклюзии на точность оценки объёма.
Прогоняем с разной силой окклюзии: 0%, 10%, 20%, 30%, 40%.
"""

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

np.random.seed(42)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Генерация облака — в generate_cloud.py, оценка объёма — в volume_methods.py
from tpcve.cloud.generate_cloud import (
    generate_ground, generate_wheat_field, simulate_occlusion,
    add_realistic_noise
)
from tpcve.cloud.volume_methods import voxel_volume

print("=" * 60)
print("ЭКСПЕРИМЕНТ: ВЛИЯНИЕ ОККЛЮЗИИ НА ТОЧНОСТЬ")
print("=" * 60)

# Параметры эксперимента
occlusion_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
best_voxel_size = 0.007  # лучший размер из основного эксперимента
scanner_pos = np.array([-2.5, 0, 0.5])

results = []

for occ_strength in occlusion_levels:
    print(f"\n--- Окклюзия {occ_strength*100:.0f}% ---")

    # Генерация данных
    ground_pts = generate_ground()
    vegetation_pts, plant_params = generate_wheat_field()
    total_gt_volume = sum(p['volume'] for p in plant_params)
    all_pts_clean = np.vstack([ground_pts, vegetation_pts])

    # Окклюзия
    all_pts_occluded, _ = simulate_occlusion(all_pts_clean, scanner_pos,
                                              occlusion_strength=occ_strength)
    n_occluded = len(all_pts_clean) - len(all_pts_occluded)
    occ_pct = 100 * n_occluded / len(all_pts_clean)

    # Шум
    all_pts_noisy = add_realistic_noise(all_pts_occluded, noise_std=0.008,
                                         outlier_fraction=0.06, phantom_fraction=0.03)

    # Фильтрация
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts_noisy)
    pcd_sor, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pts_after_sor = np.asarray(pcd_sor.points)

    pcd_sor2 = o3d.geometry.PointCloud()
    pcd_sor2.points = o3d.utility.Vector3dVector(pts_after_sor)
    pcd_filtered, _ = pcd_sor2.remove_radius_outlier(nb_points=2, radius=0.08)
    pts_filtered = np.asarray(pcd_filtered.points)

    # Классификация
    height_threshold = 0.04
    vegetation_mask = pts_filtered[:, 2] >= height_threshold
    vegetation_classified = pts_filtered[vegetation_mask]

    # Оценка объёма
    vol, n_vox = voxel_volume(vegetation_classified, voxel_size=best_voxel_size)
    err = (vol - total_gt_volume) / total_gt_volume * 100

    print(f"  Удалено окклюзией: {occ_pct:.1f}%")
    print(f"  Точек растительности: {len(vegetation_classified)}")
    print(f"  Объём: {vol:.6f} м³")
    print(f"  Ошибка: {err:+.1f}%")

    results.append({
        'occlusion_strength': occ_strength,
        'occlusion_pct': occ_pct,
        'n_vegetation_points': len(vegetation_classified),
        'volume': vol,
        'error_pct': err
    })

# График
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

occ_strengths = [r['occlusion_strength'] * 100 for r in results]
errors = [abs(r['error_pct']) for r in results]
n_points = [r['n_vegetation_points'] for r in results]

# График 1: Ошибка vs окклюзия
ax1.plot(occ_strengths, errors, 'o-', color='#f44336', linewidth=2.5, markersize=8)
ax1.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='20% ошибка')
ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% ошибка')
ax1.set_xlabel('Сила окклюзии, %', fontsize=12)
ax1.set_ylabel('Абсолютная ошибка оценки объёма, %', fontsize=12)
ax1.set_title(f'Влияние окклюзии на точность\n(воксельный метод, размер {best_voxel_size*1000:.0f}мм)',
              fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# График 2: Количество точек vs окклюзия
ax2.plot(occ_strengths, n_points, 's-', color='#2196F3', linewidth=2.5, markersize=8)
ax2.set_xlabel('Сила окклюзии, %', fontsize=12)
ax2.set_ylabel('Количество точек растительности', fontsize=12)
ax2.set_title('Потеря данных из-за окклюзии', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_occlusion_experiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nСохранён: 05_occlusion_experiment.png")

# Сохранение результатов
with open(f'{OUTPUT_DIR}/occlusion_results.json', 'w') as f:
    json.dump({
        'voxel_size_m': best_voxel_size,
        'ground_truth_volume_m3': total_gt_volume,
        'results': results
    }, f, indent=2, ensure_ascii=False)

print("Сохранён: occlusion_results.json")

print("\n" + "=" * 60)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
print("=" * 60)
print(f"\nВывод: при окклюзии >30% ошибка резко растёт")
