#!/bin/bash
# Пакетная обработка всех датасетов из data/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ -d "venv/bin" ]; then
    source venv/bin/activate
elif [ -d ".venv/bin" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "ПАКЕТНАЯ ОБРАБОТКА ДАТАСЕТОВ"
echo "=========================================="

for dataset_dir in data/*/; do
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi

    dataset_name=$(basename "$dataset_dir")
    echo ""
    echo ">>> Обработка: $dataset_name"

    # Проверка наличия settings.env
    if [ ! -f "$dataset_dir/settings.env" ]; then
        echo "    ⚠ Пропущен: settings.env не найден"
        continue
    fi

    # Поиск файла облака точек
    cloud_file=$(find "$dataset_dir" -maxdepth 1 -type f \( \
        -name "*.pcd" -o -name "*.las" -o -name "*.laz" -o \
        -name "*.ply" -o -name "*.xyz" -o -name "*.pts" -o \
        -name "*.npz" -o -name "*.db3" \) | head -n 1)

    if [ -z "$cloud_file" ]; then
        echo "    ⚠ Пропущен: облако точек не найдено"
        continue
    fi

    output_dir="results/$dataset_name"

    echo "    Облако: $(basename "$cloud_file")"
    echo "    Результаты: $output_dir"

    # Запуск обработки
    python main.py \
        --env-file "$dataset_dir/settings.env" \
        --cloud "$cloud_file" \
        --output-dir "$output_dir" \
        || echo "    ✗ Ошибка при обработке $dataset_name"

    echo "    ✓ Завершено: $dataset_name"
done

echo ""
echo "=========================================="
echo "ОБРАБОТКА ЗАВЕРШЕНА"
echo "=========================================="
