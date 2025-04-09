import os
import time
import glob
import pandas as pd
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt


def update(file_path: str, root: str) -> str:
    """
    Обновляет пути в текстовом файле, добавляя базовую директорию root.

    :param file_path: Путь к исходному файлу с путями.
    :param root: Базовая директория для добавления.
    :return: Путь к новому файлу с обновлёнными путями.
    """
    with open(file_path, "r") as file:
        paths = file.readlines()

    updated_paths = []
    for path in paths:
        path = path.strip()
        updated_paths.append(os.path.join(root, path) + "\n")

    new_file_path = os.path.join(
        os.path.dirname(file_path), "updated_" + os.path.basename(file_path)
    )
    with open(new_file_path, "w") as file:
        file.writelines(updated_paths)

    print("Updated file created successfully:", new_file_path)
    return new_file_path


def wait_for_results_file(run_folder: str, pattern: str = "results.csv",
                          timeout: int = 10) -> str:
    """
    Ждёт появления CSV-файла с результатами в папке run_folder в течение timeout секунд.

    :param run_folder: Папка, в которой ожидается файл результатов.
    :param pattern: Шаблон для поиска файла (по умолчанию 'results.csv').
    :param timeout: Время ожидания в секундах.
    :return: Путь к найденному файлу или пустая строка, если файл не найден.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        results_files = glob.glob(os.path.join(run_folder, pattern))
        if results_files:
            return results_files[0]
        time.sleep(1)
    return ""


def train_and_validate_models(models_to_train: dict, data_config: str,
                              project_name: str, epochs: int = 10) -> list:
    """
    Обучает и валидирует модели, возвращает список словарей с итоговыми метриками
    и временем обучения.

    :param models_to_train: Словарь моделей с путями к весовым файлам.
    :param data_config: Путь к конфигурационному файлу датасета.
    :param project_name: Папка для сохранения результатов.
    :param epochs: Число эпох обучения.
    :return: Список словарей с результатами.
    """
    results_list = []

    for model_name, weights_path in models_to_train.items():
        print(f"\n=== Обучение модели {model_name} ===")
        # Загружаем модель и переводим её на устройство
        model = YOLO(weights_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Задаём имя запуска для модели
        run_name = f"train_{model_name}"

        print(f"Начало обучения {model_name} на датасете {data_config} ...")
        start_time = time.time()
        # Запуск обучения с указанием project и run_name
        model.train(data=data_config, epochs=epochs, verbose=True,
                    project=project_name, name=run_name)
        training_time = time.time() - start_time
        print(f"Обучение модели {model_name} завершено за {training_time:.2f} секунд.")

        print(f"Выполняется валидация модели {model_name}...")
        val_results = model.val(data=data_config, verbose=True,
                                project=project_name, name=run_name)

        run_folder = os.path.join(project_name, run_name)
        results_file = wait_for_results_file(run_folder)

        if results_file:
            print("Найден файл с результатами:", results_file)
            df_metrics = pd.read_csv(results_file)
            # Берём метрики из последней строки (последняя эпоха)
            last_row = df_metrics.iloc[-1]
            precision = last_row["metrics/precision(B)"]
            recall = last_row["metrics/recall(B)"]
            mAP50 = last_row["metrics/mAP50(B)"]
            mAP50_95 = last_row["metrics/mAP50-95(B)"]
        else:
            print(
                f"Не удалось найти файл с результатами в {run_folder}. "
                "Используем данные из валидации (если доступны)."
            )
            precision = getattr(val_results, "precision", 0)
            recall = getattr(val_results, "recall", 0)
            mAP50 = getattr(val_results, "mAP50", 0)
            mAP50_95 = getattr(val_results, "mAP50_95", 0)

        print(
            f"Результаты {model_name}: Precision={precision}, Recall={recall}, "
            f"mAP50={mAP50}, mAP50-95={mAP50_95}"
        )

        results_list.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
            "Training Time (s)": training_time,
        })

    return results_list


def aggregate_results(result_folders: list, output_csv_path: str) -> pd.DataFrame:
    """
    Агрегирует результаты из указанных папок, формирует сводную таблицу и сохраняет её в CSV.

    :param result_folders: Список путей к папкам с результатами.
    :param output_csv_path: Путь для сохранения итогового CSV-файла.
    :return: Сформированный DataFrame с итоговыми метриками.
    """
    results_list = []

    for run_folder in result_folders:
        # Используем имя папки как название модели
        model_name = os.path.basename(run_folder)
        results_file = wait_for_results_file(run_folder)

        if results_file:
            print(f"Найден файл с результатами в {run_folder}: {results_file}")
            df_metrics = pd.read_csv(results_file)
            # Берём последнюю строку с итоговыми метриками (последняя эпоха)
            last_row = df_metrics.iloc[-1]
            precision = last_row.get("metrics/precision(B)", 0)
            recall = last_row.get("metrics/recall(B)", 0)
            mAP50 = last_row.get("metrics/mAP50(B)", 0)
            mAP50_95 = last_row.get("metrics/mAP50-95(B)", 0)
        else:
            print(
                f"Не удалось найти файл с результатами в {run_folder}. "
                "Используем значения по умолчанию."
            )
            precision = recall = mAP50 = mAP50_95 = 0

        results_list.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
        })

    results_df = pd.DataFrame(results_list)
    print("\n=== Сводная таблица результатов ===")
    print(results_df)

    for col in ["Precision", "Recall", "mAP50", "mAP50-95"]:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce").fillna(0)

    results_df.to_csv(output_csv_path, index=False)
    print("Итоговые результаты сохранены в", output_csv_path)
    return results_df


def plot_results(results_df: pd.DataFrame) -> None:
    """
    Строит графики для визуального сравнения метрик.

    :param results_df: DataFrame с результатами.
    """
    num_models = len(results_df["Model"])
    model_colors = plt.cm.tab10(range(num_models))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].bar(results_df["Model"], results_df["Precision"], color=model_colors)
    axes[0].set_title("Precision")
    axes[0].set_xlabel("Модель")
    axes[0].set_ylabel("Precision")

    axes[1].bar(results_df["Model"], results_df["Recall"], color=model_colors)
    axes[1].set_title("Recall")
    axes[1].set_xlabel("Модель")
    axes[1].set_ylabel("Recall")

    axes[2].bar(results_df["Model"], results_df["mAP50"], color=model_colors)
    axes[2].set_title("mAP50")
    axes[2].set_xlabel("Модель")
    axes[2].set_ylabel("mAP50")

    axes[3].bar(results_df["Model"], results_df["mAP50-95"], color=model_colors)
    axes[3].set_title("mAP50-95")
    axes[3].set_xlabel("Модель")
    axes[3].set_ylabel("mAP50-95")

    plt.tight_layout()
    plt.show(block=True)