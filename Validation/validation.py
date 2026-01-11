import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json


class YOLOModelComparison:
    def __init__(self, model1_path, model2_path, data_path, confidence_threshold=0.5, iou_threshold=0.5):
        """
        Инициализация сравнения моделей YOLO

        Args:
            model1_path: путь к первой модели
            model2_path: путь ко второй модели
            data_path: путь к валидационным данным
            confidence_threshold: порог уверенности для детекции
            iou_threshold: порог IoU для определения TP
        """
        self.model1 = YOLO(model1_path)
        self.model2 = YOLO(model2_path)
        self.model1_name = "Car_Detector (Synthetic)"
        self.model2_name = "YOLO_Car_Detector (Real)"

        self.data_path = Path(data_path)
        self.images_path = self.data_path / "images"
        self.labels_path = self.data_path / "labels"

        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Results for each model
        self.results = {
            self.model1_name: {'tp': 0, 'fp': 0, 'fn': 0, 'precisions': [], 'recalls': [], 'confidences': []},
            self.model2_name: {'tp': 0, 'fp': 0, 'fn': 0, 'precisions': [], 'recalls': [], 'confidences': []}
        }

    def load_ground_truth(self, label_file):
        """Загрузка ground truth разметки из YOLO формата"""
        gt_boxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        gt_boxes.append([class_id, x_center, y_center, width, height])
        return np.array(gt_boxes) if gt_boxes else np.array([]).reshape(0, 5)

    def yolo_to_xyxy(self, boxes, img_width, img_height):
        """Конвертация из YOLO формата (x_center, y_center, width, height) в (x1, y1, x2, y2)"""
        if len(boxes) == 0:
            return np.array([]).reshape(0, 4)

        xyxy_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box[1:5]
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height
            xyxy_boxes.append([x1, y1, x2, y2])

        return np.array(xyxy_boxes)

    def calculate_iou(self, box1, box2):
        """Вычисление IoU между двумя bbox"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0

        intersection = (x2_min - x1_max) * (y2_min - y1_max)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def evaluate_predictions(self, pred_boxes, pred_confidences, gt_boxes):
        """Оценка предсказаний модели"""
        tp = fp = fn = 0

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return tp, fp, fn

        if len(gt_boxes) == 0:
            return tp, len(pred_boxes), fn

        if len(pred_boxes) == 0:
            return tp, fp, len(gt_boxes)

        # Creation IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)

        sorted_indices = np.argsort(pred_confidences)[::-1]

        for pred_idx in sorted_indices:
            best_gt_idx = -1
            best_iou = 0

            for gt_idx in range(len(gt_boxes)):
                if not gt_matched[gt_idx] and iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold:
                tp += 1
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
            else:
                fp += 1

        fn = len(gt_boxes) - np.sum(gt_matched)

        return tp, fp, fn

    def process_image(self, image_path, model, model_name):
        """Обработка одного изображения"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img_height, img_width = img.shape[:2]

        results = model(str(image_path), conf=self.conf_threshold, verbose=False)

        pred_boxes = []
        pred_confidences = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                pred_boxes.append(box)
                pred_confidences.append(conf)

        # Download ground truth
        label_file = self.labels_path / (image_path.stem + '.txt')
        gt_boxes_yolo = self.load_ground_truth(label_file)
        gt_boxes_xyxy = self.yolo_to_xyxy(gt_boxes_yolo, img_width, img_height)

        # Evaluation conf matrix
        tp, fp, fn = self.evaluate_predictions(pred_boxes, pred_confidences, gt_boxes_xyxy)

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_gt': len(gt_boxes_xyxy),
            'num_pred': len(pred_boxes),
            'confidences': pred_confidences
        }

    def evaluate_models(self):
        """Оценка обеих моделей на валидационном наборе"""
        image_files = list(self.images_path.glob('*.jpg')) + list(self.images_path.glob('*.png'))

        print(f"Найдено {len(image_files)} изображений для валидации")
        print("Начинаем оценку моделей...")

        for i, image_path in enumerate(image_files):
            if (i + 1) % 50 == 0:
                print(f"Обработано {i + 1}/{len(image_files)} изображений")

            # First model Eval
            result1 = self.process_image(image_path, self.model1, self.model1_name)
            if result1:
                self.results[self.model1_name]['tp'] += result1['tp']
                self.results[self.model1_name]['fp'] += result1['fp']
                self.results[self.model1_name]['fn'] += result1['fn']
                self.results[self.model1_name]['confidences'].extend(result1['confidences'])

            # Second model Eval
            result2 = self.process_image(image_path, self.model2, self.model2_name)
            if result2:
                self.results[self.model2_name]['tp'] += result2['tp']
                self.results[self.model2_name]['fp'] += result2['fp']
                self.results[self.model2_name]['fn'] += result2['fn']
                self.results[self.model2_name]['confidences'].extend(result2['confidences'])

        print("Оценка завершена!")

    def calculate_metrics(self):
        """Вычисление метрик для обеих моделей"""
        metrics = {}

        for model_name in [self.model1_name, self.model2_name]:
            tp = self.results[model_name]['tp']
            fp = self.results[model_name]['fp']
            fn = self.results[model_name]['fn']
            tn = 0

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[model_name] = {
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'Total_GT': tp + fn,
                'Total_Pred': tp + fp
            }

        return metrics

    def print_detailed_comparison(self, metrics):
        """Подробное сравнение метрик"""
        print("\n" + "=" * 80)
        print("ДЕТАЛЬНОЕ СРАВНЕНИЕ МЕТРИК МОДЕЛЕЙ YOLO")
        print("=" * 80)

        df_data = []
        for model_name, model_metrics in metrics.items():
            df_data.append({
                'Модель': model_name,
                'TP': model_metrics['TP'],
                'FP': model_metrics['FP'],
                'FN': model_metrics['FN'],
                'Precision': f"{model_metrics['Precision']:.4f}",
                'Recall': f"{model_metrics['Recall']:.4f}",
                'F1-Score': f"{model_metrics['F1-Score']:.4f}",
                'Всего GT': model_metrics['Total_GT'],
                'Всего предсказаний': model_metrics['Total_Pred']
            })

        df = pd.DataFrame(df_data)
        print("\nТАБЛИЦА РЕЗУЛЬТАТОВ:")
        print(df.to_string(index=False))

        print("\n" + "=" * 80)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
        print("=" * 80)

        model1_metrics = metrics[self.model1_name]
        model2_metrics = metrics[self.model2_name]

        print(f"\n1. ТОЧНОСТЬ (Precision):")
        print(f"   • {self.model1_name}: {model1_metrics['Precision']:.4f}")
        print(f"   • {self.model2_name}: {model2_metrics['Precision']:.4f}")

        if model1_metrics['Precision'] > model2_metrics['Precision']:
            diff = model1_metrics['Precision'] - model2_metrics['Precision']
            print(f"   → {self.model1_name} превосходит по точности на {diff:.4f} ({diff * 100:.2f}%)")
        else:
            diff = model2_metrics['Precision'] - model1_metrics['Precision']
            print(f"   → {self.model2_name} превосходит по точности на {diff:.4f} ({diff * 100:.2f}%)")

        print(f"\n2. ПОЛНОТА (Recall):")
        print(f"   • {self.model1_name}: {model1_metrics['Recall']:.4f}")
        print(f"   • {self.model2_name}: {model2_metrics['Recall']:.4f}")

        if model1_metrics['Recall'] > model2_metrics['Recall']:
            diff = model1_metrics['Recall'] - model2_metrics['Recall']
            print(f"   → {self.model1_name} превосходит по полноте на {diff:.4f} ({diff * 100:.2f}%)")
        else:
            diff = model2_metrics['Recall'] - model1_metrics['Recall']
            print(f"   → {self.model2_name} превосходит по полноте на {diff:.4f} ({diff * 100:.2f}%)")

        print(f"\n3. F1-МЕРА:")
        print(f"   • {self.model1_name}: {model1_metrics['F1-Score']:.4f}")
        print(f"   • {self.model2_name}: {model2_metrics['F1-Score']:.4f}")

        if model1_metrics['F1-Score'] > model2_metrics['F1-Score']:
            diff = model1_metrics['F1-Score'] - model2_metrics['F1-Score']
            print(f"   → {self.model1_name} превосходит по F1-мере на {diff:.4f} ({diff * 100:.2f}%)")
        else:
            diff = model2_metrics['F1-Score'] - model1_metrics['F1-Score']
            print(f"   → {self.model2_name} превосходит по F1-мере на {diff:.4f} ({diff * 100:.2f}%)")

        print(f"\n4. АНАЛИЗ ОШИБОК:")
        print(f"   False Positives (ложные срабатывания):")
        print(f"   • {self.model1_name}: {model1_metrics['FP']}")
        print(f"   • {self.model2_name}: {model2_metrics['FP']}")

        print(f"   False Negatives (пропущенные объекты):")
        print(f"   • {self.model1_name}: {model1_metrics['FN']}")
        print(f"   • {self.model2_name}: {model2_metrics['FN']}")

        print(f"\n5. ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
        print("=" * 50)

        if model1_metrics['Precision'] > model2_metrics['Precision']:
            print(f"• Модель на синтетических данных ({self.model1_name}) показывает")
            print("  более высокую точность, что означает меньше ложных срабатываний.")
        else:
            print(f"• Модель на реальных данных ({self.model2_name}) показывает")
            print("  более высокую точность, что означает меньше ложных срабатываний.")

        if model1_metrics['Recall'] > model2_metrics['Recall']:
            print(f"• Модель на синтетических данных лучше обнаруживает объекты")
            print("  (меньше пропусков).")
        else:
            print(f"• Модель на реальных данных лучше обнаруживает объекты")
            print("  (меньше пропусков).")

        if model1_metrics['F1-Score'] > model2_metrics['F1-Score']:
            print(f"\n• ОБЩИЙ ВЫВОД: Модель {self.model1_name}")
            print("  показывает лучшие результаты по F1-мере (баланс точности и полноты).")
        else:
            print(f"\n• ОБЩИЙ ВЫВОД: Модель {self.model2_name}")
            print("  показывает лучшие результаты по F1-мере (баланс точности и полноты).")

    def create_visualizations(self, metrics):
        """Создание визуализаций для сравнения"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Сравнение метрик YOLO моделей', fontsize=16, fontweight='bold')

        models = list(metrics.keys())

        # 1. Based metrics
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        model1_values = [metrics[models[0]][m] for m in metrics_names]
        model2_values = [metrics[models[1]][m] for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35

        axes[0, 0].bar(x - width / 2, model1_values, width, label=models[0], alpha=0.8)
        axes[0, 0].bar(x + width / 2, model2_values, width, label=models[1], alpha=0.8)
        axes[0, 0].set_xlabel('Метрики')
        axes[0, 0].set_ylabel('Значение')
        axes[0, 0].set_title('Сравнение основных метрик')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. TP, FP, FN
        error_types = ['TP', 'FP', 'FN']
        model1_errors = [metrics[models[0]][e] for e in error_types]
        model2_errors = [metrics[models[1]][e] for e in error_types]

        x = np.arange(len(error_types))
        axes[0, 1].bar(x - width / 2, model1_errors, width, label=models[0], alpha=0.8)
        axes[0, 1].bar(x + width / 2, model2_errors, width, label=models[1], alpha=0.8)
        axes[0, 1].set_xlabel('Тип результата')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].set_title('Анализ TP, FP, FN')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(error_types)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Confident distribution
        if self.results[models[0]]['confidences'] and self.results[models[1]]['confidences']:
            axes[1, 0].hist(self.results[models[0]]['confidences'], bins=30, alpha=0.7,
                            label=models[0], density=True)
            axes[1, 0].hist(self.results[models[1]]['confidences'], bins=30, alpha=0.7,
                            label=models[1], density=True)
            axes[1, 0].set_xlabel('Уверенность')
            axes[1, 0].set_ylabel('Плотность')
            axes[1, 0].set_title('Распределение уверенности предсказаний')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Pivot table in the form heatmap
        comparison_data = []
        for model in models:
            comparison_data.append([
                metrics[model]['Precision'],
                metrics[model]['Recall'],
                metrics[model]['F1-Score']
            ])

        sns.heatmap(comparison_data,
                    xticklabels=['Precision', 'Recall', 'F1-Score'],
                    yticklabels=[m.split('(')[0].strip() for m in models],
                    annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=axes[1, 1])
        axes[1, 1].set_title('Тепловая карта метрик')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nВизуализация сохранена как 'model_comparison.png'")

    def save_results_to_json(self, metrics):
        """Сохранение результатов в JSON файл"""

        def convert_to_serializable(obj):
            """Конвертация NumPy типов в стандартные Python типы"""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        results_data = {
            'comparison_settings': {
                'confidence_threshold': float(self.conf_threshold),
                'iou_threshold': float(self.iou_threshold),
                'model1_name': self.model1_name,
                'model2_name': self.model2_name
            },
            'metrics': convert_to_serializable(metrics),
            'raw_results': convert_to_serializable(self.results)
        }

        with open('validation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print("Результаты сохранены в 'validation_results.json'")


def main():
    """Основная функция для запуска сравнения"""

    # Models paths
    model1_path = "../Models/Car_Detector.pt"
    model2_path = "../yolo_car_detector.pt"

    # Path to validate data
    validation_data_path = "data2/valid"

    # Existing check
    if not os.path.exists(model1_path):
        print(f"Ошибка: Модель {model1_path} не найдена!")
        return

    if not os.path.exists(model2_path):
        print(f"Ошибка: Модель {model2_path} не найдена!")
        return

    if not os.path.exists(validation_data_path):
        print(f"Ошибка: Директория с валидационными данными {validation_data_path} не найдена!")
        return

    # Can change this thing
    comparator = YOLOModelComparison(
        model1_path=model1_path,
        model2_path=model2_path,
        data_path=validation_data_path,
        confidence_threshold=0.5,
        iou_threshold=0.5
    )

    # Evaluation
    comparator.evaluate_models()

    # Metrics
    metrics = comparator.calculate_metrics()

    # Print
    comparator.print_detailed_comparison(metrics)

    # Visualisation
    comparator.create_visualizations(metrics)

    # Save
    comparator.save_results_to_json(metrics)

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("Проверьте файлы 'model_comparison.png' и 'validation_results.json'")
    print("=" * 80)


if __name__ == "__main__":
    main()
