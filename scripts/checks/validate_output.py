"""Validate single-split or K-Fold result files according to output contract."""
import argparse
import csv
import json
from pathlib import Path

HISTORY_COLUMNS = [
    "epoch", "train_loss", "train_accuracy", "train_f1_macro",
    "test_loss", "test_accuracy", "test_f1_macro",
]
CV_HISTORY_COLUMNS = ["fold"] + HISTORY_COLUMNS


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def validate_result(out_dir, errors):
    result_path = out_dir / "result.json"
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ["data", "model", "runtime", "output", "metrics"]:
            if key not in data:
                errors.append(f"result.json 缺少必要字段：{key}")
        metrics = data.get("metrics", {})
        for key in ["accuracy", "confusion_matrix"]:
            if key not in metrics:
                errors.append(f"result.json.metrics 缺少字段：{key}")
        return data
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"无法读取 result.json：{exc}")
        return None


def validate_predictions(out_dir, data, errors, cross_validation):
    path = out_dir / "predictions.csv"
    try:
        columns, rows = read_csv(path)
        if not {"y_true", "y_pred"}.issubset(columns):
            errors.append("predictions.csv 缺少 y_true 或 y_pred")
        probs = [name for name in columns if name.startswith("prob_")]
        if data is not None and "labels" in data:
            expected = [f"prob_{label}" for label in data["labels"]["classes"]]
            if probs and probs != expected:
                errors.append("概率列顺序与 labels.classes 不一致")
        if not rows:
            errors.append("predictions.csv 没有预测样本")
        if data is not None:
            matrix = data.get("metrics", {}).get("confusion_matrix")
            if isinstance(matrix, list) and sum(sum(row) for row in matrix) != len(rows):
                errors.append("混淆矩阵计数之和与预测行数不一致")

        if cross_validation:
            required = {"sample_index", "fold", "y_true", "y_pred"}
            if not required.issubset(columns):
                errors.append(f"OOF predictions.csv 缺少字段：{', '.join(sorted(required - set(columns)))}")
            else:
                try:
                    sample_indices = [int(row["sample_index"]) for row in rows]
                    fold_values = [int(row["fold"]) for row in rows]
                    n_samples = int(data.get("data", {}).get("n_samples", len(rows))) if data else len(rows)
                    n_splits = int(data.get("data", {}).get("n_splits", 0)) if data else 0
                    if len(rows) != n_samples:
                        errors.append(f"OOF 预测行数 {len(rows)} 与 n_samples={n_samples} 不一致")
                    if len(set(sample_indices)) != len(sample_indices):
                        errors.append("OOF sample_index 存在重复")
                    if sorted(sample_indices) != list(range(n_samples)):
                        errors.append("OOF sample_index 未完整覆盖 0...n_samples-1")
                    if n_splits < 2 or set(fold_values) != set(range(1, n_splits + 1)):
                        errors.append("OOF fold 编号未完整覆盖 1...n_splits")
                except (ValueError, TypeError, AttributeError):
                    errors.append("OOF sample_index / fold / result.json 数据配置不是有效整数")
        return rows
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"无法读取 predictions.csv：{exc}")
        return []


def validate_history(out_dir, errors, required, cross_validation, n_splits=None):
    path = out_dir / "training_history.csv"
    if not required and not path.exists():
        return
    try:
        columns, rows = read_csv(path)
        expected = CV_HISTORY_COLUMNS if cross_validation else HISTORY_COLUMNS
        if columns != expected:
            errors.append(f"训练历史字段不一致，实际：{columns}，要求：{expected}")
        if not rows:
            errors.append("training_history.csv 没有 epoch 数据")
        if cross_validation and rows:
            try:
                folds = {int(row["fold"]) for row in rows}
                expected_folds = set(range(1, n_splits + 1)) if n_splits else folds
                if folds != expected_folds:
                    errors.append("training_history.csv 未完整覆盖所有 Fold 编号")
            except (KeyError, ValueError):
                errors.append("training_history.csv fold 字段无效")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"无法读取 training_history.csv：{exc}")


def validate_fold_files(out_dir, data, prediction_rows, errors):
    fold_metrics_path = out_dir / "fold_metrics.csv"
    summary_path = out_dir / "fold_metrics_summary.json"
    n_splits = int(data.get("data", {}).get("n_splits", 0)) if data else 0
    try:
        columns, rows = read_csv(fold_metrics_path)
        required = {"fold", "train_samples", "val_samples"}
        if not required.issubset(columns):
            errors.append(f"fold_metrics.csv 缺少字段：{', '.join(sorted(required - set(columns)))}")
        fold_ids = [int(row["fold"]) for row in rows]
        if len(rows) != n_splits or set(fold_ids) != set(range(1, n_splits + 1)) or len(set(fold_ids)) != len(fold_ids):
            errors.append("fold_metrics.csv 的折数或 fold 编号不正确")
        if prediction_rows and "fold" in prediction_rows[0]:
            prediction_counts = {}
            for item in prediction_rows:
                fold_id = int(item["fold"])
                prediction_counts[fold_id] = prediction_counts.get(fold_id, 0) + 1
            for item in rows:
                fold_id = int(item["fold"])
                if int(item["val_samples"]) != prediction_counts.get(fold_id, 0):
                    errors.append(f"Fold {fold_id}：val_samples 与 OOF 预测数量不一致")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"无法正确读取 fold_metrics.csv：{exc}")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not summary:
            errors.append("fold_metrics_summary.json 必须是非空 JSON 对象")
        if data is not None:
            save_models = data.get("model", {}).get("configured_params", {}).get("save_fold_models", False)
            if save_models:
                for fold_id in range(1, n_splits + 1):
                    fold_dir = out_dir / f"fold_{fold_id}"
                    if not fold_dir.is_dir() or not any(path.is_file() for path in fold_dir.iterdir()):
                        errors.append(f"配置要求保存模型，但缺少 fold_{fold_id} 模型文件")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"无法读取 fold_metrics_summary.json：{exc}")


def main():
    parser = argparse.ArgumentParser(description="检查单次划分或 K-Fold 模型输出目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-history", action="store_true")
    parser.add_argument("--cross-validation", action="store_true", help="按 K-Fold OOF 输出协议检查")
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    errors = []

    data = validate_result(out_dir, errors)
    prediction_rows = validate_predictions(out_dir, data, errors, args.cross_validation)
    n_splits = int(data.get("data", {}).get("n_splits", 0)) if data else None
    validate_history(out_dir, errors, args.require_history, args.cross_validation, n_splits)
    if args.cross_validation:
        validate_fold_files(out_dir, data, prediction_rows, errors)

    if errors:
        print(f"FAIL：{out_dir} 发现 {len(errors)} 个问题")
        for issue in errors:
            print(" -", issue)
        return 1
    mode = "K-Fold" if args.cross_validation else "Single Split"
    print(f"PASS：{mode} 输出文件结构检查通过：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
