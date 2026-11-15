from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, confusion_matrix
import os
import json
import numpy as np
import pandas as pd



def metrics_classification(all_labels, all_preds, all_probs, class_labels=None):

    if class_labels is None:
        classes = np.unique(np.concatenate([all_labels, all_preds]))
    else:
        classes = np.asarray(class_labels)
    

    accuracy = accuracy_score(all_labels, all_preds)

    recall_per_class = recall_score(all_labels, all_preds, labels=classes, average=None, zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, labels=classes, average="macro", zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, labels=classes, average="weighted", zero_division=0)

    precision_per_class = precision_score(all_labels, all_preds, labels=classes, average=None, zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, labels=classes, average="macro", zero_division=0)
    precision_weighted = precision_score(all_labels, all_preds, labels=classes, average="weighted", zero_division=0)

    f1_per_class = f1_score(all_labels, all_preds, labels=classes, average=None, zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, labels=classes, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, labels=classes, average="weighted", zero_division=0)

    y_true_binary = np.column_stack([(all_labels == cls).astype(int) for cls in classes])
    prauc_per_class = average_precision_score(y_true_binary, all_probs, average=None)
    prauc_macro = average_precision_score(y_true_binary, all_probs, average="macro")
    prauc_weighted = average_precision_score(y_true_binary, all_probs, average="weighted")

    conf_matrix = confusion_matrix(all_labels, all_preds, labels=classes)

    metrics = {
        "accuracy": accuracy,

        "recall_per_class": recall_per_class,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,

        "precision_per_class": precision_per_class,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,

        "f1_per_class": f1_per_class,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,

        "prauc_per_class": prauc_per_class,
        "prauc_macro": prauc_macro,
        "prauc_weighted": prauc_weighted,

        "confusion_matrix": conf_matrix,
    }

    return metrics




def metrics_classification_epoch(all_labels, all_preds, class_labels=None):
    """计算训练过程中的轻量分类指标：Accuracy 和 Macro-F1。"""

    if class_labels is None:
        classes = np.unique(np.concatenate([all_labels, all_preds]))
    else:
        classes = np.asarray(class_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, labels=classes, average="macro", zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }

    return metrics




def save_result_json(
    out_dir,
    data_info,
    model_name,
    configured_params,
    effective_params,
    train_time,
    test_time,
    metrics=None,
    train_params=None,
    label_classes=None,
    history=None,
    model_structure=None,
):
    """保存机器学习或深度学习实验结果到 JSON 文件。"""

    os.makedirs(out_dir, exist_ok=True)

    result = {
        "data": data_info,
        "model": {
            "name": model_name,
            "configured_params": configured_params,
            "effective_params": effective_params,
        },
        "runtime": {
            "train_time_seconds": train_time,
            "test_time_seconds": test_time,
        },
        "output": {
            "out_dir": out_dir,
        },
    }

    if model_structure is not None:
        result["model"]["structure"] = model_structure

    if train_params is not None:
        result["training"] = train_params

    if label_classes is not None:
        result["labels"] = {
            "classes": label_classes,
        }

    if metrics is not None:
        result["metrics"] = metrics

    if history is not None:
        result["history"] = history

    result = _clean_json_data(result)

    json_path = os.path.join(out_dir, "result.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4, allow_nan=False)

    print(f"结果已保存：{json_path}")

    return json_path




def _clean_json_data(obj):
    """将 numpy 类型和 NaN 转换为标准 JSON 支持的 Python 类型。"""

    if isinstance(obj, dict):
        return {key: _clean_json_data(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json_data(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return [_clean_json_data(value) for value in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    raise TypeError(f"不支持保存到 JSON 的数据类型：{type(obj).__name__}")




def save_history_csv(history, out_dir):
    """保存训练过程 history 到 CSV。"""

    os.makedirs(out_dir, exist_ok=True)

    history_df = pd.DataFrame(history)
    csv_path = os.path.join(out_dir, "training_history.csv")
    history_df.to_csv(csv_path, index=False)

    print(f"训练历史已保存：{csv_path}")

    return csv_path
    


def save_predictions_csv(out_dir, y_true, y_pred, y_probs=None, class_labels=None):
    """保存测试集真实标签、预测标签和预测概率。"""

    os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 的样本数量不一致")

    result = {
        "y_true": y_true,
        "y_pred": y_pred,
    }

    if y_probs is not None:
        y_probs = np.asarray(y_probs)

        if y_probs.ndim != 2:
            raise ValueError("y_probs 必须是二维数组：[样本数, 类别数]")

        if y_probs.shape[0] != len(y_true):
            raise ValueError("y_probs 和 y_true 的样本数量不一致")

        if class_labels is None:
            class_labels = np.arange(y_probs.shape[1])

        if len(class_labels) != y_probs.shape[1]:
            raise ValueError("class_labels 数量与 y_probs 的类别数不一致")

        for i, label in enumerate(class_labels):
            result[f"prob_{label}"] = y_probs[:, i]

    predictions_df = pd.DataFrame(result)

    csv_path = os.path.join(out_dir, "predictions.csv")
    predictions_df.to_csv(csv_path, index=False)

    print(f"预测结果已保存：{csv_path}")

    return csv_path