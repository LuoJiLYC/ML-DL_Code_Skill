import os
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from common.data import load_data_csv, get_feature_columns, normalize_categorical_columns
from common.evaluation import metrics_classification, save_result_json
from common.cross_validation.splitter import get_kfold_splits, get_split_summary


CSV_PATH = "../data/openml_23_cmc.csv"
N_SPLITS = 5
RANDOM_SEED_DATA = 42
MODEL_PARAMS = {
    "random_seed": 42,
}
OUT_DIR = "../output/catboost_5fold"
SAVE_FOLD_MODELS = True


def preprocess_data(x_train, x_val):
    """CatBoost：类别特征转为字符串，数值特征保持原样。"""
    categorical_columns, numerical_columns = get_feature_columns(x_train)
    x_train = normalize_categorical_columns(x_train, categorical_columns)
    x_val = normalize_categorical_columns(x_val, categorical_columns)

    preprocessor = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "encoder": None,
        "scaler": None,
    }
    return x_train, x_val, preprocessor


def initialize_model(model_params=None, preprocessor=None):
    """初始化当前折的 CatBoost 模型。"""
    model_params = {} if model_params is None else model_params.copy()
    effective_params = {
        "cat_features": preprocessor["categorical_columns"],
        "verbose": 0,
        "allow_writing_files": False,
    }
    effective_params.update(model_params)
    return CatBoostClassifier(**effective_params)


def train_model(model, x_train, y_train):
    """训练当前折模型。"""
    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_time
    return model, train_time


def test_model(model, x_val):
    """预测当前折验证集。"""
    start_time = time.perf_counter()
    y_pred = np.asarray(model.predict(x_val)).reshape(-1)
    y_probs = np.asarray(model.predict_proba(x_val))
    test_time = time.perf_counter() - start_time
    return y_pred, y_probs, test_time


def align_probabilities(y_probs, model_classes, class_labels):
    """将模型概率列对齐到全局类别顺序。"""
    aligned_probs = np.zeros((len(y_probs), len(class_labels)), dtype=float)
    class_to_index = {label: index for index, label in enumerate(class_labels)}

    for model_index, label in enumerate(model_classes):
        if label not in class_to_index:
            raise ValueError(f"模型输出了未知类别：{label}")
        aligned_probs[:, class_to_index[label]] = y_probs[:, model_index]

    return aligned_probs


def save_fold_metrics(fold_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fold_metrics.csv")
    pd.DataFrame(fold_metrics).to_csv(path, index=False)
    print(f"五折指标已保存：{path}")


def save_oof_predictions(y_true, y_pred, y_probs, oof_fold, class_labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    result = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "fold": oof_fold,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    for index, label in enumerate(class_labels):
        result[f"prob_{label}"] = y_probs[:, index]

    path = os.path.join(out_dir, "predictions.csv")
    result.to_csv(path, index=False)
    print(f"OOF 预测已保存：{path}")


def main():
    x, y = load_data_csv(CSV_PATH)
    y_array = y.to_numpy()
    class_labels = np.unique(y_array)

    print(f"数据集：{CSV_PATH}")
    print(f"样本数：{len(x)}，特征数：{x.shape[1]}，类别数：{len(class_labels)}")


    splits = get_kfold_splits(
        x=x,
        y=y_array,
        n_splits=N_SPLITS,
        random_seed=RANDOM_SEED_DATA,
        stratified=True,
    )
    split_summary = get_split_summary(splits, y_array)
    print("五折划分情况：")
    for item in split_summary:
        print(f"Fold {item['fold']}: train={item['train_samples']}, val={item['val_samples']}, "
              f"train_classes={item['train_class_counts']}, val_classes={item['val_class_counts']}")

    oof_pred = np.empty(len(y_array), dtype=object)
    oof_probs = np.zeros((len(y_array), len(class_labels)), dtype=float)
    oof_fold = np.zeros(len(y_array), dtype=int)
    fold_metrics = []
    fold_train_times = []
    fold_test_times = []
    effective_params = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n{'=' * 20} Fold {fold}/{N_SPLITS} {'=' * 20}")
        x_train, x_val = x.iloc[train_idx].copy(), x.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"训练集：{len(x_train)}，验证集：{len(x_val)}")
        x_train, x_val, preprocessor = preprocess_data(x_train, x_val)

        fold_params = MODEL_PARAMS.copy()
        fold_params["random_seed"] = MODEL_PARAMS.get("random_seed", 42) + fold - 1
        model = initialize_model(fold_params, preprocessor)

        model, train_time = train_model(model, x_train, y_train)
        y_pred, y_probs, test_time = test_model(model, x_val)
        y_probs = align_probabilities(y_probs, model.classes_, class_labels)

        fold_result = metrics_classification(
            y_val.to_numpy(), y_pred, y_probs, class_labels=class_labels
        )
        fold_metrics.append({
            "fold": fold,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "accuracy": float(fold_result["accuracy"]),
            "precision_macro": float(fold_result["precision_macro"]),
            "recall_macro": float(fold_result["recall_macro"]),
            "f1_macro": float(fold_result["f1_macro"]),
            "precision_weighted": float(fold_result["precision_weighted"]),
            "recall_weighted": float(fold_result["recall_weighted"]),
            "f1_weighted": float(fold_result["f1_weighted"]),
            "prauc_macro": float(fold_result["prauc_macro"]),
            "train_time_seconds": train_time,
            "test_time_seconds": test_time,
        })

        oof_pred[val_idx] = y_pred
        oof_probs[val_idx] = y_probs
        oof_fold[val_idx] = fold
        fold_train_times.append(train_time)
        fold_test_times.append(test_time)
        effective_params.append(model.get_all_params())

        if SAVE_FOLD_MODELS:
            fold_dir = os.path.join(OUT_DIR, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            model.save_model(os.path.join(fold_dir, "catboost.cbm"))

        print(f"Fold {fold} Accuracy: {fold_result['accuracy']:.4f}")
        print(f"Fold {fold} Macro-F1: {fold_result['f1_macro']:.4f}")

    # 全部 OOF 预测合并后，计算整体指标
    oof_pred = np.asarray(oof_pred.tolist())
    overall_metrics = metrics_classification(
        y_array, oof_pred, oof_probs, class_labels=class_labels
    )

    save_fold_metrics(fold_metrics, OUT_DIR)
    save_oof_predictions(y_array, oof_pred, oof_probs, oof_fold, class_labels, OUT_DIR)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        f"{column}_mean": float(fold_metrics_df[column].mean())
        for column in fold_metrics_df.select_dtypes(include=[np.number]).columns
        if column != "fold"
    }
    summary.update({
        f"{column}_std": float(fold_metrics_df[column].std(ddof=1))
        for column in fold_metrics_df.select_dtypes(include=[np.number]).columns
        if column != "fold"
    })

    data_info = {
        "split_method": "stratified_kfold",
        "csv_path": CSV_PATH,
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED_DATA,
        "n_samples": len(x),
        "n_features": x.shape[1],
        "fold_summary": split_summary,
    }
    configured_params = {
        "model_params": MODEL_PARAMS,
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "random_seed": RANDOM_SEED_DATA,
            "shuffle": True,
        },
        "save_fold_models": SAVE_FOLD_MODELS,
    }

    save_result_json(
        out_dir=OUT_DIR,
        data_info=data_info,
        model_name="CatBoostClassifier",
        configured_params=configured_params,
        effective_params=effective_params,
        train_time=sum(fold_train_times),
        test_time=sum(fold_test_times),
        metrics=overall_metrics,
        label_classes=class_labels.tolist(),
    )

    summary_path = os.path.join(OUT_DIR, "fold_metrics_summary.json")
    pd.Series(summary).to_json(summary_path, force_ascii=False, indent=4)

    print("\n五折交叉验证完成")
    print(f"OOF Accuracy：{overall_metrics['accuracy']:.4f}")
    print(f"OOF Macro-F1：{overall_metrics['f1_macro']:.4f}")
    print(f"OOF Macro PR-AUC：{overall_metrics['prauc_macro']:.4f}")
    print(f"结果目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
