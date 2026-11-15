import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.data import load_data_csv, get_feature_columns, fit_one_hot_encoder, transform_one_hot_encoder
from common.data import fit_standard_scaler, transform_standard_scaler
from common.dl_data import create_classification_dataloader
from common.dl_train import run_epochs
from common.dl_utils import set_random_seed
from common.evaluation import metrics_classification, save_result_json
from common.cross_validation.splitter import get_kfold_splits, get_split_summary


CSV_PATH = "../data/openml_23_cmc.csv"
N_SPLITS = 5
RANDOM_SEED_DATA = 42
RANDOM_SEED_MODEL = 42
BATCH_SIZE = 32
EPOCHS = 300
TEST_INTERVAL = 10
LEARNING_RATE = 0.00001
MODEL_PARAMS = {
    "hidden_dims": [128, 64],
    "dropout": 0.1,
}
OUT_DIR = "../output/mlp_5fold"
SAVE_FOLD_MODELS = True


class MLP(nn.Module):
    """多层感知机分类模型。"""

    def __init__(self, input_dim, num_classes, hidden_dims=(128, 64), dropout=0.0):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def preprocess_data(x_train, x_val):
    """每折独立拟合预处理器：数值特征 StandardScaler，类别特征 One-Hot。"""
    categorical_columns, numerical_columns = get_feature_columns(x_train)
    x_train = x_train.copy()
    x_val = x_val.copy()
    scaler = None
    encoder = None

    if numerical_columns:
        scaler = fit_standard_scaler(x_train, numerical_columns)
        x_train = transform_standard_scaler(x_train, scaler, numerical_columns)
        x_val = transform_standard_scaler(x_val, scaler, numerical_columns)

    if categorical_columns:
        encoder = fit_one_hot_encoder(x_train, categorical_columns)
        x_train = transform_one_hot_encoder(x_train, encoder, categorical_columns)
        x_val = transform_one_hot_encoder(x_val, encoder, categorical_columns)

    preprocessor = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "encoder": encoder,
        "scaler": scaler,
    }
    return x_train, x_val, preprocessor


def initialize_model(input_dim, num_classes, model_params=None):
    model_params = {} if model_params is None else model_params.copy()
    effective_params = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_dims": [128, 64],
        "dropout": 0.0,
    }
    effective_params.update(model_params)
    model = MLP(
        input_dim=effective_params["input_dim"],
        num_classes=effective_params["num_classes"],
        hidden_dims=effective_params["hidden_dims"],
        dropout=effective_params["dropout"],
    )
    return model, effective_params


def align_probabilities(y_probs, fold_classes, global_classes):
    """将当前折 LabelEncoder 的概率列对齐到全局类别顺序。"""
    aligned_probs = np.zeros((len(y_probs), len(global_classes)), dtype=float)
    class_to_index = {label: index for index, label in enumerate(global_classes)}
    for model_index, label in enumerate(fold_classes):
        if label not in class_to_index:
            raise ValueError(f"当前折出现未知类别：{label}")
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


def save_training_history(history_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    history_df = pd.concat(history_list, ignore_index=True)
    path = os.path.join(out_dir, "training_history.csv")
    history_df.to_csv(path, index=False)
    print(f"训练历史已保存：{path}")


def save_fold_summary(fold_metrics, out_dir):
    metrics_df = pd.DataFrame(fold_metrics)
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
    summary = {}
    for column in numeric_columns:
        if column == "fold":
            continue
        summary[f"{column}_mean"] = float(metrics_df[column].mean())
        summary[f"{column}_std"] = float(metrics_df[column].std(ddof=1))
    path = os.path.join(out_dir, "fold_metrics_summary.json")
    pd.Series(summary).to_json(path, force_ascii=False, indent=4)
    print(f"五折汇总已保存：{path}")


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
        print(
            f"Fold {item['fold']}: train={item['train_samples']}, "
            f"val={item['val_samples']}, train_classes={item['train_class_counts']}, "
            f"val_classes={item['val_class_counts']}"
        )

    oof_pred = np.empty(len(y_array), dtype=object)
    oof_probs = np.zeros((len(y_array), len(class_labels)), dtype=float)
    oof_fold = np.zeros(len(y_array), dtype=int)
    fold_metrics = []
    history_list = []
    fold_train_times = []
    fold_test_times = []
    effective_params_list = []
    model_structure = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n{'=' * 20} Fold {fold}/{N_SPLITS} {'=' * 20}")
        set_random_seed(RANDOM_SEED_MODEL + fold - 1)

        x_train = x.iloc[train_idx].copy()
        x_val = x.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        print(f"训练集：{len(x_train)}，验证集：{len(x_val)}")

        x_train, x_val, preprocessor = preprocess_data(x_train, x_val)
        print(f"预处理后训练特征维度：{x_train.shape}")

        train_loader, val_loader, label_encoder = create_classification_dataloader(
            x_train=x_train,
            x_test=x_val,
            y_train=y_train,
            y_test=y_val,
            batch_size=BATCH_SIZE,
        )

        input_dim = x_train.shape[1]
        num_classes = len(label_encoder.classes_)
        encoded_class_labels = np.arange(num_classes)
        model, effective_params = initialize_model(input_dim, num_classes, MODEL_PARAMS)
        model_structure = str(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model, history, train_time, test_time, _, final_val = run_epochs(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            test_interval=TEST_INTERVAL,
            device=device,
            class_labels=encoded_class_labels,
        )

        fold_classes = label_encoder.classes_
        y_pred_original = label_encoder.inverse_transform(final_val["y_pred"].astype(int))
        y_probs = align_probabilities(final_val["y_probs"], fold_classes, class_labels)
        y_true_original = y_val.to_numpy()
        fold_result = metrics_classification(
            y_true_original, y_pred_original, y_probs, class_labels=class_labels
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
            "train_time_seconds": float(train_time),
            "test_time_seconds": float(test_time),
        })

        oof_pred[val_idx] = y_pred_original
        oof_probs[val_idx] = y_probs
        oof_fold[val_idx] = fold

        history_df = pd.DataFrame(history)
        history_df.insert(0, "fold", fold)
        history_list.append(history_df)
        fold_train_times.append(train_time)
        fold_test_times.append(test_time)
        effective_params_list.append(effective_params)

        if SAVE_FOLD_MODELS:
            fold_dir = os.path.join(OUT_DIR, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_params": effective_params,
                "class_labels": label_encoder.classes_.tolist(),
                "fold": fold,
            }, os.path.join(fold_dir, "model.pth"))

        print(f"Fold {fold} Accuracy: {fold_result['accuracy']:.4f}")
        print(f"Fold {fold} Macro-F1: {fold_result['f1_macro']:.4f}")

    if np.any(oof_fold == 0):
        raise RuntimeError("OOF 预测不完整：存在未获得验证预测的样本")

    oof_pred = np.asarray(oof_pred.tolist())
    overall_metrics = metrics_classification(y_array, oof_pred, oof_probs, class_labels=class_labels)

    save_fold_metrics(fold_metrics, OUT_DIR)
    save_oof_predictions(y_array, oof_pred, oof_probs, oof_fold, class_labels, OUT_DIR)
    save_training_history(history_list, OUT_DIR)
    save_fold_summary(fold_metrics, OUT_DIR)

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
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "test_interval": TEST_INTERVAL,
            "learning_rate": LEARNING_RATE,
            "random_seed_model": RANDOM_SEED_MODEL,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "device": str(device),
        },
        "save_fold_models": SAVE_FOLD_MODELS,
    }

    save_result_json(
        out_dir=OUT_DIR,
        data_info=data_info,
        model_name="MLP",
        configured_params=configured_params,
        effective_params=effective_params_list,
        model_structure=model_structure,
        train_time=sum(fold_train_times),
        test_time=sum(fold_test_times),
        metrics=overall_metrics,
        label_classes=class_labels.tolist(),
    )

    print("\nMLP 五折交叉验证完成")
    print(f"OOF Accuracy：{overall_metrics['accuracy']:.4f}")
    print(f"OOF Macro-F1：{overall_metrics['f1_macro']:.4f}")
    print(f"OOF Macro PR-AUC：{overall_metrics['prauc_macro']:.4f}")
    print(f"结果目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
