from sklearn.linear_model import LogisticRegression
import time
import numpy as np

from common.data import get_feature_columns, fit_one_hot_encoder, transform_one_hot_encoder, fit_standard_scaler, transform_standard_scaler, prepare_data
from common.evaluation import metrics_classification, save_result_json, save_predictions_csv


DATA_SPLIT_MODE = "random"

CSV_PATH = "../data/data_semantic_v2.csv"
TEST_SIZE = 0.2
RANDOM_SEED_DATA = 42

CSV_TRAIN_PATH = " "
CSV_TEST_PATH = " "

MODEL_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
}

OUT_DIR = "../output/logistic_regression"


def preprocess_data(x_train, x_test):
    """
    Logistic Regression：
    类别特征 One-Hot，
    数值特征 StandardScaler。
    """
    categorical_columns, numerical_columns = get_feature_columns(x_train)

    scaler = fit_standard_scaler(x_train, numerical_columns)

    x_train = transform_standard_scaler(x_train, scaler, numerical_columns)
    x_test = transform_standard_scaler(x_test, scaler, numerical_columns)

    encoder = fit_one_hot_encoder(x_train, categorical_columns)

    x_train = transform_one_hot_encoder(x_train, encoder, categorical_columns)
    x_test = transform_one_hot_encoder(x_test, encoder, categorical_columns)

    preprocessor = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "encoder": encoder,
        "scaler": scaler,
    }

    return x_train, x_test, preprocessor


def initialize_model(model_params=None, preprocessor=None):
    """定义 Logistic Regression 模型。"""
    if model_params is None:
        model_params = {}

    return LogisticRegression(**model_params)


def train_model(model, x_train, y_train):
    """训练 Logistic Regression 模型。"""
    print("开始训练 Logistic Regression...")
    start_time = time.perf_counter()

    model.fit(x_train, y_train)

    train_time = time.perf_counter() - start_time
    print(f"训练完成，耗时：{train_time:.2f} 秒")

    return model, train_time


def test_model(model, x_test):
    """测试 Logistic Regression 模型。"""
    print("开始测试 Logistic Regression...")
    start_time = time.perf_counter()

    y_pred = np.asarray(model.predict(x_test)).reshape(-1)
    y_probs = np.asarray(model.predict_proba(x_test))

    test_time = time.perf_counter() - start_time
    print(f"测试完成，耗时：{test_time:.2f} 秒")

    return y_pred, y_probs, test_time


def main():
    x_train, x_test, y_train, y_test, data_info = prepare_data(
        split_mode=DATA_SPLIT_MODE,
        csv_path=CSV_PATH,
        csv_train_path=CSV_TRAIN_PATH,
        csv_test_path=CSV_TEST_PATH,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED_DATA,
    )

    print(f"训练集：{x_train.shape[0]} 个样本")
    print(f"测试集：{x_test.shape[0]} 个样本")

    x_train, x_test, preprocessor = preprocess_data(x_train, x_test)

    print(f"预处理后训练特征维度：{x_train.shape}")
    print(f"预处理后测试特征维度：{x_test.shape}")

    model = initialize_model(MODEL_PARAMS, preprocessor)

    model, train_time = train_model(model, x_train, y_train)
    y_pred, y_probs, test_time = test_model(model, x_test)

    metrics = metrics_classification(y_test, y_pred, y_probs, class_labels=model.classes_)

    save_result_json(
        out_dir=OUT_DIR,
        data_info=data_info,
        model_name=model.__class__.__name__,
        configured_params=MODEL_PARAMS,
        effective_params=model.get_params(),
        train_time=train_time,
        test_time=test_time,
        metrics=metrics,
    )

    save_predictions_csv(
        out_dir=OUT_DIR,
        y_true=y_test,
        y_pred=y_pred,
        y_probs=y_probs,
        class_labels=model.classes_,
    )


if __name__ == "__main__":
    main()