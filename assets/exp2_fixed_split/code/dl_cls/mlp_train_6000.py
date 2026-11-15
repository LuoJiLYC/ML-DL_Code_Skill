import numpy as np
import torch
import torch.nn as nn

from common.data import prepare_data, get_feature_columns, fit_one_hot_encoder, transform_one_hot_encoder
from common.data import fit_standard_scaler, transform_standard_scaler
from common.dl_data import create_classification_dataloader
from common.dl_train import run_epochs
from common.evaluation import save_result_json, save_history_csv, save_predictions_csv
from common.dl_utils import set_random_seed







DATA_SPLIT_MODE = "fixed"

# random 模式
CSV_PATH = "../data/data_semantic_v2.csv"
TEST_SIZE = 0.2
RANDOM_SEED_DATA = 42

# fixed 模式
CSV_TRAIN_PATH = "../data/train_6000.csv"
CSV_TEST_PATH = "../data/test_2000.csv"

BATCH_SIZE = 32
EPOCHS = 100
TEST_INTERVAL = 1
LEARNING_RATE = 0.00001
RANDOM_SEED_MODEL = 42

MODEL_PARAMS = {
    "hidden_dims": [128, 64],
    "dropout": 0.1,
}

OUT_DIR = "../output/mlp_train_6000"







class MLP(nn.Module):
    """多层感知机分类模型。"""

    def __init__( self, input_dim, num_classes, hidden_dims=(128, 64), dropout=0.0):
        super().__init__()

        layers = []

        current_dim = input_dim

        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0: layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(current_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



def initialize_model(input_dim, num_classes, model_params=None):
    if model_params is None:
        model_params = {}

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




def preprocess_data(x_train, x_test):
    """
    MLP 数据预处理：
    类别特征：OneHotEncoder
    数值特征：StandardScaler
    注意：encoder 和 scaler 都只在训练集上拟合。
    """

    #获取类别特征和数值特征
    categorical_columns, numerical_columns = get_feature_columns(x_train)

    x_train = x_train.copy()
    x_test = x_test.copy()

    scaler = None
    encoder = None

    #数值特征 StandardScaler
    if numerical_columns:
        scaler = fit_standard_scaler(x_train, numerical_columns)
        x_train = transform_standard_scaler(x_train, scaler, numerical_columns)
        x_test = transform_standard_scaler(x_test, scaler, numerical_columns)


    #类别特征 One-Hot
    if categorical_columns:
        encoder = fit_one_hot_encoder(x_train, categorical_columns)
        x_train = transform_one_hot_encoder(x_train, encoder, categorical_columns)
        x_test = transform_one_hot_encoder(x_test, encoder, categorical_columns)


    #保存预处理器信息
    preprocessor = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "encoder": encoder,
        "scaler": scaler,
    }

    return x_train, x_test, preprocessor




def main():
    set_random_seed(RANDOM_SEED_MODEL)
    
    # 数据读取
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
    # 数据预处理
    x_train, x_test, preprocessor = preprocess_data(x_train, x_test)

    print(f"预处理后训练特征维度：{x_train.shape}")
    print(f"预处理后测试特征维度：{x_test.shape}")

    # DataLoader
    train_loader, test_loader, label_encoder = create_classification_dataloader(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        batch_size=BATCH_SIZE,
    )
    
    # 自动确定输入维数和类别数
    input_dim = x_train.shape[1]
    num_classes = len(label_encoder.classes_)
    class_labels = np.arange(num_classes)
    
    # 初始化模型
    model, effective_model_params = initialize_model(
        input_dim=input_dim,
        num_classes=num_classes,
        model_params=MODEL_PARAMS,
    )

    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练
    model, history, train_time, test_time, final_metrics, final_test = run_epochs(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        test_interval=TEST_INTERVAL,
        device=device,
        class_labels=class_labels,
    )

    # 保存结果
    train_params = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "test_interval": TEST_INTERVAL,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED_MODEL,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "device": str(device),
    }
    model_structure = str(model)
    

    save_result_json(
        out_dir=OUT_DIR,
        data_info=data_info,
        model_name=model.__class__.__name__,
        configured_params=MODEL_PARAMS,
        effective_params=effective_model_params,
        model_structure=str(model),
        train_time=train_time,
        test_time=test_time,
        metrics=final_metrics,
        train_params=train_params,
        label_classes=label_encoder.classes_,
        history=history,
    )

    save_history_csv(history=history, out_dir=OUT_DIR)

    y_true_original = label_encoder.inverse_transform(final_test["y_true"].astype(int))
    y_pred_original = label_encoder.inverse_transform(final_test["y_pred"].astype(int))

    save_predictions_csv(
        out_dir=OUT_DIR,
        y_true=y_true_original,
        y_pred=y_pred_original,
        y_probs=final_test["y_probs"],
        class_labels=label_encoder.classes_,
    )

    print(f"总训练时间：{train_time:.2f} 秒")
    print(f"总测试时间：{test_time:.2f} 秒")
    print("最终测试指标：")
    print(final_metrics)


if __name__ == "__main__":
    main()