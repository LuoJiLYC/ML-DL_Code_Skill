import os
import math
import numpy as np
import torch
import torch.nn as nn

from common.data import prepare_data, get_feature_columns
from common.dl_data import create_classification_dataloader
from common.dl_train import run_epochs
from common.dl_utils import set_random_seed
from common.evaluation import save_result_json, save_history_csv, save_predictions_csv
from .eapcr_encoder import EAPCREncoder


DATA_SPLIT_MODE = "random"

CSV_PATH = "../data/data_semantic_v2.csv"
TEST_SIZE = 0.2
RANDOM_SEED_DATA = 42

CSV_TRAIN_PATH = " "
CSV_TEST_PATH = " "

BATCH_SIZE = 32
EPOCHS = 100
TEST_INTERVAL = 1
LEARNING_RATE = 0.001
RANDOM_SEED_MODEL = 42

EAPCR_N_BINS = 5
EAPCR_CUSTOM_BINS = {}

MODEL_PARAMS = {
    "embed_dim": 128,
    "conv_channels": 8,
    "conv_kernel_size": 5,
    "conv_stride": 2,
    "conv_padding": 1,
    "hidden_dim": 32,
    "dropout": 0.1,
}

OUT_DIR = "../output/eapcr"





class EAPCR(nn.Module):
    """Embedding + 原始/置换关系矩阵双 CNN + Embedding 残差融合分类模型。"""

    def __init__(
    self,
    num_embeddings,
    num_features,
    num_classes,
    embed_dim=128,
    conv_channels=8,
    conv_kernel_size=5,
    conv_stride=1,
    conv_padding=1,
    hidden_dim=32,
    dropout=0.1,
):
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError("num_embeddings 必须大于 0")
        if num_features <= 0:
            raise ValueError("num_features 必须大于 0")
        if num_classes <= 1:
            raise ValueError("num_classes 必须大于 1")

        permutation_matrix, _, _, _ = self.generator_matrix_auto(num_features)
        self.register_buffer("T", torch.tensor(permutation_matrix, dtype=torch.float32))

        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.tanh = nn.Tanh()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
        )

        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
        )

        dummy = torch.zeros(1, 1, num_features, num_features)
        conv_out = self.conv(dummy)
        relation_flat_dim = conv_out.numel() * 2

        self.fc = nn.Sequential(
            nn.Linear(relation_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        residual_flat_dim = num_features * embed_dim
        self.res = nn.Sequential(
            nn.Linear(residual_flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x = x.long()
        E = self.embedding(x)

        residual_output = self.res(E.reshape(E.size(0), -1))

        A = torch.matmul(E, E.transpose(1, 2))
        PA = torch.matmul(self.T, A)
        PA = torch.matmul(PA, self.T.transpose(0, 1))

        A = self.tanh(A).unsqueeze(1)
        PA = self.tanh(PA).unsqueeze(1)

        C = self.conv(A)
        PC = self.pconv(PA)

        relation_output = torch.cat((C, PC), dim=1)
        relation_output = relation_output.reshape(relation_output.size(0), -1)
        relation_output = self.fc(relation_output)

        fusion_weight = torch.sigmoid(self.weight)
        output = fusion_weight * relation_output + (1.0 - fusion_weight) * residual_output

        return output

    @staticmethod
    def give_num(num, l, m):
        values = np.arange(num)
        values = values.reshape((l, m))
        values = np.transpose(values, (1, 0))
        return values.reshape(-1)

    @staticmethod
    def auto_generate_parameters(new_size):
        l = math.floor(math.sqrt(new_size))
        m = math.ceil(new_size / l)
        size = l * m
        return size, l, m

    @staticmethod
    def generator_matrix_auto(new_size):
        size, l, m = EAPCR.auto_generate_parameters(new_size)
        num_list = EAPCR.give_num(size, l, m)
        num_list = num_list[num_list < new_size]

        matrix = np.zeros((new_size, new_size), dtype=np.float32)
        for i, num in enumerate(num_list):
            matrix[i, num] = 1.0

        return matrix, size, l, m


def preprocess_data(x_train, x_test):
    """
    EAPCR 数据预处理：
    1. 类别特征：训练集拟合类别 → 全局 ID 映射。
    2. 连续数值特征：训练集拟合分位数分箱边界 → 全局 ID。
    3. 低基数数值特征：按离散值建立全局 ID。
    4. 测试集只使用训练集拟合好的 encoder.transform()。
    """
    categorical_columns, numerical_columns = get_feature_columns(x_train)

    encoder = EAPCREncoder(n_bins=EAPCR_N_BINS, custom_bins=EAPCR_CUSTOM_BINS)
    x_train = encoder.fit_transform(
        x_train,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )
    x_test = encoder.transform(x_test)

    preprocessor = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "encoder": encoder,
        "num_embeddings": encoder.num_embeddings,
        "num_features": x_train.shape[1],
    }

    return x_train, x_test, preprocessor


def initialize_model(num_embeddings, num_features, num_classes, model_params=None):
    if model_params is None:
        model_params = {}

    effective_params = {
        "num_embeddings": num_embeddings,
        "embed_dim": 128,
        "num_features": num_features,
        "num_classes": num_classes,
        "conv_channels": 8,
        "hidden_dim": 32,
        "dropout": 0.0,
    }
    effective_params.update(model_params)

    model = EAPCR(
        num_embeddings=effective_params["num_embeddings"],
        embed_dim=effective_params["embed_dim"],
        num_features=effective_params["num_features"],
        num_classes=effective_params["num_classes"],
        conv_channels=effective_params["conv_channels"],
        hidden_dim=effective_params["hidden_dim"],
        dropout=effective_params["dropout"],
    )

    return model, effective_params


def main():
    set_random_seed(RANDOM_SEED_MODEL)

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

    print(f"ID 化后训练特征维度：{x_train.shape}")
    print(f"ID 化后测试特征维度：{x_test.shape}")
    print(f"Embedding 词表大小：{preprocessor['num_embeddings']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    encoder_path = os.path.join(OUT_DIR, "eapcr_encoder.pkl")
    preprocessor["encoder"].save(encoder_path)

    train_loader, test_loader, label_encoder = create_classification_dataloader(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        batch_size=BATCH_SIZE,
        feature_dtype=torch.long,
    )

    num_classes = len(label_encoder.classes_)
    class_labels = np.arange(num_classes)

    model, effective_model_params = initialize_model(
        num_embeddings=preprocessor["num_embeddings"],
        num_features=preprocessor["num_features"],
        num_classes=num_classes,
        model_params=MODEL_PARAMS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    data_info = dict(data_info)
    data_info["preprocessing"] = preprocessor["encoder"].get_summary()

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
    print(f"最终融合权重 W：{torch.sigmoid(model.weight).item():.4f}")
    print("最终测试指标：")
    print(final_metrics)


if __name__ == "__main__":
    

    
    main()
