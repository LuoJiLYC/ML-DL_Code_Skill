import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder



def create_classification_dataloader(x_train, x_test, y_train, y_test, batch_size=32, feature_dtype=torch.float32):
    """
    将预处理后的数据转换为 PyTorch DataLoader。
    特征：转换为 float32
    标签：使用 LabelEncoder 编码为 0 ~ num_classes-1 转换为 int64
    训练集：shuffle=True
    测试集：shuffle=False
    """


    # 标签编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(np.asarray(y_train).reshape(-1))
    y_test_encoded = label_encoder.transform(np.asarray(y_test).reshape(-1))

    # DataFrame / numpy → Tensor
    x_train_tensor = torch.tensor(x_train.to_numpy(), dtype=feature_dtype)
    x_test_tensor = torch.tensor(x_test.to_numpy(), dtype=feature_dtype)

    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    # TensorDataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_encoder

