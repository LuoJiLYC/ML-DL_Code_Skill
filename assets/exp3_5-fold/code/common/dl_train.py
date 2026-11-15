import time
import numpy as np
import torch

from common.evaluation import metrics_classification, metrics_classification_epoch




def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch，返回 loss、真实标签、预测标签和训练时间。"""

    model.train()
    start_time = time.perf_counter()

    total_loss = 0.0
    total_samples = 0
    y_true_list = []
    y_pred_list = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        y_true_list.append(y_batch.detach().cpu().numpy())
        y_pred_list.append(preds.detach().cpu().numpy())

    train_loss = total_loss / total_samples
    train_time = time.perf_counter() - start_time

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    return train_loss, y_true, y_pred, train_time




def test_one_epoch(model, test_loader, criterion, device):
    """测试一个 epoch，返回 loss、真实标签、预测标签、预测概率和测试时间。"""

    model.eval()
    start_time = time.perf_counter()

    total_loss = 0.0
    total_samples = 0
    y_true_list = []
    y_pred_list = []
    y_probs_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            y_probs_list.append(probs.cpu().numpy())

    test_loss = total_loss / total_samples
    test_time = time.perf_counter() - start_time

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_probs = np.concatenate(y_probs_list)

    return test_loss, y_true, y_pred, y_probs, test_time




def run_epochs(model, train_loader, test_loader, criterion, optimizer, epochs, test_interval, device, class_labels=None):
    """
    执行分类模型的多轮训练和测试。

    每个 epoch：计算训练集 Loss、Accuracy、Macro-F1。
    每隔 test_interval 个 epoch：在测试集上计算 Loss、Accuracy、Macro-F1。
    最后一个 epoch：
        强制执行测试，并调用 metrics_classification()
        计算完整分类指标。

    本函数只负责单次训练集/测试集训练流程，
    不负责验证集、交叉验证。
    """

    if epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if test_interval <= 0:
        raise ValueError("test_interval 必须大于 0")

    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_f1_macro": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_f1_macro": [],
    }

    total_train_time = 0.0
    total_test_time = 0.0
    final_metrics = None
    final_test = None

    for epoch in range(1, epochs + 1):
        train_loss, train_y_true, train_y_pred, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        if class_labels is None:
            class_labels = np.unique(train_y_true)

        train_metrics = metrics_classification_epoch(
            train_y_true, train_y_pred, class_labels=class_labels
        )

        total_train_time += train_time

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])

        if epoch % test_interval == 0 or epoch == epochs:
            test_loss, test_y_true, test_y_pred, test_y_probs, test_time = test_one_epoch(
                model, test_loader, criterion, device
            )

            test_metrics = metrics_classification_epoch(
                test_y_true, test_y_pred, class_labels=class_labels
            )

            total_test_time += test_time

            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_metrics["accuracy"])
            history["test_f1_macro"].append(test_metrics["f1_macro"])

            if epoch == epochs:
                final_metrics = metrics_classification(
                    test_y_true, test_y_pred, test_y_probs, class_labels=class_labels
                )

                final_test = {
                    "test_loss": test_loss,
                    "y_true": test_y_true,
                    "y_pred": test_y_pred,
                    "y_probs": test_y_probs,
                }

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Train Acc: {train_metrics['accuracy']:.4f} "
                f"Train F1: {train_metrics['f1_macro']:.4f} "
                f"Test Loss: {test_loss:.6f} "
                f"Test Acc: {test_metrics['accuracy']:.4f} "
                f"Test F1: {test_metrics['f1_macro']:.4f}"
            )

        else:
            history["test_loss"].append(np.nan)
            history["test_accuracy"].append(np.nan)
            history["test_f1_macro"].append(np.nan)

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Train Acc: {train_metrics['accuracy']:.4f} "
                f"Train F1: {train_metrics['f1_macro']:.4f}"
            )

    return model, history, total_train_time, total_test_time, final_metrics, final_test