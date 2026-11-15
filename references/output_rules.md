# Output Rules

适用 Skill 版本：v0.5.1（支持 Single Split / K-Fold）

## 1. 目标

规定分类实验的统一输出目录、输出文件和字段要求。

所有模型尽量遵循同一输出结构；**现有 `scripts/common/evaluation.py` 的实际保存接口为准，本文不要求修改母版。**

---

## 2. 输出目录

每个实验或模型使用独立目录。

推荐：

```text
output/
└── <experiment_or_model_name>/
```

例如：

```text
output/
├── rf/
├── catboost/
├── mlp/
└── eapcr/
```

---

## 3. 必须输出

所有分类实验至少生成：

```text
result.json
predictions.csv
```

具有 epoch 训练过程的模型额外生成：

```text
training_history.csv
```

需要后续复用专属预处理器时，可额外保存：

```text
*.pkl
```

例如：

```text
eapcr_encoder.pkl
```

---

## 4. result.json

用于保存一次实验的核心信息。

至少应能够表达：

```text
数据来源与划分方式
模型名称
模型参数
训练/测试耗时
类别信息
评价指标
输出目录
```

推荐结构：

```json
{
  "data": {},
  "model": {},
  "runtime": {},
  "output": {},
  "metrics": {}
}
```

### model

至少包括：

```json
{
  "name": "model_name",
  "configured_params": {},
  "effective_params": {}
}
```

其中：

```text
configured_params
```

表示用户明确设置的参数。

```text
effective_params
```

表示模型最终实际使用的参数。

### runtime

统一使用秒：

```json
{
  "train_time_seconds": 0.0,
  "test_time_seconds": 0.0
}
```

### labels（可选）

`save_result_json()` 的 `label_classes` 参数为可选参数：**仅传入时才会写入 `labels`**。当前部分 ML 模型（如 CatBoost、TabPFN）的母版调用未传入该参数，因此没有 `labels` 字段是现有实现允许的情况；不得仅因缺少该字段就判定失败。

如保存此字段，类别顺序为：

```json
{
  "classes": [0, 1]
}
```

当 `labels.classes` 存在时，预测概率列顺序必须与该顺序一致；未保存此字段时，参考 `predictions.csv` 中 `prob_<class>` 的列顺序与模型 `predict_proba()` 对应类别顺序。

### metrics

至少包含：

```text
accuracy
recall_per_class
recall_macro
recall_weighted
precision_per_class
precision_macro
precision_weighted
f1_per_class
f1_macro
f1_weighted
prauc_per_class
prauc_macro
prauc_weighted
confusion_matrix
```

所有模型必须使用统一字段名称。

`prauc_per_class`、`prauc_macro` 和 `prauc_weighted` 由 `average_precision_score()` 计算，分别记录逐类、宏平均和加权平均的 Average Precision（AP）。

---

## 5. predictions.csv

至少包含：

```text
y_true
y_pred
```

如果模型支持概率输出，还应包含：

```text
prob_<class>
```

例如：

```text
y_true,y_pred,prob_0,prob_1
0,0,0.91,0.09
1,1,0.14,0.86
```

要求：

- 行数等于测试集样本数
- 概率列数量等于类别数量
- 如果 `labels.classes` 存在，概率列顺序应与之对应
- 如果 `labels.classes` 不存在，不应误判失败；可根据 `prob_<class>` 表头确认概率列及类别，并核对原模型的类别顺序

---

## 6. training_history.csv

适用于具有 epoch 训练过程的模型。

至少包含：

```text
epoch
train_loss
train_accuracy
train_f1_macro
test_loss
test_accuracy
test_f1_macro
```

这些列名直接来自 `scripts/common/dl_train.py` 中 `run_epochs()` 生成的 history，`save_history_csv()` 仅将其写入 CSV。

如果某个 epoch 未执行测试，对应 `test_loss`、`test_accuracy`、`test_f1_macro` 为空（由 `np.nan` 保存为 CSV 空值）。**不要把 `train_f1_macro/test_f1_macro` 改成 `train_f1/test_f1`，也不要为遵守旧文档重写训练母版。**

---

## 7. 专属预处理器

如果某个模型的预处理规则需要在未来预测中复用，可以保存：

```text
*.pkl
```

该文件必须是训练阶段 fit 后的对象。

新数据只能：

```text
load
→ transform(new_data)
```

不得重新 fit。

---

## 8. 绘图输出

绘图模块必须与训练模块分离。

绘图只读取已有结果，例如：

```text
result.json
training_history.csv
summary.csv
```

推荐保存到：

```text
output_xxx/
└── figures/
```

推荐格式：

```text
PNG
PDF
SVG
```

---

## 9. 多实验汇总

当任务包含多个模型、多个训练规模或多个独立运行时，可以额外生成：

```text
summary.csv
```

例如：

```text
model,accuracy
RF,0.81
CatBoost,0.84
MLP,0.82
```

或：

```text
model,2000,4000,6000
CatBoost,0.75,0.80,0.83
MLP,0.72,0.78,0.81
```

汇总数据必须来源于各实验真实 `result.json`，不得手工编造。

---

## 10. 完成判定

基础实验至少检查：

```text
result.json exists
predictions.csv exists
```

具有 epoch 训练过程：

```text
training_history.csv exists
```

如果用户要求绘图：

```text
对应图像 exists
```

如果用户要求多实验比较：

```text
summary.csv 或等价汇总结果 exists
```

任一必要输出缺失时，任务不得标记为完成。

---

## 11. 失败条件

以下情况视为输出失败：

- JSON 无法解析
- 必须文件不存在
- 输出文件为空
- predictions 行数与测试集不一致
- 概率列数量与类别数量不一致（类别数量可由可选 `labels.classes` 或 `prob_` 列确定）
- 用户要求的图没有生成
- 汇总结果无法追溯到各实验结果


## 12. K-Fold 输出协议

K-Fold 实验每个模型使用独立的 `OUT_DIR`。以下目录名以五折为例。

```text
<model>_5fold/
├── result.json
├── predictions.csv
├── fold_metrics.csv
├── fold_metrics_summary.json
├── training_history.csv       # DL 必须
├── fold_1/                    # 按配置保存模型时
├── fold_2/
├── ...
└── fold_K/
```

### 12.1 result.json

沿用现有 `save_result_json()` 的顶层结构：`data`、`model`、`runtime`、`output`、`metrics`，如传入标签则包含 `labels`。`data` 至少记录：

```json
{
  "split_method": "stratified_kfold",
  "n_splits": 5,
  "random_seed": 42,
  "n_samples": 1000,
  "n_features": 20,
  "fold_summary": []
}
```

`metrics` 记录用全部 OOF 预测计算的总体指标。总体混淆矩阵计数之和必须等于 OOF 样本数。

### 12.2 predictions.csv（OOF）

必须包含：

```text
sample_index,fold,y_true,y_pred
```

模型支持概率输出时，还应包含按全局类别顺序排列的 `prob_<class>` 列。要求：

- 行数等于 `data.n_samples`。
- `sample_index` 唯一且完整覆盖 `0...n_samples-1`。
- `fold` 取值为 `1...n_splits`。
- 每个样本仅有一条 OOF 预测。
- 概率列顺序与 `labels.classes` 一致；若 `labels` 不存在，必须与模型全局类别顺序一致。

### 12.3 fold_metrics.csv

每行代表一折，必须有 `fold`、`train_samples`、`val_samples`，并保存可用的 Accuracy、Precision、Recall、F1、PR-AUC 和训练/测试耗时。折编号必须完整覆盖 `1...n_splits`。

### 12.4 fold_metrics_summary.json

保存折级数值指标的均值和样本标准差（Std，`ddof=1`）。`result.json` 单独记录 OOF 总体指标。

### 12.5 training_history.csv（DL）

单次划分沿用第 6 节字段。K-Fold DL 在相同字段前增加 `fold`：

```text
fold,epoch,train_loss,train_accuracy,train_f1_macro,test_loss,test_accuracy,test_f1_macro
```

每行必须标明所属折。不得把五折历史简单拼接后丢失 fold 信息。

### 12.6 每折模型

当 `SAVE_FOLD_MODELS=True` 时，在 `fold_1/` 至 `fold_K/` 保存各折模型。当前 CatBoost 保存 `catboost.cbm`，MLP 保存 `model.pth`，其中包含权重、模型参数、类别顺序和折号。

如果任务要求从原始 CSV 独立推理，还要持久化各折在训练数据上拟合的预处理器及所需的特征列顺序。当前 MLP checkpoint 未保存拟合好的 OneHotEncoder 和 StandardScaler。
