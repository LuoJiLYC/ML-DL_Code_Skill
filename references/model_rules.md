# Model Rules

适用 Skill 版本：v0.5.1

## 1. 目标

本文件规定各分类模型的标准预处理方式和可配置超参数。

---

## 2. 预处理总表

| 模型 | 类别特征 | 数值特征 | 模型侧要求 |
|---|---|---|---|
| DT | OneHotEncoder | 不处理 | 无 |
| RF | OneHotEncoder | 不处理 | 无 |
| GBDT | OneHotEncoder | 不处理 | 无 |
| CatBoost | 保留类别值 | 不处理 | `cat_features` |
| LightGBM | pandas `category` | 不处理 | 类别 dtype |
| XGBoost | pandas `category` | 不处理 | `enable_categorical=True` |
| KNN | OneHotEncoder | StandardScaler | 无 |
| Logistic Regression | OneHotEncoder | StandardScaler | 无 |
| SVM | OneHotEncoder | StandardScaler | `probability=True` |
| MLP | OneHotEncoder | StandardScaler | `torch.float32` |
| TabPFN | 保留原始类别输入 | 不手动标准化 | 类别列索引 |
| EAPCR | Global ID | 分箱 + Global ID | `torch.long` |

---

## 3. 通用规则

所有需要学习参数的预处理器：

```text
fit(train)
transform(train)
transform(test)
```

禁止：

```text
fit(train + test)
fit(test)
```

用户通过：

```python
MODEL_PARAMS = {
    ...
}
```

设置模型超参数。

用户明确给出的参数优先，未设置参数使用当前模型默认值。

---

## 4. Decision Tree

常用参数：

| 参数 | 含义 |
|---|---|
| `criterion` | 节点划分评价方式 |
| `max_depth` | 最大树深度 |
| `min_samples_split` | 节点继续划分的最少样本数 |
| `min_samples_leaf` | 叶节点最少样本数 |
| `max_features` | 每次划分考虑的最大特征数 |
| `max_leaf_nodes` | 最大叶节点数 |
| `class_weight` | 类别权重 |
| `ccp_alpha` | 剪枝强度 |
| `random_state` | 随机种子 |

---

## 5. Random Forest

常用参数：

| 参数 | 含义 |
|---|---|
| `n_estimators` | 树的数量 |
| `max_depth` | 单棵树最大深度 |
| `min_samples_split` | 节点继续划分的最少样本数 |
| `min_samples_leaf` | 叶节点最少样本数 |
| `max_features` | 每次划分使用的特征数量 |
| `bootstrap` | 是否使用 Bootstrap |
| `max_samples` | 每棵树抽取的样本数或比例 |
| `class_weight` | 类别权重 |
| `n_jobs` | CPU 并行数量 |
| `random_state` | 随机种子 |

---

## 6. GBDT

常用参数：

| 参数 | 含义 |
|---|---|
| `n_estimators` | Boosting 阶段数量 |
| `learning_rate` | 每个弱学习器的贡献权重 |
| `max_depth` | 弱学习器最大深度 |
| `subsample` | 每轮使用的样本比例 |
| `min_samples_split` | 节点划分最少样本数 |
| `min_samples_leaf` | 叶节点最少样本数 |
| `max_features` | 每次划分使用的特征数量 |
| `random_state` | 随机种子 |

---

## 7. CatBoost

常用参数：

| 参数 | 含义 |
|---|---|
| `iterations` | Boosting 迭代次数 |
| `learning_rate` | 学习率 |
| `depth` | 树深度 |
| `l2_leaf_reg` | L2 正则化强度 |
| `loss_function` | 优化目标 |
| `eval_metric` | 评价指标 |
| `random_seed` | 随机种子 |
| `bagging_temperature` | Bootstrap 随机性 |
| `border_count` | 数值特征离散边界数量 |
| `class_weights` | 类别权重 |
| `task_type` | CPU/GPU |
| `devices` | GPU 设备 |
| `thread_count` | CPU 线程数 |

框架自动提供：

```text
cat_features
```

建议：

```text
allow_writing_files=False
```

避免生成额外 `catboost_info`。

---

## 8. LightGBM

常用参数：

| 参数 | 含义 |
|---|---|
| `boosting_type` | Boosting 类型 |
| `num_leaves` | 单棵树最大叶节点数 |
| `max_depth` | 最大树深度 |
| `learning_rate` | 学习率 |
| `n_estimators` | 树数量 |
| `min_child_samples` | 叶节点最少样本数 |
| `subsample` | 样本采样比例 |
| `colsample_bytree` | 特征采样比例 |
| `reg_alpha` | L1 正则化 |
| `reg_lambda` | L2 正则化 |
| `class_weight` | 类别权重 |
| `random_state` | 随机种子 |
| `n_jobs` | CPU 并行数量 |

---

## 9. XGBoost

常用参数：

| 参数 | 含义 |
|---|---|
| `n_estimators` | 树数量 |
| `learning_rate` | 学习率 |
| `max_depth` | 最大树深度 |
| `min_child_weight` | 子节点最小 Hessian 权重 |
| `gamma` | 节点继续划分所需最小增益 |
| `subsample` | 样本采样比例 |
| `colsample_bytree` | 特征采样比例 |
| `reg_alpha` | L1 正则化 |
| `reg_lambda` | L2 正则化 |
| `scale_pos_weight` | 类别不平衡权重 |
| `tree_method` | 树构建方式 |
| `random_state` | 随机种子 |
| `n_jobs` | CPU 并行数量 |

当前原生类别处理要求：

```python
enable_categorical=True
```

推荐：

```python
tree_method="hist"
```

---

## 10. KNN

常用参数：

| 参数 | 含义 |
|---|---|
| `n_neighbors` | 最近邻数量 K |
| `weights` | 邻居投票权重 |
| `algorithm` | 最近邻搜索算法 |
| `leaf_size` | KDTree/BallTree 叶节点大小 |
| `p` | Minkowski 距离指数 |
| `metric` | 距离度量 |
| `n_jobs` | 并行数量 |

---

## 11. Logistic Regression

常用参数：

| 参数 | 含义 |
|---|---|
| `C` | 正则化强度倒数 |
| `solver` | 优化算法 |
| `max_iter` | 最大迭代次数 |
| `tol` | 收敛容差 |
| `fit_intercept` | 是否使用截距 |
| `class_weight` | 类别权重 |
| `l1_ratio` | Elastic-Net 中 L1 比例 |
| `random_state` | 随机种子 |

---

## 12. SVM

常用参数：

| 参数 | 含义 |
|---|---|
| `C` | 正则化参数 |
| `kernel` | 核函数 |
| `degree` | 多项式核次数 |
| `gamma` | 核参数 |
| `coef0` | poly/sigmoid 独立项 |
| `tol` | 收敛容差 |
| `cache_size` | 核缓存大小 |
| `class_weight` | 类别权重 |
| `max_iter` | 最大迭代次数 |
| `random_state` | 随机种子 |

统一评价需要预测概率，因此必须：

```python
probability=True
```

---

## 13. MLP

模型参数：

| 参数 | 含义 |
|---|---|
| `hidden_dims` | 隐藏层维度，例如 `[128, 64]` |
| `dropout` | Dropout 概率 |

由数据自动决定：

```text
input_dim
num_classes
```

训练参数不属于 `MODEL_PARAMS`，例如：

```text
epochs
batch_size
learning_rate
optimizer
criterion
device
random_seed
```

输入：

```text
torch.float32
```

输出层返回 logits。

训练使用：

```text
CrossEntropyLoss
```

模型内部不添加 Softmax。

---

## 14. TabPFN

常用参数：

| 参数 | 含义 |
|---|---|
| `n_estimators` | 推理 ensemble 数量 |
| `device` | CPU/GPU/auto |
| `random_state` | 随机种子 |
| `fit_mode` | fit 与缓存方式 |
| `memory_saving_mode` | 内存节省方式 |
| `inference_precision` | 推理精度 |
| `softmax_temperature` | 概率温度 |
| `balance_probabilities` | 是否平衡预测概率 |

类别列索引由框架生成：

```text
categorical_features_indices
```

不执行：

```text
OneHotEncoder
StandardScaler
```

---

## 15. EAPCR

模型参数：

| 参数 | 含义 |
|---|---|
| `embed_dim` | ID Embedding 维度 |
| `conv_channels` | CNN 输出通道数 |
| `conv_kernel_size` | 卷积核大小 |
| `conv_stride` | 卷积步长 |
| `conv_padding` | 卷积 padding |
| `hidden_dim` | FC/Residual 隐藏层维度 |
| `dropout` | Dropout 概率 |

由数据自动决定：

```text
num_embeddings
num_features
num_classes
```

Encoder 参数属于预处理配置，例如：

```text
n_bins
custom_bins
```

EAPCR Encoder 必须：

```text
fit(train)
transform(train)
transform(test)
```

测试集未见值：

```text
unknown_id
```

缺失值：

```text
missing_id
```

EAPCR 输入类型：

```text
torch.long
```

---

## 16. 禁止事项

禁止：

- 使用测试集 fit OneHotEncoder
- 使用测试集 fit StandardScaler
- train/test 分别建立类别编码
- 使用测试集生成类别定义
- 使用测试集生成 EAPCR bin_edges
- 使用测试集生成 EAPCR mapping
- 对 MLP logits 先 Softmax 再送入 CrossEntropyLoss
- 未经说明修改用户模型参数


## K-Fold 模型训练要求

以下规则适用于 `ml_cls_fold/` 和 `dl_cls_fold/`，不改变各模型原有的特征处理原则。

| 模型类型 | 每折要求 |
|---|---|
| DT / RF / GBDT | 新建模型并仅在当前训练折 fit |
| CatBoost | 每折新建 `CatBoostClassifier`；类别特征列按当前折输入保持一致 |
| LightGBM / XGBoost | 每折新建分类器并独立 fit |
| KNN / LR / SVM | 每折独立拟合模型；Scaler / Encoder 仅 fit 当前训练折 |
| MLP | 每折重新初始化网络、Criterion 和 Optimizer，独立创建 DataLoader。使用 Scheduler 时也要每折重建 |
| EAPCR | 每折独立拟合 Encoder、离散化边界、ID 映射及分类网络 |
| TabPFN | 每折独立拟合/推理；预处理与类别编码不得使用验证折信息 |

### 深度学习 Fold 状态隔离

Fold 之间不得共享或继承：模型权重、Optimizer 状态、Scheduler 状态、Early Stopping 状态、训练过程缓存及已 fit 的预处理器。若使用 Early Stopping，停止轮数只能依据当前训练折内部划分的内部验证数据确定；外层验证折只用于最终折级评价。
