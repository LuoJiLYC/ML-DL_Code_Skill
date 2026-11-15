# Single-Split Classification Workflow

适用 Skill 版本：v0.5.1

## 0. 强制的执行门槛（先检查，再训练）

本 Workflow 的执行顺序：加载 References → 生成文件级计划 → 复制母版并修改配置 → **自动检查** → 不合格则修复并复检 → 合格后训练 → **校验输出** → 交付。

新实验代码生成后、训练前，从 Skill 根目录运行代码检查。下面的命令使用包内案例路径，新实验应替换为实际路径。

```bash
python scripts/checks/validate_code.py --experiment assets/exp1_random_split
```

如果检查退出码非 0，必须修改**实验副本**并重查，不得继续训练。检查以母版原始写法为基线，不要求修改 `scripts/` 母版。

训练完成后，对每个模型的实际输出目录运行：

```bash
python scripts/checks/validate_output.py --output-dir assets/exp1_random_split/output/catboost
```

对于训练历史确实属于必需输出的模型，追加 `--require-history`。

此处的检查只能保障其覆盖的规则；无法证明 Codex 必定调用检查命令，也不能替代算法与实验语义复核。若没有真实执行检查，不得声称已验证。

---

## 1. 目标

对标准表格分类数据执行一次 train/test 分类实验。

流程：

```text
输入数据
→ 数据划分
→ 数据检查
→ 模型专属预处理
→ 模型初始化
→ 训练
→ 测试
→ 评价
→ 保存结果
→ 可选绘图
```

---

## 2. 适用范围

适用于：

- 二分类
- 多分类
- random split
- fixed train/test split
- 单模型实验
- 多模型对比实验

不负责：

```text
K-Fold
Early Stopping
Hyperparameter Search
回归
原始数据清洗
模型部署
```

---

## 3. Step 1：确定实验配置

确认：

```text
数据路径
split mode
test_size（random 时）
random_seed（random 时）
模型名称
MODEL_PARAMS
输出目录
```

如果用户没有提供模型参数，使用当前模型默认配置。

---

## 4. Step 2：读取并划分数据

调用公共数据代码：

```python
prepare_data()
```

得到：

```text
x_train
x_test
y_train
y_test
data_info
```

### random 模式

输入：

```text
一个完整 CSV
```

通过训练/测试划分生成 train 和 test。

### fixed 模式

输入：

```text
train.csv
test.csv
```

直接作为训练集和测试集。

具体规则见：

```text
references/data_rules.md
```

---

## 5. Step 3：数据检查

至少检查：

- 数据不为空
- 特征列存在
- 标签列存在
- 标签无缺失
- fixed 模式 train/test 特征列一致
- fixed 模式 train/test 特征列顺序一致
- test 标签中不存在 train 未见类别

失败时立即终止任务并报告原因。

---

## 6. Step 4：模型专属预处理

调用当前模型自己的：

```python
preprocess_data()
```

不同模型的类别特征和数值特征处理方式必须遵循：

```text
references/model_rules.md
```

所有需要学习参数的预处理器必须：

```text
fit(train)
transform(train)
transform(test)
```

不得使用测试集拟合预处理器。

---

## 7. Step 5：初始化模型

调用：

```python
initialize_model()
```

记录：

```text
configured_params
effective_params
```

由数据自动推导的参数不要求用户手动设置，例如：

```text
input_dim
num_classes
num_features
num_embeddings
```

---

## 8. Step 6：训练模型

传统机器学习：

```text
model.fit()
```

具有 epoch 训练过程的模型：

```text
DataLoader
→ Forward
→ Loss
→ Backward
→ Optimizer Step
```

训练过程中不得使用测试集更新模型参数。

---

## 9. Step 7：测试模型

在测试集上生成：

```text
y_pred
y_probs
test_time
```

预测概率的类别顺序必须与最终记录的 class labels 一致。

---

## 10. Step 8：统一评价

调用统一评价函数：

```python
metrics_classification()
```

至少计算：

```text
Accuracy
Recall
Precision
F1
PR-AUC
Confusion Matrix
```

不同模型必须使用相同评价函数和字段定义。

---

## 11. Step 9：保存结果

按照：

```text
references/output_rules.md
```

保存实验结果。

基础输出：

```text
result.json
predictions.csv
```

具有 epoch 训练过程：

```text
training_history.csv
```

需要持久化专属预处理器时可以额外保存：

```text
*.pkl
```

---

## 12. Step 10：可选绘图

只有用户要求绘图或实验本身要求对比图时执行。

绘图代码只能读取已经保存的结果文件。

例如：

```text
result.json
training_history.csv
summary.csv
```

可生成：

```text
confusion matrix
accuracy curve
loss curve
model comparison bar chart
accuracy vs training size
```

绘图与训练代码必须分离。

---

## 13. Step 11：结果检查

检查：

```text
必要文件是否存在
JSON 是否可读取
predictions 行数是否正确
概率列数量是否与类别数量一致
训练历史是否完整
要求的图是否生成
```

全部通过后：

```text
Workflow = Completed
```

否则：

```text
Workflow = Failed
```

并返回具体失败原因。

---

## 14. 多模型任务

如果用户指定多个模型：

```text
使用同一数据划分
→ 分别执行本 Workflow
→ 各自保存独立输出目录
→ 最后汇总主要指标
```

不得为了某个模型单独改变测试集。

---

## 15. 案例参考

随机划分多模型实验可参考：

```text
assets/exp1_random_split/
```

固定划分及固定测试集下的训练数据量比较可参考：

```text
assets/exp2_fixed_split/
```

该案例的训练规模汇总表为 `assets/exp2_fixed_split/output/summary/acc_compare.csv`。进入该案例的 `code/` 目录后，可执行 `python -m plotting.plot_accuracy_vs_datasize` 生成对比图。

案例用于理解结构，不作为新实验结果。
