# K-Fold Classification Workflow

适用 Skill 版本：v0.5.1

## 0. 强制执行门槛

执行顺序：加载工程契约与代码风格 → 加载本 Workflow 和数据/模型/输出规则 → 文件级计划 → 复制规范母版 → 代码检查 → 训练 → 输出检查 → 交付。前一步失败不得跳过。

从 Skill 根目录执行检查。以下命令使用包内五折案例路径，新实验应替换为实际路径。

训练前：

```bash
python scripts/checks/validate_code.py --experiment assets/exp3_5-fold
```

训练后（ML）：

```bash
python scripts/checks/validate_output.py --output-dir assets/exp3_5-fold/output/catboost_5fold --cross-validation
```

训练后（DL）：

```bash
python scripts/checks/validate_output.py --output-dir assets/exp3_5-fold/output/mlp_5fold --cross-validation --require-history
```

## 1. 目标与适用范围

对标准表格分类数据执行 K-Fold 交叉验证。默认 `n_splits=5`，默认采用 `StratifiedKFold`。支持传统 ML 和 DL 分类模型；不覆盖回归、时间序列验证、分组验证和超参数搜索。

```text
完整数据
→ 生成 K 折索引
→ Fold 1...K 独立训练与验证
→ OOF 预测
→ OOF 整体评价
→ Fold 指标 Mean / Std
→ 保存与检查
```

## 2. Step 1：确定实验配置

至少明确：

- `CSV_PATH`
- 标签列及类别特征列
- `N_SPLITS`（默认 5）
- `RANDOM_SEED_DATA`
- 是否分层（现有 CatBoost 和 MLP 母版直接向划分器传入 `stratified=True`）
- 模型及 `MODEL_PARAMS`
- DL 训练参数（如 `EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE`）
- `OUT_DIR`、是否保存每折模型

如样本存在同一患者、设备或主体分组，或存在时间先后关系，先判断是否应使用 GroupKFold、StratifiedGroupKFold 或时间划分。不得仅为方便而随机打散相关样本。

## 3. Step 2：读取数据与检查

读取完整 CSV，得到 `x`、`y`。检查数据非空、特征列存在、标签无缺失、类别定义明确。分层 K-Fold 下每个类别样本数必须不少于 `N_SPLITS`。

## 4. Step 3：生成并检查 Fold 索引

调用公共母版：

```python
from common.cross_validation.splitter import get_kfold_splits, get_split_summary

splits = get_kfold_splits(
    x=x,
    y=y,
    n_splits=N_SPLITS,
    random_seed=RANDOM_SEED_DATA,
    stratified=True,
)
split_summary = get_split_summary(splits, y)
```

使用普通 `KFold` 时，在实验副本中传入 `stratified=False`，同步将结果中的 `data.split_method` 记为 `kfold`、`model.configured_params.cross_validation.method` 记为 `KFold`。确认每个训练折包含模型拟合和预测所需的类别。

`splitter.py` 负责索引生成、合法性检查和折级分布统计。每折训练集与验证集的索引必须互斥，两者合并后覆盖全部样本。每个样本恰好进入一次验证集。

## 5. Step 4：执行 Fold 循环

每折从原始 `x`、`y` 按索引提取数据，不得从上一折已处理的数据继续切分。

### 5.1 所有模型共用的约束

1. 当前折训练数据和验证数据独立提取。
2. 所有需要学习参数的预处理器仅在当前训练折 fit，再分别 transform 训练折和验证折。
3. 当前折模型重新初始化。
4. 验证折不参与模型参数更新、预处理器拟合或训练折内采样。
5. 当前折验证预测按原始样本索引写入 OOF 缓冲区。
6. 保存当前折指标、耗时及可选模型文件。

### 5.2 传统机器学习（`ml_cls_fold/`）

每个模型文件独立实现：

```text
preprocess_data()
→ initialize_model()
→ train_model()
→ test_model()
→ metrics_classification()
→ 保存当前折结果
```

模型创建和训练由模型文件负责；`main()` 负责组织 Fold 循环。CatBoost 参考 `assets/exp3_5-fold/code/ml_cls_fold/catboost.py`，规范母版为 `scripts/ml_cls_fold/catboost.py`。

### 5.3 深度学习（`dl_cls_fold/`）

每折必须重新执行：

```text
预处理器 fit
→ Dataset / DataLoader 创建
→ 网络初始化
→ Criterion / Optimizer 初始化（如使用 Scheduler，也在此初始化）
→ Epoch 训练与验证
→ 当前折 OOF 预测
→ 保存当前折模型和训练历史
```

普通 Epoch 模型优先复用 `common/dl_train.py` 的 `run_epochs()`。不得跨折复用模型权重、Optimizer 状态、Scheduler 状态或已 fit 的预处理器。验证阶段必须处于 eval 模式且不计算梯度。

MLP 参考 `assets/exp3_5-fold/code/dl_cls_fold/mlp.py`，规范母版为 `scripts/dl_cls_fold/mlp.py`。

## 6. Step 5：OOF 预测

建立长度等于原始样本数的 OOF 缓冲区。每折只填充其验证索引对应的位置。

至少保存：

```text
sample_index
fold
y_true
y_pred
prob_<class>（模型支持概率输出时）
```

要求：

- `sample_index` 对应原始输入数据行索引。
- 每个样本恰好有一条 OOF 记录。
- `fold` 标识该样本所属验证折，取值为 1...K。
- 多分类概率列必须按全局类别顺序对齐；某折模型概率类别顺序不同，必须显式重排。
- OOF 缓冲区不得有未填充样本或重复填充。

## 7. Step 6：评价与折间汇总

分别计算：

1. **OOF 整体指标**：将所有样本的 OOF 预测合并后计算 Accuracy、Precision、Recall、F1、PR-AUC、混淆矩阵等。
2. **折级指标**：每折验证集独立计算指标，保存至 `fold_metrics.csv`。
3. **折间统计**：对折级数值指标计算 Mean 和样本标准差（Std，`ddof=1`），保存至 `fold_metrics_summary.json`。

总体混淆矩阵由完整 OOF 预测计算，其计数总和应等于样本数。Mean / Std 描述折级指标的均值和波动。

## 8. Step 7：保存结果

遵守 `references/output_rules.md`。每个模型独立输出目录，至少包括：

```text
result.json
predictions.csv
fold_metrics.csv
fold_metrics_summary.json
```

DL 额外保存 `training_history.csv`，其中每行必须带 `fold`。每折模型文件按配置保存到 `fold_1/`...`fold_K/`。

## 9. Step 8：自动检查与交付

确认训练前已通过代码检查。训练完成后，按第 0 节命令检查每个模型的实际输出目录，DL 追加 `--require-history`。若退出码非 0，先修复并重查。交付前复核数据分组和时间泄漏风险、预处理边界、模型每折重置及 OOF 类别概率对齐。

## 10. 完成条件

- K 折索引数量与配置一致，划分合法。
- 每折均完成训练和验证。
- OOF 样本覆盖完整且唯一。
- OOF 指标、折级指标和 Mean / Std 已保存。
- DL 训练历史符合字段规范。
- 代码检查和输出检查真实执行且退出码为 0。
- 未实际运行的步骤必须明确标注，禁止编造实验结果。
