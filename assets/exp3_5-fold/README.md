# exp3_5-fold：K-Fold 分类交叉验证案例

适用 Skill 版本：v0.5.1

## 1. 实验目的

展示传统机器学习 CatBoost 与深度学习 MLP 在同一份表格分类数据上的五折交叉验证实现。该案例用于验证 K-Fold 数据划分、每折独立训练、OOF 预测和结果保存协议。

## 2. 数据

- 数据文件：`data/openml_23_cmc.csv`
- 样本数：1425
- 特征数：9
- 任务：多分类
- 标签：数据 CSV 最后一列

## 3. 交叉验证配置

- 方法：`StratifiedKFold`
- 折数：5
- `shuffle=True`
- 数据划分随机种子：42
- 每折 1140 条训练样本（80%）、285 条验证样本（20%）
- 每个样本恰好作为一次验证样本

## 4. 代码结构

```text
code/
├── common/
│   └── cross_validation/
│       └── splitter.py
├── ml_cls_fold/
│   └── catboost.py
├── dl_cls_fold/
│   └── mlp.py
└── run_all.sh
```

`splitter.py` 只生成和检查折索引。CatBoost 与 MLP 分别在自己的模型文件中组织 Fold 循环。每折独立预处理、初始化模型并完成验证。

## 5. 运行方式

激活已安装所需依赖的 Python 环境，从本案例根目录执行：

```bash
cd code
python -m ml_cls_fold.catboost
python -m dl_cls_fold.mlp
```

运行前检查两个模型文件顶部的 CSV 路径和训练参数。若使用 `bash run_all.sh`，先将脚本中的 Conda 初始化路径和环境名改为本机配置。脚本当前只启用 MLP，如需批量运行 CatBoost，取消对应命令的注释。

## 6. 输出结构

```text
output/
├── catboost_5fold/
│   ├── result.json
│   ├── predictions.csv
│   ├── fold_metrics.csv
│   ├── fold_metrics_summary.json
│   └── fold_1...fold_5/
└── mlp_5fold/
    ├── result.json
    ├── predictions.csv
    ├── fold_metrics.csv
    ├── fold_metrics_summary.json
    ├── training_history.csv
    └── fold_1...fold_5/
```

总体指标和混淆矩阵由全部 OOF 预测计算；Mean / Std 来自五个 Fold 的独立指标。

## 7. 案例边界

本案例是 K-Fold 分类的规范参考，不代表其他数据集的最佳超参数或性能保证。新增模型应遵守 Skill 的 `workflows/kfold_classification.md` 和 `references/output_rules.md`。
