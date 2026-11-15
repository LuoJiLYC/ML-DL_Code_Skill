---
name: ml-dl-skill
description: 基于固定 Python 工程母版创建和维护 ML/DL 表格分类实验，支持 CSV、random/fixed train-test、K-Fold 交叉验证、模型专属预处理、统一评价、结果保存与多模型比较。

---

# ML/DL Code Skill

版本：v0.5.1

## 1. 职责与边界

本 Skill 是唯一任务入口和路由器，负责识别实验类型、加载规则、选择代码母版、拆解文件计划，并组织执行与验收。具体标准在 `references/`，执行步骤在 `workflows/`，规范实现在 `scripts/`，真实案例在 `assets/`。

当前支持标准表格 CSV 二分类和多分类、单次 Random / Fixed train-test、K-Fold 分类交叉验证、单模型和多模型对比及结果绘图。K-Fold 模型母版使用 `StratifiedKFold`，公共划分器也支持普通 `KFold`，使用时按 Workflow 调整实验副本。

可用模型：
- `ml_cls/`：DT、RF、GBDT、CatBoost、LightGBM、XGBoost、KNN、Logistic Regression、SVM，适用于单次划分。
- `dl_cls/`：MLP、TabPFN、EAPCR，适用于单次划分。
- `ml_cls_fold/`：已建立 CatBoost 五折规范母版；其他传统 ML 模型按同一职责扩展。
- `dl_cls_fold/`：已建立 MLP 五折规范母版；其他 DL 模型按各自训练机制扩展。

当前 K-Fold Workflow 适用于分类任务。回归、超参数搜索、原始数据清洗及模型部署尚无对应 Workflow，不得强行套用现有分类流程。用户未指定模型时，不得默认运行全部模型。

## 2. 强制加载规则（先读，后写代码）

每次创建、修改或复现实验代码，必须首先读取：

1. `references/framework_contract.md`：工程结构、模块边界、母版优先级。
2. `references/code_style.md`：Python 排版和修改边界。

确认任务为表格分类后，根据训练方式选择 Workflow：

- Random / Fixed 单次划分：读取 `workflows/single_split_classification.md`。
- K-Fold 交叉验证：读取 `workflows/kfold_classification.md`。

随后读取：

3. `references/data_rules.md`：数据输入、划分、预处理与防泄漏。
4. `references/model_rules.md`：所选模型的特征处理与超参数规则。
5. `references/output_rules.md`：指标、输出文件和结果格式。

路径相对于 Skill 根目录。路径不存在、文件不可读或内容冲突时，先报告并处理，不得跳过或声称已加载。

## 3. 工作流路由与资源选择

| 用户任务 | Workflow / 案例 |
|---|---|
| 单次 Random 分类 | `workflows/single_split_classification.md`，参考 `assets/exp1_random_split/` |
| 单次 Fixed 分类或固定测试集下比较训练规模 | `workflows/single_split_classification.md`，参考 `assets/exp2_fixed_split/` |
| K-Fold 分类交叉验证 | `workflows/kfold_classification.md`；参考 `assets/exp3_5-fold/` |
| 多模型比较 | 在相同数据划分协议下分别运行对应模型，再汇总结果 |
| 仅绘制已有结果 | 按 `references/output_rules.md`，复用 `scripts/plotting/`，不重新训练 |

生成前必须查看对应的实际模型母版、依赖的 `common/` 函数及最近似案例。`scripts/` 是 Canonical Implementation（规范母版）；`assets/` 用于参考真实流程与输出，不能照搬数据路径、超参数或实验指标。

## 4. 必须先拆解实验

先形成文件级实施计划，至少明确：
- 数据路径、标签列、类别列及训练方式。
- 模型清单及其单次划分或 Fold 模型文件路径。
- 模型专属预处理、初始化和训练入口。
- 需要复用的 `common/` 函数及其实际参数接口。
- 输出目录、绘图需求、案例参考。
- 哪些文件直接复制、哪些只修改配置、哪些必须扩展。

新实验目录根据任务类型选择模型目录：

```text
experiment/
├── data/                       # 或记录外部数据路径
├── code/
│   ├── common/
│   ├── ml_cls/                  # 单次划分 ML
│   ├── dl_cls/                  # 单次划分 DL
│   ├── ml_cls_fold/             # K-Fold ML
│   ├── dl_cls_fold/             # K-Fold DL
│   ├── plotting/
│   └── run_all.sh               # 需要批量运行时使用
├── output/
├── task.md
└── README.md
```

只复制本次任务需要的模型、公共依赖和绘图代码；按导入要求保留 `__init__.py`。单次划分模型不得放入 Fold 目录，K-Fold 模型不得放入单次划分目录。

## 5. 模型文件职责边界

- 一个传统 ML 模型 = 一个完整模型 `.py` 文件。单次划分放在 `ml_cls/`，K-Fold 放在 `ml_cls_fold/`。
- 一个 DL 模型的专属网络、预处理和实验入口归对应模型文件。单次划分放在 `dl_cls/`，K-Fold 放在 `dl_cls_fold/`。
- ML Fold 模型文件独立组织 `preprocess_data()`、`initialize_model()`、`train_model()`、`test_model()`、`main()` 和 Fold 循环。
- DL Fold 模型文件保留模型专属网络、预处理、DataLoader、模型/优化器初始化及 `main()`；普通 Epoch 训练优先复用 `common/dl_train.py`。
- `common/cross_validation/splitter.py` 只负责折索引生成、合法性检查和折级分布统计，不负责预处理、模型训练或评价。
- 不强制建立统一的 `ml_runner.py` / `dl_runner.py`。每个模型的 `main()` 负责组织自身 Fold 训练，避免集中式模型执行器削弱模型文件独立性。
- Fold 之间必须重新初始化模型。DL 还必须重新初始化优化器，使用调度器时也要每折重建。每折的预处理器只能在该折训练数据上 fit。
- OOF 预测和五折结果汇总遵守 `references/output_rules.md`。

禁止 `common/` 通过模型名称选择并执行多个模型；禁止只含转发语句的模型包装器；禁止将训练、评价和最终绘图全部塞入一个脚本。

## 6. 修改优先级与代码风格

严格按以下顺序修改：配置 → 复用现有函数 → 对应模块内局部扩展 → 新增职责单一模块。不得为新任务自由重构整个工程。新写代码遵守 `references/code_style.md`；保留未修改母版的既有风格。

## 7. 执行与验收（固定顺序）

```text
识别 Skill
  ↓
加载工程契约和代码风格
  ↓
识别 Single Split / K-Fold
  ↓
加载对应 Workflow 与 References
  ↓
生成文件级计划
  ↓
复制规范母版并修改实验副本
  ↓
validate_code.py
  ├── FAIL → 修复实验副本 → 重新检查
  └── PASS
        ↓
      运行实验
        ↓
      validate_output.py
        ├── FAIL → 排查修复 → 重新检查
        └── PASS → 交付
```

从 Skill 根目录执行下列检查。示例使用包内案例路径，执行新实验时必须换成实际路径，并在训练前执行代码检查。

```bash
python scripts/checks/validate_code.py --experiment assets/exp1_random_split
```

单次划分输出检查：

```bash
python scripts/checks/validate_output.py --output-dir assets/exp1_random_split/output/catboost
```

K-Fold 输出检查：

```bash
python scripts/checks/validate_output.py --output-dir assets/exp3_5-fold/output/catboost_5fold --cross-validation
```

DL Fold 还需追加 `--require-history`。退出码非 0 时不得继续交付。检查器不能替代人工复核数据泄漏、模型职责和实验语义。未实际执行检查或训练时，必须如实说明，不得编造通过状态或实验结果。
