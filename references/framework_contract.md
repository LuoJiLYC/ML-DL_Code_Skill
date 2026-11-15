# Framework Contract

适用 Skill 版本：v0.5.1
适用范围：使用 ML/DL Code Skill 创建、复现、修改和扩展实验代码。

## 1. 文件职责

本文是本 Skill 的**工程结构契约**，规定：

1. 实验项目必须采用什么目录结构。
2. 每类文件和模块分别负责什么。
3. 哪些内容固定（FIXED）、可配置（CONFIGURABLE）、可扩展（EXTENSIBLE）。
4. 如何基于 `scripts/` 母版创建新实验。
5. 何时允许修改框架，以及如何检查结构是否合格。

本文**不负责**规定详细代码排版、模型超参数表、具体数据处理算法和结果字段；这些分别由以下文件负责：

- `references/code_style.md`：行长、换行、空行、命名、注释及函数格式。
- `references/data_rules.md`：数据输入、划分、编码、防泄漏。
- `references/model_rules.md`：各模型特征处理和超参数。
- `references/output_rules.md`：指标与输出文件协议。
- `workflows/single_split_classification.md`：单次划分分类实验的执行顺序。
- `workflows/kfold_classification.md`：K-Fold 分类交叉验证的执行顺序。

**执行原则：先遵守本契约，再选择代码实现。**

## 2. 三级修改权限

| 等级           | 含义                           | 默认行为                           |
| -------------- | ------------------------------ | ---------------------------------- |
| `FIXED`        | 框架结构及公共约定             | 普通实验不得修改                   |
| `CONFIGURABLE` | 每次实验的输入、参数与模型选择 | 可按任务修改                       |
| `EXTENSIBLE`   | 母版不足时新增的任务能力       | 允许最小范围扩展，必须守住模块边界 |

### 2.1 FIXED：默认保持不变

- 实验项目的目录层级与模块职责。
- **每个传统 ML 模型独立拥有一个完整的模型 `.py` 文件**。
- 公共工具与模型专属逻辑分离。
- 模型、数据、训练、评价、保存、绘图的责任边界。
- 既有公共函数的接口与统一评价、输出协议。
- 训练集拟合预处理器、测试集只转换的原则。
- `main()` 作为实验流程入口的职责。
- 已有代码母版的算法实现与调用关系（除明确的修复或扩展外）。
- `code_style.md` 中的强制代码风格。

`FIXED` **不是文件系统只读**，而是指：普通实验不允许为了方便而随意修改。如果用户明确要求修复框架、修复已确认的 Bug，或现有接口确实无法承担必要扩展，可以按第 10 节变更。

### 2.2 CONFIGURABLE：正常实验允许修改

典型配置如下；以实际母版中存在的配置项为准，不得凭空假定某个参数存在：

| 配置类别   | 代表参数                                                     |
| ---------- | ------------------------------------------------------------ |
| 输入与划分 | `CSV_PATH`、`CSV_TRAIN_PATH`、`CSV_TEST_PATH`、`DATA_SPLIT_MODE` |
| 划分与特征 | `TEST_SIZE`、`RANDOM_SEED_DATA`、`CATEGORICAL_COLUMNS`       |
| 模型       | `MODEL_PARAMS`、本次运行的模型清单                           |
| DL 训练    | `EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE`、`TEST_INTERVAL`、`RANDOM_SEED_MODEL` |
| 环境与保存 | `DEVICE`、`OUT_DIR`                                          |

具体名称和作用以模型母版为准。**正常实验应优先只修改顶部配置区域**，不得把数据路径、学习率、训练轮数等临时硬编码到函数内部。

### 2.3 EXTENSIBLE：确实需要时允许扩展

允许新增：新的模型、模型专属 Encoder、任务专属特征工程、特殊 Dataset、特殊训练方法、新的绘图类型、可复用的公共辅助函数。

扩展必须满足：

- 现有配置或函数不能完整实现任务。
- 新代码放入职责对应的目录。
- 旧接口能够保持时，不修改旧接口。
- 新增内容不能代替已经存在的规范模型文件。
- 不引入无必要的复杂架构。

## 3. Skill 与实验的目录关系

### 3.1 Skill 本体（固定）

```text
ml_dl_skill_v0.5.1/
├── SKILL.md
├── README.md
├── references/
│   ├── framework_contract.md
│   ├── code_style.md
│   ├── data_rules.md
│   ├── model_rules.md
│   └── output_rules.md
├── workflows/
│   ├── single_split_classification.md
│   └── kfold_classification.md
├── scripts/
│   ├── common/
│   │   └── cross_validation/
│   ├── ml_cls/
│   ├── dl_cls/
│   ├── ml_cls_fold/
│   ├── dl_cls_fold/
│   ├── checks/
│   ├── plotting/
│   └── run_all.sh
└── assets/
    ├── exp1_random_split/
    ├── exp2_fixed_split/
    └── exp3_5-fold/
```

- `scripts/`：**Canonical Implementation（规范代码母版）**。
- `assets/`：真实完成案例，用于参考结构与运行方式。
- `ml_cls/`、`dl_cls/`：单次划分模型母版；`ml_cls_fold/`、`dl_cls_fold/`：K-Fold 模型母版。
- `references/`：固定标准。
- `workflows/`：任务执行步骤。

不再增加 `static/`，不依赖 `manifest.yaml`；文件加载统一由 `SKILL.md` 指定。

### 3.2 生成的新实验（固定）

```text
experiment_name/
├── data/
├── code/
│   ├── common/
│   │   └── cross_validation/
│   ├── ml_cls/
│   ├── dl_cls/
│   ├── ml_cls_fold/
│   ├── dl_cls_fold/
│   ├── plotting/
│   └── run_all.sh
├── output/
├── task.md
└── README.md
```

要求：

- `data/` 保存数据或说明外部数据路径；大数据可不复制。
- `code/` 只放本任务需要的模型和公共依赖。
- `ml_cls/` 与 `dl_cls/` 分别存放单次划分模型专属实现。
- `ml_cls_fold/` 与 `dl_cls_fold/` 分别存放 K-Fold 模型专属实现；不得将单次划分和 Fold 训练入口混放。
- `plotting/` 只放独立绘图模块。
- `output/` 存放各模型结果及汇总结果。
- `task.md` 记录任务要求和实验配置；`README.md` 记录运行方法。
- 不需要的代码文件可以不复制；即使本次不使用某类模型，也不得将另一类模型错放入其目录。
- 按实际 Python 包导入方式保留所需 `__init__.py`。
- `run_all.sh` 仅在批量执行需要时复制或创建。

**禁止**为方便而把所有代码平铺在 `code/` 根目录，或自行发明与上述职责重复的目录。

## 4. 公共模块职责：`code/common/`

`common/` 是共享工具层，**不是多模型统一执行器**。

| 模块            | 允许承担的职责                                     | 不得承担的职责                                       |
| --------------- | -------------------------------------------------- | ---------------------------------------------------- |
| `data.py`       | CSV 读取、划分、特征列识别、通用编码与缩放辅助函数 | 决定并执行 CatBoost/XGBoost 等模型的完整专属处理流程 |
| `evaluation.py` | 统一指标、结果序列化、保存结果和预测               | 定义具体分类器、替模型决定训练策略                   |
| `dl_data.py`    | 通用 Dataset、DataLoader、标签和 Tensor 输入准备   | 某个模型的主体网络或整套实验流程                     |
| `dl_train.py`   | 通用 epoch 训练/测试循环及训练历史                 | 集中实现不同模型的专属结构与预处理                   |
| `dl_utils.py`   | 随机种子、设备选择等通用 DL 工具                   | 具体模型的主体逻辑                                   |

准则：

- 一个底层辅助函数能供多个模型复用，可以放入 `common/`。
- “如何组合这些辅助函数”属于模型专属处理时，组合逻辑必须留在对应模型 `.py` 的 `preprocess_data()` 中。
- 通用评价、保存和训练函数能复用时，不得重复实现。
- `common/` 不得通过模型名称选择并执行多个传统 ML 模型。


### 4.1 K-Fold 公共划分模块：`common/cross_validation/`

```text
common/cross_validation/
├── __init__.py
└── splitter.py
```

`splitter.py` 仅负责：

- `validate_split_config()`：检查折数、样本数、随机种子及分层条件。
- `get_kfold_splits()`：生成 KFold / StratifiedKFold 的训练和验证索引。
- `validate_splits()`：检查折数、索引越界、训练/验证交集、样本覆盖和验证索引唯一性。
- `get_split_summary()`：统计每折样本量和类别分布。

该模块不负责数据预处理、模型初始化、训练、评价、OOF 汇总或文件保存。ML 和 DL Fold 模型共用同一划分器。

## 5. 传统 ML 模型：One Model, One File

### 5.1 文件归属（强制）

```text
code/ml_cls/
├── dt.py
├── rf.py
├── gbdt.py
├── catboost.py
├── lightgbm.py
├── xgboost.py
├── knn.py
├── logistic_regression.py
└── svm.py
```

以上是**规范母版提供的可选模型文件**，新实验只需复制本次选择的模型。

每个实际使用的模型文件必须独立承担：

1. 模型专属配置。
2. `preprocess_data()`：确定本模型的处理方式，并调用必要的 `common.data` 基础工具。
3. `initialize_model()`：在本文件中调用对应模型类并建立模型。
4. `train_model()`：本模型的训练入口。
5. `test_model()`：本模型的预测入口。
6. `main()`：调用上述步骤，并复用统一评价与保存函数。

**模型可以调用通用函数，但不能丧失自身的完整实验入口。**

### 5.2 严格禁止的实现方式

禁止出现：

```text
ml_cls/run_catboost.py  → common/classical_runner.py
ml_cls/run_xgboost.py   → common/classical_runner.py
```

如果 `run_catboost.py` 仅修改模型名、随后转发到集中式 Runner，就不满足“一个模型一个文件”。

不得使用下列方式规避：

- `runpy.run_path()` 统一执行所有模型。
- `common/` 中的 `if model_name == ...` 或模型注册表，代替各模型专属初始化。
- 多个空壳模型入口，实际预处理、模型定义与训练全部在一个公共大文件中。
- 为了减少重复的少量普通代码而牺牲模型文件的独立可读性。

**允许的复用示例：**

`catboost.py` 自己实现 `preprocess_data()`，在内部调用 `normalize_categorical_columns()`；自己实现 `initialize_model()`，构造 `CatBoostClassifier`。

`xgboost.py` 自己实现 `preprocess_data()`，在内部调用 `transform_category_dtype()`；自己实现 `initialize_model()`，构造 `XGBClassifier`。

上述基础工具可共享，但不能把模型的具体处理决策迁入 `common/`。


### 5.3 K-Fold 传统 ML 模型：`ml_cls_fold/`

```text
code/ml_cls_fold/
└── catboost.py
```

每个 Fold ML 模型仍遵守 One Model, One File。模型文件独立组织 `preprocess_data()`、`initialize_model()`、`train_model()`、`test_model()` 和 `main()`；`main()` 调用公共 `splitter.py` 并组织各折训练、OOF 预测及结果保存。模型每折必须重新初始化，预处理器每折只在训练折 fit。

当前规范案例为 CatBoost，后续 RF、XGBoost 等模型沿此目录扩展。不得将 Fold 循环集中到一个按模型名称分发的公共 Runner。

## 6. DL 模型：专属网络与公共训练分开

```text
code/dl_cls/
├── mlp.py
├── tabpfn.py
└── eapcr/
    ├── eapcr.py
    └── eapcr_encoder.py
```

- `mlp.py`：MLP 网络结构、模型专属预处理、训练参数配置与完整入口。
- `tabpfn.py`：TabPFN 专属数据适配、初始化、预测与入口；不强行套用 epoch 训练。
- `eapcr/eapcr.py`：EAPCR 模型和实验入口。
- `eapcr/eapcr_encoder.py`：EAPCR 专属编码器。
- 普通 epoch 模型优先复用 `common/dl_train.py`。
- 特殊模型确实不能使用通用循环时，可在所属模型目录扩展专属训练实现；不得因此重写统一评价与保存协议。

一个复杂模型可以有自己的子目录；不应把所有 DL 模型合并成一个集中式 Runner。

## 7. 绘图与批量执行边界

### 7.1 `code/plotting/`

只负责：

- 从 `result.json`、`training_history.csv` 或汇总 CSV 读取结果。
- 生成混淆矩阵、折线图、柱状图等。
- 统一图形样式与导出。

**禁止**在模型训练文件中直接使用 Matplotlib 完成最终图形；绘图代码不得依赖模型对象、优化器或 DataLoader。

### 7.2 `code/run_all.sh`

只负责组织启动命令、选择执行模型和必要环境；不能在 Shell 中实现预处理、模型评价或绘图算法。

### 7.3 `output/`

各模型结果与多实验汇总按 `references/output_rules.md` 保存。不得因模型不同而自行更改统一结果协议。

## 8. 普通实验应该修改哪些地方

新任务先检查母版配置区。典型位置：

```python
# Editable experiment configuration

DATA_SPLIT_MODE = "random"
CSV_PATH = "..."
CSV_TRAIN_PATH = "..."
CSV_TEST_PATH = "..."
TEST_SIZE = 0.2
RANDOM_SEED_DATA = 42

MODEL_PARAMS = {
    ...
}

OUT_DIR = "..."
```

DL 文件还可能有 `BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`TEST_INTERVAL` 等。

要求：

- 普通实验的变化集中于模型文件配置区，或母版明确提供的配置入口。
- 不必为了配置变更修改 `common/`。
- 新实验应在实验副本中修改配置，不能直接覆盖 Skill 的 `scripts/` 母版。
- 不允许无理由引入新任务引擎、模型注册系统或新的 YAML/CLI 配置架构。
- 用户明确提出新的配置形式时，可以局部扩展，但仍要保持每个模型文件的职责。

## 9. 代码组织的强制要求

这部分只规定**结构边界**；具体排版由 `code_style.md` 决定。

- `main()` 应清楚展示：数据 → 预处理 → 初始化 → 训练 → 测试 → 评价 → 保存。
- `main()` 中各功能阶段应当清晰可辨。**新写或扩展的阶段尽量用空行分隔；原样复制母版时保留母版已有的空行布局，不强制重新排版。**
- 不得把完整的数据读取、模型定义、训练循环、评价和绘图实现都塞进 `main()`。
- 单个文件不允许同时充当模型、公共工具、绘图工具和任务总执行器。
- 一个模型文件可以包含它自己的预处理、模型初始化、训练与测试函数；**这种模型内聚不是违规的单体脚本**。
- 代码遵守 `references/code_style.md`，按**母版基线感知**原则检查新增代码；不因母版已有长行或连续空行强制修改原始 `scripts/`。

## 10. 修改母版及扩展权限

严格遵循：

```text
Level 1  修改实验配置
   ↓ 不足
Level 2  调用既有公共函数
   ↓ 不足
Level 3  在职责正确的模块中局部扩展
   ↓ 不足
Level 4  增加职责明确的专属模块
```

不得跳过前面的层级直接重新设计项目。

如果确实需要改变 FIXED 内容，例如修复公共函数 Bug 或增加必须的公共能力：

1. 明确需要修改的文件与原因。
2. 判断能否仅在实验副本中完成。
3. 优先保持既有接口兼容。
4. 说明对其他模型与案例的潜在影响。
5. 修改后检查相关模型的导入、训练、评价和输出。
6. **只有用户明确要求更新 Skill 母版时，才把改动写回 `scripts/`。**

不要因为生成代码“不够简短”，就合并模块或删除模型文件。

## 11. Golden Examples 使用边界

- `assets/exp1_random_split/`：参考 Random 单次划分、多模型分类、评价和绘图。
- `assets/exp2_fixed_split/`：参考固定测试集、不同训练样本规模和结果汇总。
- `assets/exp3_5-fold/`：参考 CatBoost / MLP 五折交叉验证、OOF 预测和折级结果汇总。

案例与 `scripts/` 的关系：

```text
scripts/         = 当前规范实现
assets/exp*/     = 已完成实验的历史实例
```

优先采用规范实现。案例用于参考如何配置与组合模块，不可替代当前母版，也不能直接复制案例结果。

一个模型在不同训练规模下需要多个可独立运行的文件时，可以按案例采用 `catboost_train_2000.py` 等**完整模型文件副本**，但不允许以只负责转发的空壳文件替代完整实现。

## 12. 创建实验前的文件级拆解

在动手生成代码前，必须完成如下映射：

| 用户需求                | 应使用的位置                             |
| ----------------------- | ---------------------------------------- |
| 数据输入与划分          | `common/data.py`，参数由对应模型入口传入 |
| CatBoost 单次划分处理和训练 | `ml_cls/catboost.py`                  |
| XGBoost 单次划分处理和训练 | `ml_cls/xgboost.py`                  |
| MLP 单次划分网络与入口 | `dl_cls/mlp.py`                         |
| K-Fold 划分 | `common/cross_validation/splitter.py` |
| CatBoost K-Fold | `ml_cls_fold/catboost.py` |
| MLP K-Fold | `dl_cls_fold/mlp.py` |
| 多模型公用指标与保存    | `common/evaluation.py`                   |
| 训练历史                | `common/dl_train.py` 等已有公共组件      |
| 混淆矩阵和模型对比图    | `plotting/`                              |
| 多模型启动              | `run_all.sh`（如有需要）                 |

实际操作时**只选用户要求的模型**。读取母版并核实接口后，决定复制文件、修改配置或进行必要扩展；不能根据上表虚构不存在的函数。

## 13. 交付前的结构验收

以下检查属于必须项：

| 检查内容    | 合格条件                                        |
| ----------- | ----------------------------------------------- |
| 项目目录    | 保持 `data/`、`code/`、`output/` 与规定模块层级 |
| ML 模型文件 | 每个模型有独立、实际可用的专属实现              |
| `common/`   | 只有共享工具，不含集中式多模型 Runner           |
| 模型入口    | 不是空壳转发文件，具有合理的 `main()` 组织      |
| DL 模型     | 专属网络在模型模块，公共训练合理复用            |
| 绘图        | 与训练分离，读取保存结果                        |
| 配置        | 普通任务只修改必要参数与任务专属代码            |
| 母版        | 没有无授权覆盖 Skill 的原始 `scripts/`          |
| 代码风格    | 符合 `references/code_style.md`                 |
| 执行与输出  | 满足 Workflow 和 `references/output_rules.md`   |

**不符合时必须修改到合格，不能只因为代码能够运行就宣称完成。**

静态脚本可以自动检查部分结构与语法条件；“模型逻辑是否真正归属于模型文件”等语义约束，还应结合实际代码审查。未执行的检查不能声称已经通过。

## 14. 最终约束

**普通实验 = 保持现有框架 + 复用规范母版 + 修改配置 + 必要的最小扩展。**

模型专属的内容归模型文件；多个模型共享的基础能力归 `common/`；绘图归 `plotting/`。

**禁止为了代码短或复用率高，牺牲“一个模型一个文件”、清晰的 `main()`、可读的模块边界。**

## 15. K-Fold DL 模型：`dl_cls_fold/`

```text
code/dl_cls_fold/
└── mlp.py
```

- 模型专属网络、预处理、数据加载配置和实验入口归对应 Fold 模型文件。
- 每折重新初始化网络、Criterion、Optimizer 和 Scheduler（若使用）。
- 每折独立创建训练/验证 Dataset 与 DataLoader。
- 普通 Epoch 训练优先复用 `common/dl_train.py`，例如 `run_epochs()`。
- 每折独立记录训练历史，合并后每行必须包含 `fold`。
- 每折模型权重按 `fold_i/` 保存；OOF 预测按原始样本索引还原。

当前规范案例为 MLP。TabPFN 等不使用普通 Epoch 循环的模型，必须按其实际拟合/推理接口实现，不得强行套用 `run_epochs()`。

## 16. K-Fold 与单次划分的模块边界

| 类型 | 单次划分 | K-Fold |
|---|---|---|
| 传统 ML | `ml_cls/` | `ml_cls_fold/` |
| 深度学习 | `dl_cls/` | `dl_cls_fold/` |
| 数据划分 | `common/data.py` 的实际接口 | `common/cross_validation/splitter.py` |
| DL Epoch 训练 | `common/dl_train.py` | 复用 `common/dl_train.py`，外层 Fold 由模型文件组织 |
| 结果评价 | `common/evaluation.py` | 同一评价函数；OOF 汇总由 Fold 模型组织 |

不要求创建统一 `ml_runner.py` 或 `dl_runner.py`。优先保持每个模型文件可独立阅读和执行。
