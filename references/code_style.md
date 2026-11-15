# Code Style Contract

适用 Skill 版本：v0.5.1（以 `scripts/` 现有母版为准）

## 1. 目标与适用范围

本文件规定**新实验生成、修改部分**的代码风格。`scripts/` 中现有代码是 Canonical Implementation（规范母版），不得因为通用格式规范而批量修改母版。

**优先级：现有模型母版的组织方式及实际接口 > 本文一般性排版建议。** 本文不能成为重新格式化母版、重写模型或改变其输出的理由。普通实验只修改配置；新增实现应当自然融入相应模型文件。

## 2. 母版优先与修改边界

- **原样复制**所选模型、通用模块和绘图模块，再在实验副本中修改任务需要的部分。
- 保留母版已经存在的 `import` 组织、配置区分隔、函数排列、注释和空行节奏。母版中的多行连续空行**不构成要求修复的错误**。
- 母版既有的超长行**不构成要求修复的错误**；无需为了达到固定字符数而整理未修改的行。
- 不得借代码风格调整变更模型算法、数据处理、评价指标、文件命名、接口或输出格式。
- 新写的功能应符合下文规范。已有母版代码仅在用户明确要求重构、存在实际 Bug 或为完成任务必须改动时进行最小范围修正。

## 3. 新写代码的可读性要求

### 3.1 一行一个主要动作

禁止新写分号拼接多个独立语句：

```python
# 禁止
model.train(); optimizer.zero_grad(); loss.backward()

# 推荐
model.train()
optimizer.zero_grad()
loss.backward()
```

禁止新写单行函数：

```python
# 禁止
def test_model(model, x): return model.predict(x)

# 推荐
def test_model(model, x):
    return model.predict(x)
```

已有母版中存在的单行 `if` 或其他短写法，不应被当成修改无关代码的理由；新代码在涉及多个动作、分支或副作用时应拆为清楚的多行。

### 3.2 长行与换行

- 对**新增或实际修改的行**，以约 120 字符为可读性参考界限；过长且难以理解时使用括号、参数分行或拆分多个独立操作。
- 不对未修改的母版行强制 120 字符上限，也不因一行略超界限而重写现有实现。
- 短表达式和参数较少的调用保留一行；长调用按参数拆行，避免压缩成密集的一行。
- 长字典、长列表按语义换行，保留母版已有的格式风格。

### 3.3 空行与 `main()` 结构

- **不设全局最大连续空行数**。母版已有的大段空行（尤其是配置区与函数区之间）允许原样保留。
- 新增顶层函数时参考所在母版的分隔风格，不做整文件的空行统一清理。
- `main()` 应清楚体现数据读取、预处理、初始化、训练、测试、评价和保存等逻辑阶段；**新插入的不同阶段之间建议保留一个空行**。
- 如果母版将训练与测试、评价与保存等相邻步骤连续书写，复制母版时保持原样，**不得以空行规则要求重排**。
- 同一功能阶段内部可以连续书写；不得为每个简单赋值机械地插入空行。

下面是**新写 `main()` 时的推荐示意**（实际函数参数以母版为准）：

```python
def main():
    x_train, x_test, y_train, y_test, info = prepare_data(...)

    x_train, x_test, preprocessor = preprocess_data(x_train, x_test)

    model = initialize_model(MODEL_PARAMS, preprocessor)

    model, train_time = train_model(model, x_train, y_train)
    y_pred, y_probs, test_time = test_model(model, x_test)

    metrics = metrics_classification(y_test, y_pred, y_probs, class_labels=model.classes_)

    save_result_json(...)
    save_predictions_csv(...)
```

此示意表示**逻辑分块**，不是要求所有历史母版 `main()` 重排成同样空行。

### 3.4 命名、注释、复用

- 尽量延续母版的 `x_train`、`y_pred`、`MODEL_PARAMS`、`preprocess_data()` 等命名。
- 新增变量使用能表达含义的名称；避免为了省字使用连续的无意义单字母变量。
- 注释解释非显然的约束和原因，不重复每行代码本身。
- 模型专属预处理、模型定义及训练入口必须归相应 `ml_cls/*.py` 或 `dl_cls/`；不能把它们挪到集中式 `common/` Runner。
- `main()` 用于编排流程；绘图与训练必须分离。

## 4. 代码检查策略（母版基线感知）

运行 `scripts/checks/validate_code.py` 时：

1. 对所有新实验 `.py` 执行 Python 语法解析。
2. 检查 ML 模型文件是否具备母版规定的独立函数与入口。
3. 对于可对应的母版源文件，比较**新引入或修改的行**，重点阻止新增分号串联语句、单行函数、明显过长的代码行。
4. 对于原样复制的母版行，即使原本存在长行或多行空行，也**不得因此判定新实验失败**。
5. `main()` 的阶段分隔以及模型职责等语义问题需要结合实际代码复核，静态检查未覆盖的项目不得宣称已自动证明。

**不得运行自动格式化工具对整个母版或整个实验副本一键重新排版**，除非用户明确要求。

## 5. 交付标准

代码可读、能运行且与规范母版保持结构和接口一致。若为了遵守所谓“格式”而破坏母版风格、修改未涉及的代码或改变训练行为，视为不符合本 Skill 的目标。


## 6. K-Fold 模型代码目录

- 单次划分模型放在 `ml_cls/`、`dl_cls/`。
- K-Fold 模型放在 `ml_cls_fold/`、`dl_cls_fold/`。
- 公共划分器放在 `common/cross_validation/splitter.py`。
- `splitter.py` 只实现索引生成、合法性检查和划分统计；不得把模型训练逻辑放入其中。
- ML Fold 模型沿用传统 ML 模型函数组织；DL Fold 模型保留网络、DataLoader、优化器和 Epoch 训练的职责边界。
- 检查 Fold 模型风格时，优先与 `scripts/ml_cls_fold/`、`scripts/dl_cls_fold/` 对应母版比较；若母版尚未建立，再参考 `assets/exp3_5-fold/` 的案例。
