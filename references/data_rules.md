# Data Rules

适用 Skill 版本：v0.5.1

## 1. 输入格式

当前 Skill 使用标准 CSV。

要求：

```text
第 0 行 = 表头
前 n 列 = 特征
最后一列 = 标签
每一行 = 一个样本
```

示例：

```text
age,gender,bmi,smoking,label
65,male,23.5,yes,1
52,female,21.8,no,0
```

---

## 2. 特征类型

特征分为：

```text
numerical feature
categorical feature
```

模型训练前必须能够确定每列属于哪种类型。

如果自动判断存在歧义，例如整数编码的类别特征，应优先使用用户明确指定的类别列。

---

## 3. 缺失值

当前分类 Workflow 不负责完整的原始数据清洗。

进入训练阶段前：

### 数值特征

当前公共 `scripts/common/data.py` 未实现通用均值填补。进入训练前，按模型要求检查数值特征是否允许缺失。需要在实验中新增均值填补时，只从训练集或当前训练折计算均值，并用同一组均值处理对应的测试集或验证折。

禁止：

```text
使用 test mean
使用 train + test 共同计算 mean
```

### 类别特征

缺失值统一表示为：

```text
__MISSING__
```

### 标签

标签列不得包含缺失值。

---

## 4. random split

输入：

```text
一个完整 CSV
```

使用：

```python
train_test_split()
```

划分训练集和测试集。

至少记录：

```text
test_size
random_seed
```

分类任务默认使用 stratified split，除非用户明确要求其他方式。

---

## 5. fixed split

输入：

```text
train.csv
test.csv
```

必须检查：

- 特征列数量一致
- 特征列名称一致
- 特征列顺序一致
- 标签列定义一致
- test 中不存在 train 未见标签类别

---

## 6. 数据泄漏

所有会从数据中学习参数的操作必须只在训练集上 fit。

正确：

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

该规则适用于：

```text
OneHotEncoder
StandardScaler
LabelEncoder
数值填补统计量
类别定义
EAPCR bin_edges
EAPCR ID mapping
```

---

## 7. 未见类别

测试集出现训练集未见的特征类别时，必须使用模型对应的安全处理方式。

例如：

```text
OneHotEncoder
→ handle_unknown="ignore"

EAPCR
→ unknown_id
```

不得重新在测试集上建立编码器。

---

## 8. 标签规则

标签编码必须由训练集建立。

测试集标签只执行 transform。

如果测试集存在训练集未出现的标签：

```text
立即报错
```

因为模型没有学习过该类别。

---

## 9. 数据完成条件

数据可以进入模型训练阶段的条件：

- CSV 可读取
- 至少存在一个特征
- 标签列存在
- 标签无缺失
- train/test 结构合法
- 基础缺失值已经处理
- 不存在明显的数据泄漏


## 10. K-Fold 交叉验证

### 10.1 默认划分

分类任务默认使用：

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

普通 `KFold` 仅在用户明确要求或任务不适合分层时使用。`n_splits`、随机种子和是否分层必须记录到实验配置及 `result.json`。

### 10.2 分层样本量要求

使用 `StratifiedKFold` 时，每个类别样本数必须不少于 `n_splits`。否则应终止并说明原因，不得静默切换为普通 KFold。

### 10.3 每折独立预处理

对每一折分别执行：

```text
fit(train_fold)
transform(train_fold)
transform(validation_fold)
```

不得在完整数据上提前 fit StandardScaler、OneHotEncoder、缺失值填补器、类别映射、EAPCR 分箱边界或其他学习型预处理器。

SMOTE 等重采样仅能作用于当前训练折，不能作用于验证折；验证折保持原始分布。

### 10.4 每折独立训练

- 每折必须重新初始化模型。
- DL 每折必须重新初始化网络和优化器，使用学习率调度器时也要每折重建。
- 每折训练/验证数据必须由原始数据和当前折索引提取，不能从上一折处理后的数据继续切分。
- 验证折不参与梯度更新、模型拟合或预处理器拟合。

### 10.5 OOF 预测约束

- 每个样本恰好作为一次外层验证样本。
- OOF 结果必须保留原始 `sample_index` 和对应 `fold`。
- 多分类概率必须按全局类别顺序对齐。
- OOF 样本数必须等于原始样本数；不允许遗漏或重复。

### 10.6 分组与时间相关数据

普通 KFold / StratifiedKFold 假定样本可独立随机分配。若同一患者、设备、用户、站点或其他主体存在多条相关记录，应评估 GroupKFold / StratifiedGroupKFold；若任务要求预测未来，应采用符合时间顺序的验证方式。不得把相关主体或未来信息随机分散到训练折和验证折后，仍宣称验证结果具有独立泛化意义。
