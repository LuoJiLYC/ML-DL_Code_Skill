import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold



def validate_split_config(n_samples, y, n_splits, random_seed, stratified):
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits 必须是大于等于 2 的整数")
    if n_samples < n_splits:
        raise ValueError("样本数量不能小于折数")
    if not isinstance(random_seed, int):
        raise TypeError("random_seed 必须是整数")
    if not isinstance(stratified, bool):
        raise TypeError("stratified 必须是 bool")
    if stratified:
        if y is None:
            raise ValueError("StratifiedKFold 必须提供 y")
        _, counts = np.unique(np.asarray(y), return_counts=True)
        if counts.min() < n_splits:
            raise ValueError("每个类别的样本数都必须不少于 n_splits")



def get_kfold_splits(x, y, n_splits=5, random_seed=42, stratified=True):
    """生成 K 折索引；分类任务默认采用分层交叉验证。"""
    validate_split_config(len(x), y, n_splits, random_seed, stratified)

    if stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = list(splitter.split(x, y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = list(splitter.split(x))

    validate_splits(splits, len(x), n_splits)
    return splits



def validate_splits(splits, n_samples, n_splits):
    if len(splits) != n_splits:
        raise ValueError(f"划分折数错误：期望 {n_splits} 折，实际 {len(splits)} 折")

    val_indices = []
    expected_indices = np.arange(n_samples)

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)

        if len(np.unique(train_idx)) != len(train_idx):
            raise ValueError(f"Fold {fold}: 训练索引存在重复")
        if len(np.unique(val_idx)) != len(val_idx):
            raise ValueError(f"Fold {fold}: 验证索引存在重复")
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"Fold {fold}: 训练集与验证集存在交集")
        if np.any(train_idx < 0) or np.any(train_idx >= n_samples):
            raise ValueError(f"Fold {fold}: 训练索引越界")
        if np.any(val_idx < 0) or np.any(val_idx >= n_samples):
            raise ValueError(f"Fold {fold}: 验证索引越界")
        if not np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), expected_indices):
            raise ValueError(f"Fold {fold}: 训练集与验证集没有完整覆盖全部样本")

        val_indices.extend(val_idx.tolist())

    counts = np.bincount(val_indices, minlength=n_samples)
    if len(val_indices) != n_samples or not np.all(counts == 1):
        raise ValueError("每个样本必须且只能作为一次验证样本")



def get_split_summary(splits, y):
    """返回每折训练集和验证集的样本量、类别分布。"""
    y = np.asarray(y)
    summaries = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train_labels, train_counts = np.unique(y[train_idx], return_counts=True)
        val_labels, val_counts = np.unique(y[val_idx], return_counts=True)
        summaries.append({
            "fold": fold,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_class_counts": {str(k): int(v) for k, v in zip(train_labels, train_counts)},
            "val_class_counts": {str(k): int(v) for k, v in zip(val_labels, val_counts)},
        })

    return summaries
