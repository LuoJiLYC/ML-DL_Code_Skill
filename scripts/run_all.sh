#!/bin/bash
set -e

# 加载 conda
source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate eapcr310

# 切换到当前脚本所在目录
cd "$(dirname "$0")"


python -m ml_cls.dt
python -m ml_cls.rf
python -m ml_cls.gbdt
python -m ml_cls.catboost
python -m ml_cls.lightgbm
python -m ml_cls.xgboost
python -m ml_cls.knn
python -m ml_cls.logistic_regression
python -m ml_cls.svm

python -m dl_cls.mlp
python -m dl_cls.eapcr.eapcr

conda activate tabpfn310

python -m dl_cls.tabpfn

