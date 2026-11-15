#!/bin/bash
set -e

# 加载 conda
source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate eapcr310

# 切换到当前脚本所在目录
cd "$(dirname "$0")"



# python -m ml_cls.catboost_train_2000
# python -m ml_cls.catboost_train_4000
# python -m ml_cls.catboost_train_6000

python -m dl_cls.mlp_train_2000
python -m dl_cls.mlp_train_4000
python -m dl_cls.mlp_train_6000

