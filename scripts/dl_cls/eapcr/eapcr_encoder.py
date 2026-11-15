import numpy as np
import pandas as pd
import pickle


class EAPCREncoder:
    """EAPCR 特征编码器：仅在训练集上拟合分箱规则和全局 ID 映射。"""

    def __init__(self, n_bins=5, custom_bins=None):
        self.n_bins = n_bins
        self.custom_bins = custom_bins or {}
        self.feature_names = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.feature_info = {}
        self.num_embeddings = 0
        self.is_fitted = False

    def fit(self, x_train, categorical_columns=None, numerical_columns=None):
        x_train = x_train.copy()
        self.feature_names = list(x_train.columns)

        if categorical_columns is None:
            categorical_columns = [c for c in x_train.columns if not pd.api.types.is_numeric_dtype(x_train[c])]
        if numerical_columns is None:
            numerical_columns = [c for c in x_train.columns if c not in categorical_columns]

        if set(categorical_columns) & set(numerical_columns):
            raise ValueError("categorical_columns 和 numerical_columns 存在重复列")
        if set(categorical_columns) | set(numerical_columns) != set(self.feature_names):
            raise ValueError("类别列和数值列必须完整覆盖所有特征列")

        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = list(numerical_columns)
        self.feature_info = {}

        current_id = 0

        for column in self.feature_names:
            series = x_train[column]
            info = {"start_id": current_id}

            if column in self.categorical_columns:
                info["type"] = "categorical"
                values = series.astype("string").dropna().unique().tolist()
                values = sorted(str(value) for value in values)
                info["mapping"] = {value: current_id + i for i, value in enumerate(values)}
                current_id += len(values)

            else:
                numeric_series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
                bins_num = int(self.custom_bins.get(column, self.n_bins))
                if bins_num < 1:
                    raise ValueError(f"{column} 的分箱数必须大于 0")

                unique_count = numeric_series.nunique(dropna=True)

                if unique_count > bins_num:
                    info["type"] = "binned_numeric"
                    clean_values = numeric_series.dropna()
                    _, bin_edges = pd.qcut(clean_values, q=bins_num, retbins=True, duplicates="drop")
                    bin_edges = np.asarray(bin_edges, dtype=float)
                    bin_edges[0] = -np.inf
                    bin_edges[-1] = np.inf
                    info["bin_edges"] = bin_edges

                    num_actual_bins = len(bin_edges) - 1
                    info["mapping"] = {i: current_id + i for i in range(num_actual_bins)}
                    current_id += num_actual_bins

                else:
                    info["type"] = "discrete_numeric"
                    values = sorted(numeric_series.dropna().unique().tolist())
                    info["mapping"] = {value: current_id + i for i, value in enumerate(values)}
                    current_id += len(values)

            info["unknown_id"] = current_id
            current_id += 1
            info["missing_id"] = current_id
            current_id += 1
            info["end_id"] = current_id - 1
            self.feature_info[column] = info

        self.num_embeddings = current_id
        self.is_fitted = True
        return self

    def transform(self, x):
        if not self.is_fitted:
            raise RuntimeError("EAPCREncoder 尚未 fit")
        if list(x.columns) != self.feature_names:
            raise ValueError("输入特征列与训练时不一致，请检查列名和列顺序")

        encoded = pd.DataFrame(index=x.index)

        for column in self.feature_names:
            series = x[column]
            info = self.feature_info[column]
            mapping = info["mapping"]

            if info["type"] == "categorical":
                values = []
                for value in series:
                    if pd.isna(value):
                        values.append(info["missing_id"])
                    else:
                        values.append(mapping.get(str(value), info["unknown_id"]))

            elif info["type"] == "discrete_numeric":
                numeric_series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
                values = []
                for value in numeric_series:
                    if pd.isna(value):
                        values.append(info["missing_id"])
                    else:
                        values.append(mapping.get(value, info["unknown_id"]))

            elif info["type"] == "binned_numeric":
                numeric_series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
                bin_indices = pd.cut(numeric_series, bins=info["bin_edges"], labels=False, include_lowest=True)
                values = []
                for original_value, bin_index in zip(numeric_series, bin_indices):
                    if pd.isna(original_value):
                        values.append(info["missing_id"])
                    elif pd.isna(bin_index):
                        values.append(info["unknown_id"])
                    else:
                        values.append(mapping.get(int(bin_index), info["unknown_id"]))

            else:
                raise ValueError(f"未知特征类型：{info['type']}")

            encoded[column] = values

        return encoded.astype(np.int64)

    def fit_transform(self, x_train, categorical_columns=None, numerical_columns=None):
        self.fit(x_train, categorical_columns=categorical_columns, numerical_columns=numerical_columns)
        return self.transform(x_train)

    def get_summary(self):
        if not self.is_fitted:
            raise RuntimeError("EAPCREncoder 尚未 fit")

        return {
            "n_bins": self.n_bins,
            "custom_bins": self.custom_bins,
            "num_features": len(self.feature_names),
            "num_embeddings": self.num_embeddings,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "feature_types": {column: info["type"] for column, info in self.feature_info.items()},
        }




    def save(self, file_path):
        """保存已经 fit 的 EAPCREncoder。"""
        if not self.is_fitted:
            raise RuntimeError("EAPCREncoder 尚未 fit，不能保存")

        with open(file_path, "wb") as f:
            pickle.dump(self, f)

        print(f"EAPCR Encoder 已保存：{file_path}")


    @staticmethod
    def load(file_path):
        """加载已经保存的 EAPCREncoder。"""
        with open(file_path, "rb") as f:
            encoder = pickle.load(f)

        if not isinstance(encoder, EAPCREncoder):
            raise TypeError("文件中保存的不是 EAPCREncoder")

        return encoder