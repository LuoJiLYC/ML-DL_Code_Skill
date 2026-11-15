import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SMALL_DATASET_CATEGORY_THRESHOLD = 100 # 样本数超过该阈值时，"取值很少的数值列"才被当作类别列。


CATEGORICAL_COLUMNS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "diag_1",
    "diag_2",
    "diag_3",
    "metformin",
    "repaglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
    "change",
    "diabetesMed",
]



def load_data_csv(csv_path):
    data = pd.read_csv(csv_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y



def split_dataset(x, y, test_size, random_seed, stratify=True):
    stratify_values = y if stratify else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_values,
    )

    return x_train, x_test, y_train, y_test



# def get_feature_columns(x):
    # """根据原始特征表，确定类别列和数值列。"""
    # categorical_columns = []
    # numerical_columns = []

    # for column in x.columns:
        # unique_count = x[column].nunique()

        # if not pd.api.types.is_numeric_dtype(x[column]):
            # categorical_columns.append(column)

        # elif (
            # len(x)
            # > SMALL_DATASET_CATEGORY_THRESHOLD
            # and unique_count < 4
        # ):
            # categorical_columns.append(column)

        # else:
            # numerical_columns.append(column)

    # return categorical_columns, numerical_columns


def get_feature_columns(x):
    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS
        if column in x.columns
    ]

    numerical_columns = [
        column for column in x.columns
        if column not in categorical_columns
    ]

    return categorical_columns, numerical_columns



def fit_one_hot_encoder(x_train, categorical_columns):
    """只用训练集拟合 One-Hot 编码器。"""
    if not categorical_columns:
        return None

    x_train = normalize_categorical_columns(x_train, categorical_columns)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(x_train[categorical_columns])

    return encoder



def transform_one_hot_encoder(x, encoder, categorical_columns):
    """使用训练集拟合的编码器转换数据；测试集中的未见类别会被忽略。"""
    if encoder is None or not categorical_columns:
        return x


    x = normalize_categorical_columns(x, categorical_columns)

    encoded = encoder.transform(x[categorical_columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=x.index,
    )
    
    x_processed = pd.concat(
        [x.drop(columns=categorical_columns), encoded_df],
        axis=1,
    )

    return x_processed



def normalize_categorical_columns(x, categorical_columns):
    """
    将类别特征统一转换为字符串类型。

    这样可以避免固定训练集/测试集分别读取 CSV 时，
    同一类别列被 pandas 推断成不同 dtype。
    """
    x = x.copy()

    for column in categorical_columns:
        x[column] = (
            x[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return x



def prepare_data(split_mode, csv_path=None, csv_train_path=None, csv_test_path=None, test_size=0.2, random_seed=42):
    """根据数据划分模式准备训练集和测试集。"""

    if split_mode == "random":

        if csv_path is None:
            raise ValueError("random 模式下必须提供 csv_path")

        x, y = load_data_csv(csv_path=csv_path)

        x_train, x_test, y_train, y_test = split_dataset(
            x,
            y,
            test_size=test_size,
            random_seed=random_seed,
        )

        data_info = {
            "split_method": "random",
            "csv_path": csv_path,
            "test_size": test_size,
            "random_seed": random_seed,
        }

    elif split_mode == "fixed":

        if csv_train_path is None or csv_test_path is None:
            raise ValueError("fixed 模式下必须提供 csv_train_path 和 csv_test_path")

        x_train, y_train = load_data_csv(csv_path=csv_train_path)
        x_test, y_test = load_data_csv(csv_path=csv_test_path)

        # 检查训练集和测试集特征列
        if list(x_train.columns) != list(x_test.columns):
            raise ValueError("训练集和测试集的特征列不一致，请检查列名和列顺序。")

        # 检查测试集标签
        check_train_test_labels(y_train, y_test)

        data_info = {
            "split_method": "fixed",
            "train_path": csv_train_path,
            "test_path": csv_test_path,
        }

    else:
        raise ValueError(
            f"不支持的数据划分模式：{split_mode}"
        )

    return x_train, x_test, y_train, y_test, data_info





def fit_standard_scaler(x_train, numerical_columns):
    if not numerical_columns:
        return None

    scaler = StandardScaler()
    scaler.fit(x_train[numerical_columns])

    return scaler



def transform_standard_scaler( x, scaler, numerical_columns):
    if scaler is None or not numerical_columns:
        return x

    x = x.copy()
    x[numerical_columns] = scaler.transform(x[numerical_columns])

    return x




def check_train_test_labels(y_train, y_test):
    """检查测试集是否包含训练集中未出现的标签。"""

    train_labels = set(y_train.unique())
    test_labels = set(y_test.unique())

    unknown_labels = test_labels - train_labels

    if unknown_labels:
        raise ValueError(f"测试集中存在训练集未出现的标签：{sorted(unknown_labels)}")



def transform_category_dtype(x_train, x_test, categorical_columns):
    """将类别特征转换为 pandas category，并保持 train/test 类别定义一致。"""

    x_train = x_train.copy()
    x_test = x_test.copy()

    for column in categorical_columns:
        train_values = x_train[column].astype("string").fillna("__MISSING__")
        test_values = x_test[column].astype("string").fillna("__MISSING__")

        categories = train_values.unique()

        x_train[column] = pd.Categorical(train_values, categories=categories)
        x_test[column] = pd.Categorical(test_values, categories=categories)

    return x_train, x_test