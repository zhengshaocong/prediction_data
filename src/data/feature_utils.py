import pandas as pd
import sys
import os
# 引入 data/car/pre/clean_data.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/car/pre')))
from clean_data import default as clean_data

# 特征工程：添加新特征（只通过 feature_engineer 函数实现）
def add_features(df, train_df=None, feature_engineer=None):
    """
    通过 feature_engineer 函数进行特征工程
    """
    if feature_engineer is not None:
        df = feature_engineer(df, train_df=train_df)
    return df

# 特征工程：统一处理训练集和测试集特征
def preprocess_features(df, feature_cols):
    # 先将所有类别型特征转为字符串，避免fillna报错
    for col in feature_cols:
        if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)
    X = df[feature_cols].fillna('-1')  # 用字符串'-1'填充
    X = pd.get_dummies(X, dummy_na=True)
    # 检查重复列
    duplicated = X.columns[X.columns.duplicated()].unique()
    if len(duplicated) > 0:
        print("[错误] 检测到重复的特征列名：")
        for col in duplicated:
            idxs = [i for i, c in enumerate(X.columns) if c == col]
            print(f"列名: {col}，位置索引: {idxs}")
        raise ValueError("特征列名存在重复，请检查特征工程和特征处理流程！")
    return X

def preprocess_data(df, train_df=None, clean_data=None, feature_engineer=None):
    """
    数据预处理函数：包含数据清洗和特征工程。
    clean_data/feature_engineer均可选，未传则跳过。
    参数：
        df: 需要处理的数据（DataFrame）
        train_df: 训练集DataFrame（测试集特征工程时用，默认为None）
        clean_data: 数据清洗函数（可选）
        feature_engineer: 特征工程函数（可选）
    返回：
        处理后的DataFrame
    """
    if clean_data is not None:
        df = clean_data(df)
    if feature_engineer is not None:
        df = feature_engineer(df, train_df=train_df)
    return df 