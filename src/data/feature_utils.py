import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/pre')))
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
    X = df[feature_cols].fillna(-1)
    X = pd.get_dummies(X, dummy_na=True)
    return X

def preprocess_data(df, train_df=None, feature_engineer=None):
    """
    数据预处理函数：包含数据清洗和特征工程。
    参数：
        df: 需要处理的数据（DataFrame）
        train_df: 训练集DataFrame（测试集特征工程时用，默认为None）
        feature_engineer: 特征工程函数
    返回：
        处理后的DataFrame
    """
    df = clean_data(df)
    df = add_features(df, train_df=train_df, feature_engineer=feature_engineer)
    return df 