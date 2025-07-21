import numpy as np
import pandas as pd
import os
import json
import hashlib
import time
from typing import TypedDict, List, Optional

def auto_bins(series, n_bins=5, method='quantile'):
    """
    自动分箱函数。
    参数：
        series: pd.Series，待分箱数据
        n_bins: 分箱数量
        method: 分箱方法，支持 'quantile'（分位数）和 'uniform'（等宽）
    返回：labels, bins
    """
    if method == 'quantile':
        bins = pd.qcut(series, q=n_bins, retbins=True, duplicates='drop')[1]
    elif method == 'uniform':
        bins = np.linspace(series.min(), series.max(), n_bins + 1)
    else:
        raise ValueError('method 仅支持 quantile 或 uniform')
    labels = pd.cut(series, bins=bins, labels=False, include_lowest=True)
    return labels, bins

def check_and_get_feature_blocks(model_type, FEATURE_BLOCKS):
    """
    检查 cache/best_feature_selection.json 是否存在最优特征组合。
    若有则返回 best_features，否则询问用户是否继续，继续则返回 FEATURE_BLOCKS，不继续则终止程序。
    """
    cache_path = 'cache/best_feature_selection.json'
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        # 可扩展：如需区分模型类型，可在json中加model_type字段
        best_features = cache.get('best_features', None)
        if best_features is not None and len(best_features) > 0:
            print(f"检测到最优特征组合，将仅使用：{best_features}")
            return best_features
    # 没有最优特征组合，询问用户
    ans = input("未找到最优特征组合，是否继续使用全部特征工程？（y/n）: ").strip().lower()
    if ans != 'y':
        print("已终止程序。若想生成最优特征工程组合请到config.yaml修改feature_selection_mode为true生成最优组合！")
        exit(0)
    return FEATURE_BLOCKS 

def make_cache_key(model_type, train_path, test_path):
    """
    生成特征工程缓存的唯一短key（md5哈希），用于高效检索。
    参数：
        model_type: 模型类型（如xgboost、catboost等）
        train_path: 训练集路径
        test_path: 测试集路径
    返回：
        唯一短key字符串
    """
    raw = f"{model_type}|{train_path}|{test_path}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

class FeatureSelectionCache(TypedDict):
    """
    特征工程组合缓存结构体：
    - model_type: str，模型类型（如 'xgboost', 'catboost' 等）
    - train_path: str，训练集数据文件路径
    - test_path: str，测试集数据文件路径
    - timestamp: str，生成时间（格式 'YYYY-MM-DD HH:MM:SS'）
    - best_features: List[str]，最优特征工程组合（特征工程名称列表）
    - best_score: float，最优组合对应的评估分数（如MAE）
    - generated_columns: List[str]，最终生成的新特征列名
    - last_used_time: str，最近一次被主流程使用的时间（自动维护）
    """
    model_type: str
    train_path: str
    test_path: str
    timestamp: str
    best_features: List[str]
    best_score: float
    generated_columns: List[str]
    last_used_time: str

def save_feature_selection_cache(new_result: FeatureSelectionCache, cache_path: str = 'cache/best_feature_selection.json') -> None:
    """
    保存一份特征工程组合缓存，支持多份缓存共存，key为哈希值。
    若已存在相同模型+数据的缓存则覆盖，否则新增。
    参数：
        new_result: FeatureSelectionCache，特征工程组合缓存结构体
        cache_path: 缓存文件路径
    """
    key = make_cache_key(new_result['model_type'], new_result['train_path'], new_result['test_path'])
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_dict = json.load(f)
    else:
        cache_dict = {}
    new_result['last_used_time'] = new_result['timestamp']
    cache_dict[key] = new_result
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_dict, f, ensure_ascii=False, indent=2)

def load_feature_selection_cache(model_type: str, train_path: str, test_path: str, cache_path: str = 'cache/best_feature_selection.json') -> Optional[FeatureSelectionCache]:
    """
    加载指定模型+数据的特征工程组合缓存，并自动更新最近使用时间。
    参数：
        model_type: 模型类型
        train_path: 训练集路径
        test_path: 测试集路径
        cache_path: 缓存文件路径
    返回：
        匹配的缓存FeatureSelectionCache，若无则返回None
    """
    key = make_cache_key(model_type, train_path, test_path)
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_dict = json.load(f)
    if key in cache_dict:
        cache_dict[key]['last_used_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(cache_path, 'w', encoding='utf-8') as f2:
            json.dump(cache_dict, f2, ensure_ascii=False, indent=2)
        return cache_dict[key]
    return None

def save_feature_selection_cache_list(new_result: FeatureSelectionCache, cache_path='cache/best_feature_selection.json') -> None:
    """
    保存一份特征工程组合缓存到有序列表，若已存在则覆盖并置顶。
    参数：
        new_result: FeatureSelectionCache，特征工程组合缓存结构体
        cache_path: 缓存文件路径
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_list = json.load(f)
    else:
        cache_list = []
    key = (new_result['model_type'], new_result['train_path'], new_result['test_path'])
    new_result['last_used_time'] = new_result['timestamp']
    # 去重并置顶
    cache_list = [c for c in cache_list if (c['model_type'], c['train_path'], c['test_path']) != key]
    cache_list.insert(0, new_result)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_list, f, ensure_ascii=False, indent=2)

def load_feature_selection_cache_list(model_type: str, train_path: str, test_path: str, cache_path: str = 'cache/best_feature_selection.json') -> Optional[FeatureSelectionCache]:
    """
    加载指定模型+数据的特征工程组合缓存（有序列表），并自动更新最近使用时间和置顶。
    参数：
        model_type: 模型类型
        train_path: 训练集路径
        test_path: 测试集路径
        cache_path: 缓存文件路径
    返回：
        匹配的缓存FeatureSelectionCache，若无则返回None
    """
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_list = json.load(f)
    key = (model_type, train_path, test_path)
    for i, c in enumerate(cache_list):
        if (c['model_type'], c['train_path'], c['test_path']) == key:
            c['last_used_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            cache_list.insert(0, cache_list.pop(i))
            with open(cache_path, 'w', encoding='utf-8') as f2:
                json.dump(cache_list, f2, ensure_ascii=False, indent=2)
            return c
    return None

class ProgressPrinter:
    def __init__(self, total, prefix="进度"):
        self.total = total
        self.current = 0
        self.prefix = prefix
    def add(self, msg=None):
        self.current += 1
        if msg:
            print(f"{msg}，{self.prefix}: {self.current}/{self.total}")
        else:
            print(f"{self.prefix}: {self.current}/{self.total}") 