import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import importlib.util
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import json
import time

from src.config.config_utils import load_config
from src.data.feature_utils import preprocess_features
from src.models.utils.model_utils_tools import select_model
from src.utils.tools import ProgressPrinter, FeatureSelectionCache, save_feature_selection_cache, load_feature_selection_cache

def get_full_block_set(block, selected, dependencies):
    # 递归补全依赖
    blocks = set([block])
    for dep in dependencies.get(block, []):
        if dep not in selected:
            blocks |= get_full_block_set(dep, selected, dependencies)
    return blocks

def load_clean_and_feature_modules(config):
    # 动态加载清洗和特征工程模块
    clean_module_path = config['data']['clean_module']
    preprocess_module_path = config['data']['preprocess_module']
    # 加载clean_data
    spec_clean = importlib.util.spec_from_file_location('clean_data', clean_module_path)
    clean_data_mod = importlib.util.module_from_spec(spec_clean)
    spec_clean.loader.exec_module(clean_data_mod)
    clean_data = clean_data_mod.default
    # 加载feature_blocks
    spec_feat = importlib.util.spec_from_file_location('feature_blocks', preprocess_module_path)
    feature_blocks_mod = importlib.util.module_from_spec(spec_feat)
    spec_feat.loader.exec_module(feature_blocks_mod)
    feature_blocks_default = feature_blocks_mod.default
    FEATURE_BLOCKS = feature_blocks_mod.FEATURE_BLOCKS
    feature_dependencies = getattr(feature_blocks_mod, 'feature_dependencies', {})
    return clean_data, feature_blocks_default, FEATURE_BLOCKS, feature_dependencies

def load_data(config):
    df = pd.read_csv(config['data']['train_path'], sep=' ')
    return df

def evaluate_feature_blocks(df, clean_data, feature_blocks_default, feature_blocks, model_type, target_col, id_col, all_generated_cols=None):
    # 数据清洗
    df_clean = clean_data(df.copy())
    # 特征工程
    df_feat = feature_blocks_default(df_clean, feature_blocks=feature_blocks)
    # 新增列
    new_cols = set(df_feat.columns) - set(df_clean.columns)
    if all_generated_cols is not None:
        all_generated_cols.update(new_cols)
    # 构造特征和标签
    feature_cols = [col for col in df_feat.columns if col not in [id_col, target_col]]
    # 在独热编码前，去除重复列和已生成列
    X = df_feat[feature_cols]
    if all_generated_cols is not None:
        X = X.loc[:, ~X.columns.isin(list(all_generated_cols - set(feature_cols)))]
    X = X.loc[:, ~X.columns.duplicated()]
    # 独热编码
    X = pd.get_dummies(X, dummy_na=True)
    # 再次去除重复列
    X = X.loc[:, ~X.columns.duplicated()]
    y = df_feat[target_col]
    # 模型
    model = select_model(model_type)
    # 交叉验证
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    return np.mean(scores)


# 2. 修改greedy_feature_selection，自动补全依赖组

def greedy_feature_selection(df, clean_data, feature_blocks_default, all_blocks, model_type, target_col, id_col, feature_dependencies):
    selected = []
    best_score = -np.inf
    remaining = list(all_blocks)
    all_generated_cols = set()
    progress = ProgressPrinter(total=len(all_blocks), prefix="特征组合进度")
    while remaining:
        scores = []
        for block in remaining:
            # 自动补全依赖组
            full_blocks = list(set(selected) | get_full_block_set(block, selected, feature_dependencies))
            score = evaluate_feature_blocks(df, clean_data, feature_blocks_default, full_blocks, model_type, target_col, id_col, all_generated_cols)
            scores.append((score, block, full_blocks))
            print(f"尝试添加: {block}（依赖组: {set(full_blocks) - set(selected)}），MAE: {-score:.4f}")
        scores.sort(reverse=True)
        if scores[0][0] > best_score:
            best_score, best_block, best_full_blocks = scores[0]
            # 只添加本轮新加入的块（包括依赖）
            new_blocks = set(best_full_blocks) - set(selected)
            selected.extend(list(new_blocks))
            for b in new_blocks:
                if b in remaining:
                    remaining.remove(b)
            progress.add(f"添加特征块组: {new_blocks}, 当前最优MAE: {-best_score:.4f}，已选特征数: {len(selected)}/{len(all_blocks)}")
        else:
            break
    return selected, best_score, list(all_generated_cols)


def main():
    config = load_config('config.yaml')
    clean_data, feature_blocks_default, FEATURE_BLOCKS, feature_dependencies = load_clean_and_feature_modules(config)
    df = load_data(config)
    model_type = config['model']['type']
    target_col = config.get('prediction_column', 'price')
    id_col = config.get('id_column', 'SaleID')

    # 先判断是否已有缓存（字典结构）
    cache = load_feature_selection_cache(model_type, config['data']['train_path'], config['data']['test_path'])
    if cache is not None:
        print(f"已存在该模型+数据的特征工程组合缓存，最近一次调用时间：{cache.get('last_used_time', cache.get('timestamp', '未知'))}")
        ans = input("是否覆盖并重新生成？（y/n）：").strip().lower()
        if ans != 'y':
            print("流程已终止，未覆盖原有缓存。")
            return

    # 先评测无特征工程的分数
    print("\n[基线] 仅用原始字段（无特征工程）分数：")
    baseline_score = evaluate_feature_blocks(
        df, clean_data, feature_blocks_default, [], model_type, target_col, id_col
    )
    print(f"无特征工程 MAE: {-baseline_score:.4f}")

    # 再进行特征工程组合搜索
    selected_blocks, best_score, all_generated_cols = greedy_feature_selection(
        df, clean_data, feature_blocks_default, FEATURE_BLOCKS, model_type, target_col, id_col, feature_dependencies
    )
    print("最优特征工程组合：", selected_blocks)
    print("最优得分：", -best_score)
    os.makedirs('cache', exist_ok=True)
    result: FeatureSelectionCache = {
        "model_type": model_type,
        "train_path": config['data']['train_path'],
        "test_path": config['data']['test_path'],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_features": selected_blocks,
        "best_score": best_score,
        "generated_columns": all_generated_cols,
        "last_used_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_feature_selection_cache(result)
    print("最优特征组合及分数已保存到 cache/best_feature_selection.json")

if __name__ == "__main__":
    main()