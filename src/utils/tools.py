import numpy as np
import pandas as pd

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