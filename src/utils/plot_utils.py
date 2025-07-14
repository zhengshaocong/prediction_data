import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_price_dist(price_series, img_dir, font, title='价格分布', filename='price_dist.png'):
    """
    绘制价格分布直方图并保存。
    参数：
        price_series: pd.Series，价格数据
        img_dir: 图片保存目录
        font: 字体对象或 None
        title: 图标题
        filename: 文件名
    """
    plt.figure(figsize=(10, 6))
    plt.hist(price_series, bins=50, color='skyblue', edgecolor='black')
    if font is not None:
        plt.title(title, fontproperties=font)
        plt.xlabel('价格', fontproperties=font)
        plt.ylabel('频数', fontproperties=font)
    else:
        plt.title(title)
        plt.xlabel('价格')
        plt.ylabel('频数')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, filename))
    plt.close()

def plot_corr_heatmap(df, num_cols, img_dir, font, filename='corr_heatmap.png'):
    """
    绘制数值特征相关性热力图并保存。
    参数：
        df: DataFrame，原始数据
        num_cols: 数值型特征列名列表
        img_dir: 图片保存目录
        font: 字体对象或 None
        filename: 文件名
    """
    corr = df[num_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    if font is not None:
        plt.title('数值特征相关性热力图', fontproperties=font)
    else:
        plt.title('数值特征相关性热力图')
    plt.tight_layout()
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, filename))
    plt.close() 