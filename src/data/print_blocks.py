# 打印控制模块

from ..utils import plot_utils

# PRINT_BLOCKS 数组，控制主脚本各部分打印内容
PRINT_BLOCKS = [
    'data_read',        # 读取数据
    'data_clean',       # 数据清洗
    'feature_engineer', # 特征工程
    'info',             # 基本信息
    'missing',          # 缺失值统计
    'price_dist',       # 目标变量分布
    'num_dist',         # 数值型特征分布
    'cat_dist',         # 分类型特征分布
    'corr_heatmap',     # 相关性热力图
    'model',            # 线性回归建模与预测
    'feature_cols',     # 特征列打印
]

def print_block(block, *args, **kwargs):
    """
    分块打印函数。只有block在PRINT_BLOCKS数组中时才打印。
    """
    if block in PRINT_BLOCKS:
        print(*args, **kwargs)


def print_data_analysis(df, img_dir, font, feature_cols, enable_feature_engineering):
    # 数据维度
    print_block('data_read', f"数据维度: {df.shape}")
    # 基本信息
    print_block('info', "\n===== 2. 基本信息 =====")
    print_block('info', df.info())
    # 缺失值统计
    print_block('missing', "\n===== 3. 缺失值统计 =====")
    print_block('missing', df.isnull().sum())
    # 目标变量分布
    print_block('price_dist', "\n===== 4. 目标变量（price）分布（包含价格为0） =====")
    print_block('price_dist', df['price'].describe())
    plot_utils.plot_price_dist(df['price'], img_dir, font, title='价格分布', filename='price_dist.png')
    # 数值型特征分布
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # 假设 config 变量可用，否则需传参
    id_column = config.get('id_column', 'SaleID') if 'config' in locals() else 'SaleID'
    prediction_column = config.get('prediction_column', 'price') if 'config' in locals() else 'price'
    num_cols = [col for col in num_cols if col not in [id_column, prediction_column]]
    print_block('num_dist', "\n===== 5. 数值型特征分布 =====")
    print_block('num_dist', df[num_cols].describe())
    # 分类型特征分布
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols += [col for col in df.columns if 'Type' in col or col in ['model','brand','bodyType','fuelType','gearbox','notRepairedDamage','regionCode','seller','offerType']]
    cat_cols = list(set(cat_cols))
    print_block('cat_dist', "\n===== 6. 分类型特征分布（前5个类别） =====")
    for col in cat_cols:
        print_block('cat_dist', f"\n{col} 分类分布:")
        print_block('cat_dist', df[col].value_counts(dropna=False).head())
    # 相关性热力图
    print_block('corr_heatmap', "\n===== 7. 数值特征相关性热力图 =====")
    plot_utils.plot_corr_heatmap(df, num_cols, img_dir, font, filename='corr_heatmap.png')
    # 特征列打印
    print_block('feature_cols', "特征列：", feature_cols) 