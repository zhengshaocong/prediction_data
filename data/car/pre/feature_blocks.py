import pandas as pd

# 特征工程模块数组，控制启用哪些特征
FEATURE_BLOCKS = [
    'car_age',           # 车龄特征（creatYear - regYear）
    'model_count',       # 车型销量特征
    'region_count',      # 区域销量特征
    'model_target_mean', # 车型目标均值编码
    'region_target_mean',# 区域目标均值编码
    'car_age_bin',       # 车龄分箱特征
    'power_bin',         # power分箱特征
    'km_bin',            # 公里数分箱特征（kilometer分区间）
    'reg_month',         # 注册月份/季度特征（reg_month, reg_quarter）
    'creat_month',       # 信息创建月份/季度特征（creat_month, creat_quarter）
    'reg_creat_diff',    # 注册与创建时间差（月）特征
    'power_age_ratio',   # 功率与车龄比值特征（power / (car_age+1)）
    'brand_count',       # 品牌销量特征（品牌出现次数）
    'brand_std',         # 品牌价格标准差特征
    'age_km',            # 车龄与公里数交互特征（car_age * kilometer）
    'v_pca',             # 匿名特征主成分
    'v_bins',            # 匿名特征分箱
    'v_target_mean',     # 匿名特征分箱目标均值编码
    'v_stats',           # 匿名特征统计量
    'name_length',       # name字段长度
    'missing_flags',     # 缺失值指示特征
    'v_brand_interactions', # 匿名特征与品牌均价交互
    'v_cluster',         # 匿名特征KMeans聚类
    'v_stats_ext',       # 匿名特征统计量扩展
    'reg_season',
    'creat_season',
    'reg_creat_year_diff',
    'reg_weekend',
    'has_main_missing',
    'model_brand_combo',
    'v_standardize',
    'v_minmax',
    'v_bin',
    'v_agg_stats',
    'v_pca3',
    'v_pair_diff',
]

# 依赖声明装饰器
def feature_depends_on(*deps):
    def decorator(func):
        func.dependencies = list(deps)
        return func
    return decorator

# 各特征工程实现

def car_age(df, train_df=None):
    if 'regDate' in df.columns and 'creatDate' in df.columns:
        df['regYear'] = df['regDate'] // 10000
        df['creatYear'] = df['creatDate'] // 10000
        df['car_age'] = df['creatYear'] - df['regYear']
    else:
        print("car_age跳过，缺少 regDate 或 creatDate")
    return df

def brand_mean(df, train_df=None):
    if train_df is not None:
        brand_mean = train_df.groupby('brand')['price'].mean()
        df['brand_mean_price'] = df['brand'].map(brand_mean)
    else:
        df['brand_mean_price'] = df['brand'].map(df.groupby('brand')['price'].mean())
    return df

def km_bin(df, train_df=None):
    df['km_bin'] = pd.cut(df['kilometer'], bins=[0, 5, 10, 15, 20, 30, 50], labels=False)
    return df

def reg_month(df, train_df=None):
    df['reg_month'] = df['regDate'] // 100 % 100
    return df

def creat_month(df, train_df=None):
    df['creat_month'] = df['creatDate'] // 100 % 100
    return df

@feature_depends_on('reg_month')
def reg_season(df, train_df=None):
    if 'reg_month' not in df.columns:
        df['reg_month'] = df['regDate'] // 100 % 100
    df['reg_season'] = pd.cut(df['reg_month'], bins=[0,3,6,9,12], labels=['春','夏','秋','冬'], right=True)
    return df

@feature_depends_on('creat_month')
def creat_season(df, train_df=None):
    if 'creat_month' not in df.columns:
        df['creat_month'] = df['creatDate'] // 100 % 100
    df['creat_season'] = pd.cut(df['creat_month'], bins=[0,3,6,9,12], labels=['春','夏','秋','冬'], right=True)
    return df

@feature_depends_on('reg_month', 'creat_month')
def reg_creat_diff(df, train_df=None):
    if 'reg_month' not in df.columns:
        df['reg_month'] = df['regDate'] // 100 % 100
    if 'creat_month' not in df.columns:
        df['creat_month'] = df['creatDate'] // 100 % 100
    if 'regYear' not in df.columns:
        df['regYear'] = df['regDate'] // 10000
    if 'creatYear' not in df.columns:
        df['creatYear'] = df['creatDate'] // 10000
    df['reg_creat_month_diff'] = (df['creatYear'] - df['regYear']) * 12 + (df['creat_month'] - df['reg_month'])
    return df

def brand_count(df, train_df=None):
    if train_df is not None:
        brand_count = train_df['brand'].value_counts()
        df['brand_count'] = df['brand'].map(brand_count)
    else:
        df['brand_count'] = df['brand'].map(df['brand'].value_counts())
    return df

def brand_std(df, train_df=None):
    if train_df is not None:
        brand_std = train_df.groupby('brand')['price'].std()
        df['brand_price_std'] = df['brand'].map(brand_std)
    else:
        df['brand_price_std'] = df.groupby('brand')['price'].transform('std')
    return df

@feature_depends_on('car_age')
def age_km(df, train_df=None):
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['age_km'] = df['car_age'] * df['kilometer']
    return df

@feature_depends_on('car_age')
def power_age_ratio(df, train_df=None):
    if 'car_age' in df.columns and 'power' in df.columns:
        df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
    return df

def model_mean(df, train_df=None):
    # 车型均价
    if train_df is not None:
        model_mean = train_df.groupby('model')['price'].mean()
        df['model_mean_price'] = df['model'].map(model_mean)
    else:
        df['model_mean_price'] = df['model'].map(df.groupby('model')['price'].mean())
    return df

def model_count(df, train_df=None):
    # 车型销量
    if train_df is not None:
        model_count = train_df['model'].value_counts()
        df['model_count'] = df['model'].map(model_count)
    else:
        df['model_count'] = df['model'].map(df['model'].value_counts())
    return df

def region_mean(df, train_df=None):
    # 区域均价
    if train_df is not None:
        region_mean = train_df.groupby('regionCode')['price'].mean()
        df['region_mean_price'] = df['regionCode'].map(region_mean)
    else:
        df['region_mean_price'] = df['regionCode'].map(df.groupby('regionCode')['price'].mean())
    return df

def region_count(df, train_df=None):
    # 区域销量
    if train_df is not None:
        region_count = train_df['regionCode'].value_counts()
        df['region_count'] = df['regionCode'].map(region_count)
    else:
        df['region_count'] = df['regionCode'].map(df['regionCode'].value_counts())
    return df

def model_target_mean(df, train_df=None):
    # 车型目标均值编码
    if train_df is not None:
        model_mean = train_df.groupby('model')['price'].mean()
        df['model_target_mean'] = df['model'].map(model_mean)
    else:
        df['model_target_mean'] = df.groupby('model')['price'].transform('mean')
    return df

def region_target_mean(df, train_df=None):
    # 区域目标均值编码
    if train_df is not None:
        region_mean = train_df.groupby('regionCode')['price'].mean()
        df['region_target_mean'] = df['regionCode'].map(region_mean)
    else:
        df['region_target_mean'] = df.groupby('regionCode')['price'].transform('mean')
    return df

@feature_depends_on('car_age')
def car_age_bin(df, train_df=None):
    # 车龄分箱
    if 'car_age' in df.columns:
        df['car_age_bin'] = pd.cut(df['car_age'], bins=[-1,1,3,5,8,12,100], labels=False)
    else:
        print("car_age_bin跳过，缺少car_age")
    return df

def power_bin(df, train_df=None):
    # power分箱
    df['power_bin'] = pd.cut(df['power'], bins=[-1,50,100,150,200,300,10000], labels=False)
    return df

def v_pca(df, train_df=None, n_components=3):
    # 匿名特征主成分
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    if len(v_cols) > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, len(v_cols)))
        v_pca = pca.fit_transform(df[v_cols].fillna(-1))
        for i in range(v_pca.shape[1]):
            df[f'v_pca_{i+1}'] = v_pca[:,i]
    return df

def v_bins(df, train_df=None, n_bins=5):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    for col in v_cols:
        df[f'{col}_bin'] = pd.qcut(df[col], n_bins, labels=False, duplicates='drop')
    return df

def v_target_mean(df, train_df=None, n_bins=5):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    for col in v_cols:
        bin_col = f'{col}_bin'
        if bin_col not in df:
            df[bin_col] = pd.qcut(df[col], n_bins, labels=False, duplicates='drop')
        if train_df is not None:
            means = train_df.groupby(bin_col)['price'].mean()
        else:
            means = df.groupby(bin_col)['price'].mean()
        df[f'{col}_target_mean'] = df[bin_col].map(means)
    return df

def v_stats(df, train_df=None):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    if len(v_cols) > 0:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
    return df

def name_length(df, train_df=None):
    if 'name' in df.columns:
        df['name_length'] = df['name'].astype(str).apply(len)
    return df

def missing_flags(df, train_df=None):
    # 缺失值指示特征
    for col in ['notRepairedDamage','fuelType','bodyType']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
    return df

@feature_depends_on('brand_mean')
def v_brand_interactions(df, train_df=None):
    # 匿名特征与品牌均价交互
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    if 'brand_mean_price' in df.columns:
        for col in v_cols:
            df[f'{col}_x_brand_mean'] = df[col] * df['brand_mean_price']
    return df

def v_cluster(df, train_df=None, n_clusters=5):
    # 匿名特征KMeans聚类
    from sklearn.cluster import KMeans
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    if len(v_cols) > 0:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['v_cluster'] = kmeans.fit_predict(df[v_cols].fillna(-1))
    return df

def v_stats_ext(df, train_df=None):
    # 匿名特征统计量扩展
    import numpy as np
    from scipy.stats import skew, kurtosis
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    if len(v_cols) > 0:
        df['v_median'] = df[v_cols].median(axis=1)
        df['v_q25'] = df[v_cols].quantile(0.25, axis=1)
        df['v_q75'] = df[v_cols].quantile(0.75, axis=1)
        df['v_skew'] = df[v_cols].apply(lambda x: skew(x, nan_policy='omit'), axis=1)
        df['v_kurt'] = df[v_cols].apply(lambda x: kurtosis(x, nan_policy='omit'), axis=1)
    return df

def info_length(df, train_df=None):
    # 统计所有object类型字段的字符串长度总和
    obj_cols = df.select_dtypes(include='object').columns
    df['info_length'] = df[obj_cols].astype(str).apply(lambda x: sum([len(i) for i in x]), axis=1)
    return df

def reg_creat_year_diff(df, train_df=None):
    if 'regDate' in df.columns and 'creatDate' in df.columns:
        reg_year = df['regDate'] // 10000
        creat_year = df['creatDate'] // 10000
        df['reg_creat_year_diff'] = creat_year - reg_year
    return df

@feature_depends_on('model', 'brand')
def model_brand_combo(df, train_df=None):
    if 'model' in df.columns and 'brand' in df.columns:
        df['model_brand'] = df['model'].astype(str) + '_' + df['brand'].astype(str)
    return df

def has_main_missing(df, train_df=None):
    main_cols = ['model', 'brand', 'regDate', 'kilometer', 'power']
    df['has_main_missing'] = df[main_cols].isnull().any(axis=1).astype(int)
    return df

def reg_weekend(df, train_df=None):
    if 'regDate' in df.columns:
        reg_date = pd.to_datetime(df['regDate'], format='%Y%m%d', errors='coerce')
        df['reg_weekend'] = reg_date.dt.weekday.isin([5,6]).astype(int)
    return df

# 匿名特征标准化
@feature_depends_on()
def v_standardize(df, train_df=None):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    for col in v_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f'{col}_std'] = (df[col] - mean) / std
        else:
            df[f'{col}_std'] = 0
    return df

# 匿名特征归一化
@feature_depends_on()
def v_minmax(df, train_df=None):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    for col in v_cols:
        minv = df[col].min()
        maxv = df[col].max()
        if maxv > minv:
            df[f'{col}_minmax'] = (df[col] - minv) / (maxv - minv)
        else:
            df[f'{col}_minmax'] = 0
    return df

# 匿名特征分箱（等宽）
@feature_depends_on()
def v_bin(df, train_df=None, n_bins=5):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    for col in v_cols:
        df[f'{col}_bin5'] = pd.cut(df[col], bins=n_bins, labels=False)
    return df

# 匿名特征均值/方差/最大/最小
@feature_depends_on()
def v_agg_stats(df, train_df=None):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    if len(v_cols) > 0:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
    return df

# 匿名特征主成分分析（PCA，前3主成分）
@feature_depends_on()
def v_pca3(df, train_df=None):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    if len(v_cols) >= 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        v_pca = pca.fit_transform(df[v_cols].fillna(-1))
        for i in range(3):
            df[f'v_pca3_{i+1}'] = v_pca[:, i]
    return df

# 匿名特征两两差值
@feature_depends_on()
def v_pair_diff(df, train_df=None):
    v_cols = [f'v_{i}' for i in range(15) if f'v_{i}' in df.columns]
    new_cols = {}
    for i in range(len(v_cols)):
        for j in range(i+1, len(v_cols)):
            new_col = f'{v_cols[i]}_minus_{v_cols[j]}'
            new_cols[new_col] = df[v_cols[i]] - df[v_cols[j]]
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df

# 特征名到函数的映射
feature_block_funcs = {
    'car_age': car_age,
    'km_bin': km_bin,
    'reg_month': reg_month,
    'creat_month': creat_month,
    'reg_season': reg_season,
    'creat_season': creat_season,
    'reg_creat_diff': reg_creat_diff,
    'brand_count': brand_count,
    'brand_std': brand_std,
    'age_km': age_km,
    'power_age_ratio': power_age_ratio,
    'model_mean': model_mean,
    'model_count': model_count,
    'region_count': region_count,
    'model_target_mean': model_target_mean,
    'region_target_mean': region_target_mean,
    'car_age_bin': car_age_bin,
    'power_bin': power_bin,
    'v_pca': v_pca,
    'v_bins': v_bins,
    'v_target_mean': v_target_mean,
    'v_stats': v_stats,
    'name_length': name_length,
    'missing_flags': missing_flags,
    'v_brand_interactions': v_brand_interactions,
    'v_cluster': v_cluster,
    'v_stats_ext': v_stats_ext,
    'reg_creat_year_diff': reg_creat_year_diff,
    'model_brand_combo': model_brand_combo,
    'info_length': info_length,
    'has_main_missing': has_main_missing,
    'reg_weekend': reg_weekend,
    'v_standardize': v_standardize,
    'v_minmax': v_minmax,
    'v_bin': v_bin,
    'v_agg_stats': v_agg_stats,
    'v_pca3': v_pca3,
    'v_pair_diff': v_pair_diff,
}

# 自动生成依赖关系字典
feature_dependencies = {}
for name, func in feature_block_funcs.items():
    if hasattr(func, 'dependencies'):
        feature_dependencies[name] = func.dependencies

def default(df, train_df=None, feature_blocks=None):
    """
    按 feature_blocks 顺序依次应用本文件内的特征工程函数。
    :param df: 需要处理的数据
    :param train_df: 训练集数据（部分特征工程用）
    :param feature_blocks: 特征工程模块名列表（如为 None 则用本文件 FEATURE_BLOCKS）
    :return: 处理后的 DataFrame
    """
    if feature_blocks is None:
        feature_blocks = FEATURE_BLOCKS
    # 按 FEATURE_BLOCKS 顺序排序，保证依赖先于被依赖特征工程执行
    block_order = {name: i for i, name in enumerate(FEATURE_BLOCKS)}
    feature_blocks = sorted(feature_blocks, key=lambda x: block_order.get(x, 9999))
    for block in feature_blocks:
        func = feature_block_funcs.get(block)
        if func is not None:
            df = func(df, train_df)
            # print(f"应用特征工程: {block} 后列名: {df.columns.tolist()}")  # 调试
    return df

# 只暴露 default
__all__ = ['default'] 