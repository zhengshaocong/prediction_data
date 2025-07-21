# 内部步骤控制数组，决定哪些清洗函数会被调用
STEPS = [
    'price',        # 价格异常过滤
    # 'kilometer',   # 公里数异常过滤
    # 'reg_date',    # 注册日期合理性
    # 'reg_year',    # 注册年份合理性
    'missing',      # 关键特征缺失
]

def clean_price(df):
    if 'price' in df.columns:
        df = df[(df['price'] > 100) & (df['price'] < 100000)]
    return df

def clean_kilometer(df):
    if 'kilometer' in df.columns:
        df = df[(df['kilometer'] > 0) & (df['kilometer'] < 50)]
    return df

def clean_reg_date(df):
    if 'regDate' in df.columns and 'creatDate' in df.columns:
        df = df[df['regDate'] <= df['creatDate']]
    return df

def clean_reg_year(df):
    if 'regYear' in df.columns:
        df = df[(df['regYear'] >= 1980) & (df['regYear'] <= 2025)]
    return df

def clean_missing(df):
    for col in ['regDate', 'kilometer', 'power']:
        if col in df.columns:
            df = df.dropna(subset=[col])
    return df

# 步骤名到函数的映射
STEP_FUNCS = {
    'price': clean_price,
    'kilometer': clean_kilometer,
    'reg_date': clean_reg_date,
    'reg_year': clean_reg_year,
    'missing': clean_missing,
}

def default(df):
    for step in STEPS:
        func = STEP_FUNCS.get(step)
        if func is not None:
            df = func(df)
    return df