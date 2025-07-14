# SRC 目录结构说明

## 概述
src目录已经重新组织为模块化的包结构，便于维护和扩展。

## 目录结构
```
src/
├── __init__.py                 # 主包初始化，导出所有主要功能
├── config/                     # 配置管理模块
│   ├── __init__.py
│   └── config_utils.py         # 配置加载、获取、目录创建工具
├── data/                       # 数据处理模块
│   ├── __init__.py
│   ├── feature_utils.py        # 特征工程核心功能
│   ├── feature_blocks.py       # 特征块定义和函数
│   └── print_blocks.py         # 打印控制和数据分析
├── models/                     # 模型相关模块
│   ├── __init__.py
│   ├── regression/             # 回归模型
│   │   ├── __init__.py
│   │   ├── regression_train_predict.py    # 回归训练预测主流程
│   │   ├── cross_validate_regression.py   # 交叉验证
│   │   └── segmented_regression.py        # 分段回归
│   └── utils/                  # 模型工具
│       ├── __init__.py
│       ├── model_utils_tools.py # 模型工具函数（调参、评估等）
│       └── handle_model.py     # 模型保存加载
└── utils/                      # 通用工具模块
    ├── __init__.py
    ├── plot_utils.py           # 可视化工具
    ├── font_utils.py           # 字体工具
    └── tools.py                # 通用工具函数
```

## 模块说明

### 1. config/ - 配置管理
- **config_utils.py**: 配置文件的加载、解析、目录创建等工具函数
- 主要功能：`load_config()`, `get_config_value()`, `ensure_directories()`

### 2. data/ - 数据处理
- **feature_utils.py**: 特征工程的核心功能
  - `preprocess_data()`: 数据预处理主函数
  - `clean_data()`: 数据清洗
  - `add_features()`: 特征工程
  - `preprocess_features()`: 特征预处理
- **feature_blocks.py**: 特征块定义和具体实现
- **print_blocks.py**: 打印控制和数据分析报告

### 3. models/ - 模型相关
#### 3.1 regression/ - 回归模型
- **regression_train_predict.py**: 回归训练预测的主流程
  - 配置类：`RegressionTrainConfig`, `DataConfig`, `ModelConfig`, `TrainConfig`, `ParamTuningConfig`
  - 主函数：`regression_train_predict()`
- **cross_validate_regression.py**: 交叉验证
  - 配置类：`CrossValidateConfig`
  - 主函数：`cross_validate_regression()`
- **segmented_regression.py**: 分段回归
  - 配置类：`SegmentedRegressionConfig`
  - 主函数：`segmented_regression()`

#### 3.2 utils/ - 模型工具
- **model_utils_tools.py**: 模型相关的工具函数
  - 模型选择：`select_model()`
  - 参数调优：`tune_model_params()`
  - 特征重要性：`plot_feature_importance()`
  - 模型评估：`train_and_evaluate()`
  - 数据变换：`log1p_transform()`, `log1p_inverse()`
- **handle_model.py**: 模型保存和加载
  - `save_model()`, `load_model()`

### 4. utils/ - 通用工具
- **plot_utils.py**: 可视化工具
  - `plot_price_dist()`: 价格分布图
  - `plot_corr_heatmap()`: 相关性热力图
- **font_utils.py**: 字体工具
  - `get_chinese_font()`: 获取中文字体
- **tools.py**: 通用工具函数
  - `auto_bins()`: 自动分箱

## 使用方式

### 1. 从主包导入（推荐）
```python
from src import (
    preprocess_data, regression_train_predict, 
    RegressionTrainConfig, load_config
)
```

### 2. 从子模块导入
```python
from src.data import preprocess_data
from src.models.regression import regression_train_predict
from src.config import load_config
```

### 3. 在scripts中使用
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src import preprocess_data, load_config
```

## 优势

1. **模块化**: 功能按领域分组，便于理解和维护
2. **层次清晰**: 从配置→数据→模型→工具的逻辑层次
3. **易于扩展**: 新功能可以轻松添加到对应模块
4. **导入简化**: 主包提供统一的导入接口
5. **向后兼容**: 保持了原有的功能接口

## 注意事项

1. 所有相对导入都使用正确的包路径
2. 主包的`__init__.py`导出了所有主要功能
3. 子模块的`__init__.py`导出了该模块的主要功能
4. 脚本中的导入路径已经更新为新的结构 