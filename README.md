# 二手车价格预测实验项目

## 项目简介
本项目基于真实二手车数据，完成了数据清洗、特征工程、分段建模、主流回归模型（如XGBoost、CatBoost等）训练与预测的全流程，支持分段模型与融合模型的灵活切换，适合机器学习建模实战与工程化复现。

## 项目特色
- 🎯 **统一配置管理**: 使用 `config.yaml` 集中管理所有配置，易于维护和调整
- 🔧 **模块化设计**: 清晰的模块结构，便于扩展和复用
- 📊 **完整流程**: 从数据预处理到模型部署的完整机器学习流程
- 🚀 **工程化优化**: 参数缓存、自动目录创建、错误处理等工程化特性
- 📈 **多种模型**: 支持XGBoost、CatBoost、LightGBM等多种主流模型
- 🧩 **特征工程与清洗解耦**: 支持 data/pre/ 目录下自定义特征工程与清洗模块，主流程自动动态加载
- 📁 **输出路径自动化**: 所有图片、模型、预测结果等输出路径均自动拼接 config.yaml 配置，无需手动修改

## 目录结构
```
test/
├── config.yaml              # 统一配置文件
├── run.py                   # 主流程入口
├── main.py                  # 核心流程逻辑
├── data/                    # 数据文件目录
│   ├── used_car_train_20200313.csv
│   └── used_car_testB_20200421.csv
│   └── pre/                 # 特征工程与清洗模块
│       ├── feature_blocks.py    # 特征工程实现与顺序控制
│       ├── clean_data.py        # 数据清洗实现与顺序控制
│       └── README.md            # 维护与扩展规则说明
├── src/                     # 源代码目录
│   ├── config_utils.py      # 配置管理工具
│   ├── data/                # 数据处理相关
│   │   ├── feature_utils.py     # 数据预处理主入口
│   │   └── print_blocks.py      # 分块打印与可视化
│   ├── models/              # 模型相关
│   │   ├── regression/          # 回归模型
│   │   │   ├── regression_train_predict.py
│   │   │   ├── cross_validate_regression.py
│   │   │   └── segmented_regression.py
│   │   └── utils/               # 模型工具
│   │       ├── model_utils_tools.py
│   │       ├── handle_model.py
│   │       └── __init__.py
│   ├── utils/                # 通用工具
│   │   ├── plot_utils.py         # 可视化工具
│   │   ├── font_utils.py         # 字体工具
│   │   ├── tools.py              # 通用工具函数
│   │   └── __init__.py
│   └── __init__.py           # 主包初始化
├── results/                 # 结果输出目录
│   ├── models/              # 训练好的模型
│   ├── predictions/         # 预测结果
│   └── images/              # 图表和可视化
├── cache/                   # 缓存目录
│   └── param_cache.json     # 参数调优缓存
└── README.md               # 项目说明文档
```

## 主要模块说明

### 1. utils/ - 通用工具
- **plot_utils.py**: 价格分布、相关性热力图等可视化工具
- **font_utils.py**: 自动适配操作系统的中文字体
- **tools.py**: 自动分箱等通用函数

### 2. models/utils/ - 模型工具
- **model_utils_tools.py**: 特征重要性、模型选择、调参、评估等
- **handle_model.py**: 模型保存与加载

### 3. data/pre/ - 特征工程与清洗模块
- **feature_blocks.py**: 所有特征工程函数，顺序由 FEATURE_BLOCKS 控制，只暴露 default 统一入口
- **clean_data.py**: 所有数据清洗函数，顺序由 STEPS 控制，只暴露 default 统一入口
- **README.md**: 详细维护、更换、扩展规则说明

### 4. config_utils.py
- 配置文件加载、参数获取、自动创建输出目录

### 5. feature_utils.py
- 数据清洗、特征工程主入口，自动动态加载 data/pre/ 下的模块

### 6. print_blocks.py
- 分块打印数据分析报告，自动调用可视化工具

### 7. models/regression/
- 回归训练、交叉验证、分段回归等主流程

## 特征工程与清洗机制
- 支持通过 config.yaml 的 preprocess_module 配置动态切换特征工程模块
- data/pre/feature_blocks.py、clean_data.py 只暴露 default 统一入口，主流程无需关心内部细节
- 详细维护与扩展规则见 data/pre/README.md

## 输出路径自动化
- 所有输出（图片、模型、预测结果等）路径均通过 config.yaml 配置自动拼接，无需手动修改
- 结果自动归档到 results/ 下对应子目录

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行主流程
```bash
python run.py
```

### 3. 配置说明
- 详见 config.yaml，支持灵活切换数据路径、输出目录、特征工程、模型参数等

## 贡献指南
欢迎提交 Issue 和 Pull Request 改进项目！
- 新增特征/清洗函数请参考 data/pre/README.md
- 新增模型工具请参考 models/utils/model_utils_tools.py
- 其他建议和优化也欢迎反馈
