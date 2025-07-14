"""
二手车价格预测项目 - 主包
"""

# 配置相关
from .config.config_utils import load_config, get_config_value, ensure_directories

# 数据处理相关
from .data.feature_utils import preprocess_data, preprocess_features, clean_data, add_features
from .data.print_blocks import print_data_analysis

# 模型相关
from .models.regression.regression_train_predict import (
    regression_train_predict, RegressionTrainConfig, 
    DataConfig, ModelConfig, TrainConfig, ParamTuningConfig
)
from .models.regression.cross_validate_regression import (
    cross_validate_regression, CrossValidateConfig
)
from .models.regression.segmented_regression import (
    segmented_regression, SegmentedRegressionConfig
)
from .models.utils.handle_model import save_model, load_model
from .models.utils.model_utils_tools import (
    log1p_transform, select_model, plot_feature_importance,
    retrain_with_top_features, auto_linear_adjust, train_and_evaluate
)

# 工具相关
from .utils.plot_utils import plot_price_dist, plot_corr_heatmap
from .utils.font_utils import get_chinese_font
from .utils.tools import auto_bins

__all__ = [
    # 配置
    'load_config', 'get_config_value', 'ensure_directories',
    
    # 数据处理
    'preprocess_data', 'preprocess_features', 'clean_data', 'add_features',
    'print_data_analysis',
    
    # 模型训练
    'regression_train_predict', 'RegressionTrainConfig', 'DataConfig', 
    'ModelConfig', 'TrainConfig', 'ParamTuningConfig',
    'cross_validate_regression', 'CrossValidateConfig',
    'segmented_regression', 'SegmentedRegressionConfig',
    
    # 模型工具
    'save_model', 'load_model', 'log1p_transform', 'select_model',
    'plot_feature_importance', 'retrain_with_top_features', 
    'auto_linear_adjust', 'train_and_evaluate',
    
    # 可视化工具
    'plot_price_dist', 'plot_corr_heatmap', 'get_chinese_font',
    
    # 通用工具
    'auto_bins'
] 