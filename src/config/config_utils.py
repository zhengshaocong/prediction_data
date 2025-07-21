import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = './config.yaml') -> Dict[str, Any]:
    """
    加载配置文件，并自动拼接output_subdir到所有output目录前。
    支持auto_output_subdir逻辑：
    - 若auto_output_subdir为false，则输出目录为results/output_subdir/
    - 若auto_output_subdir为true，且分段建模启用，则输出目录为results/output_subdir_modeltype_segmentcol/
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    output_subdir = config.get('output_subdir', 'default')
    auto_output = config.get('auto_output_subdir', False)
    model_type = config.get('model', {}).get('type', 'model')
    segment_col = config.get('segmentation', {}).get('segment_col', 'segment')
    segmentation_enabled = config.get('segmentation', {}).get('enabled', False)

    # 动态拼接output_subdir
    if auto_output:
        if segmentation_enabled:
            output_subdir = f"{output_subdir}_{model_type}_{segment_col}"
        else:
            output_subdir = f"{output_subdir}"
    base_prefix = os.path.join('results', output_subdir)
    # 只拼接 output 目录
    for k, v in config.get('output', {}).items():
        if isinstance(v, str) and not os.path.isabs(v):
            config['output'][k] = os.path.join(base_prefix, v)
    return config

def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    从配置字典中获取嵌套键值
    
    Args:
        config: 配置字典
        key_path: 键路径，如 'data.train_path'
        default: 默认值
        
    Returns:
        配置值
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def ensure_directories(config: Dict[str, Any]):
    """
    确保输出目录存在
    
    Args:
        config: 配置字典
    """
    output_config = config.get('output', {})
    
    # 确保基础输出目录存在
    base_dir = output_config.get('base_dir', './results')
    os.makedirs(base_dir, exist_ok=True)
    
    # 确保子目录存在
    for subdir in ['models', 'predictions', 'images']:
        dir_path = output_config.get(f'{subdir}_dir', f'{base_dir}/{subdir}')
        os.makedirs(dir_path, exist_ok=True) 