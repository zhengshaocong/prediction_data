import yaml
import os

def load_config(config_path):
    """
    加载yaml配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_config_value(config, key_path, default=None):
    """
    按点路径获取配置项，如 'data.train_path'
    """
    keys = key_path.split('.')
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val

def ensure_directories(config):
    """
    根据config自动创建输出目录
    """
    output = config.get('output', {})
    for key in ['models_dir', 'predictions_dir', 'images_dir']:
        d = output.get(key)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True) 