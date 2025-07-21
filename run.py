import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description='二手车价格预测主流程')
    parser.add_argument('--config', type=str, default='./config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 读取配置，判断是否特征选择模式
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config.get('feature_selection_mode', False):
        print("检测到 feature_selection_mode=True，自动切换到特征工程组合搜索流程。")
        os.system(f'{sys.executable} scripts/feature_selection.py')
        sys.exit(0)

    # 运行主流程
    from main import run_pipeline
    run_pipeline(args.config)

if __name__ == '__main__':
    main()