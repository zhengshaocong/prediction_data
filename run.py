import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import yaml
from main import run_pipeline

def main():
    parser = argparse.ArgumentParser(description='二手车价格预测主流程')
    parser.add_argument('--config', type=str, default='./config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 运行主流程
    run_pipeline(args.config)

if __name__ == "__main__":
    main() 