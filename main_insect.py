import json
import argparse
from trainer_insect import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    
    # 添加昆虫特定的默认参数
    insect_defaults = {
        "max_stages": 5,                  # 最大虫态数
        "feat_dim": 512,                  # 特征维度
        "stage_weight": 0.3,              # 阶段预测损失权重
        "proto_momentum": 0.9,            # 原型更新动量
        "shared_proto_per_stage": 5,      # 每个阶段的共享原型数
        "proto_heads": 4,                 # 原型头数
        "vae_weight": 0.2,               # VAE损失权重
    }
    
    # 更新配置参数
    for k, v in insect_defaults.items():
        if k not in param:
            param[k] = v
    
    args.update(param)  # Add parameters from json

    # 调用昆虫训练流程
    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Insect-specific incremental learning framework.'
    )
    
    # 基础参数
    parser.add_argument('--config', type=str, default='./exps/insect_config.json',
                        help='Json file of settings.')
    
    # 昆虫特定参数
    parser.add_argument('--stages', type=int, default=5,
                        help='Maximum number of insect stages')
    parser.add_argument('--stage_loss', type=str, default='kl',
                        help='Stage prediction loss type: kl/ce/focal')
    parser.add_argument('--temporal', action='store_true',
                        help='Enable temporal modeling')
    parser.add_argument('--proto_pool', action='store_true', 
                        help='Enable prototype pool')
    
    return parser

if __name__ == '__main__':
    main()
