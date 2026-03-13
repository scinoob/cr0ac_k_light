import json
# 数据配置
DATA_CONFIG = {
    # 数据集名称: 'crack500', 'cfd', 'sun520'
    'dataset_name': 'crack500',
    
    # 数据根目录
    'data_root': '../../data/crack500',
    
    # 输入尺寸
    'input_size': 512,
    
    # 批次大小
    'batch_size': 12,
    
    # 工作进程数
    'num_workers': 4,
    
    # Sun520滑动窗口配置
    'crop_size': 512,
    'stride': 256,
    'use_sliding_window': True,
}

# 模型配置
MODEL_CONFIG = {
    # 输入通道数
    'in_channels': 3,
    
    # 输出类别数
    'num_classes': 1,
    
    # 基础通道数
    'base_channels': 32,
    
    # 输入尺寸
    'input_size': 512,
    
    # Mamba状态维度
    'd_state': 16,
    
    # Mamba卷积核大小
    'd_conv': 3,
    
    # Mamba扩展因子
    'expand': 2,
    
    # 各阶段注意力头数
    'num_heads_list': [2, 2, 4, 4],
    
    # ViT窗口大小
    'window_size': 7,
    
    # 是否使用GBC模块
    'use_gbc': False,
    
    # 是否使用ASPP模块
    'use_aspp': False,
    
    # @kimi 新增: 随机深度丢弃概率
    # 理由: 实现深度网络的随机深度(Stochastic Depth)机制，随网络深度线性增加丢弃概率
    # 例如: 0.2 表示最深层有20%的概率被丢弃
    'drop_path_rate': 0.0,
}

# 训练配置
TRAIN_CONFIG = {
    # 训练轮数
    'epochs': 100,
    
    # 学习率
    'lr': 1e-4,
    
    # 权重衰减
    'weight_decay': 1e-4,
    
    # 学习率调度器
    'scheduler': 'cosine',  # 'step', 'cosine', 'plateau'
    
    # 预热轮数
    'warmup_epochs': 5,
    
    # 混合精度训练
    'use_amp': False,
    
    # 梯度裁剪
    'grad_clip': 1.0,
    
    # 保存目录
    'save_dir': './checkpoints',
    
    # 日志间隔
    'log_interval': 10,
    
    # 保存间隔
    'save_interval': 10,
}

# 损失函数配置
LOSS_CONFIG = {
    # 损失类型
    'loss_type': 'combined',  # 'dice', 'focal', 'combined', 'bce_dice'
    
    # Dice损失权重
    'dice_weight': 1.0,
    
    # Focal损失权重
    'focal_weight': 1.0,
    
    # Focal损失alpha
    'focal_alpha': 0.25,
    
    # Focal损失gamma
    'focal_gamma': 2.0,
}

# 评估配置
EVAL_CONFIG = {
    # 评估阈值
    'threshold': 0.5,
    
    # 评估指标
    'metrics': ['precision', 'recall', 'f1', 'iou', 'miou'],
}

# Grad-CAM配置
GRADCAM_CONFIG = {
    # 目标层
    'target_layers': ['encoder1', 'encoder2', 'encoder3', 'encoder4'],
    
    # 输出目录
    'output_dir': './gradcam_results',
}

# 完整配置
CONFIG = {
    'data': DATA_CONFIG,
    'model': MODEL_CONFIG,
    'train': TRAIN_CONFIG,
    'loss': LOSS_CONFIG,
    'eval': EVAL_CONFIG,
    'gradcam': GRADCAM_CONFIG,
}


def get_config():
    """获取完整配置"""
    return CONFIG


def update_config(config_dict: dict):
    """更新配置"""
    global CONFIG
    for key, value in config_dict.items():
        if key in CONFIG:
            CONFIG[key].update(value)
    return CONFIG


if __name__ == "__main__":
    print("默认配置:")
    print(json.dumps(CONFIG, indent=2))
