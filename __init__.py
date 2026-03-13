"""
裂缝分割网络包
基于Mamba与ViT双分支的非对称UNet架构

主要组件：
- CrackSegmentationNet: 完整的裂缝分割网络
- 各种模块：Mamba分支、ViT分支、双分支融合等
- 损失函数：Dice Loss、Focal Loss、组合损失等
- 工具：评估指标、Grad-CAM可视化等
"""
from .models.crack_net import CrackSegmentationNet
from .modules import (
    MambaBranch,
    ViTBranch,
    DualBranchModuleA,
    Encoder,
    MAMBA_AVAILABLE
)
from .utils.losses import CombinedLoss, DiceLoss, FocalLoss
from .utils.metrics import MetricCalculator, count_parameters
from .utils.gradcam import CrackGradCAM, GradCAM

__version__ = '1.0.0'

__all__ = [
    'CrackSegmentationNet',
    'MambaBranch',
    'ViTBranch',
    'DualBranchModuleA',
    'Encoder',
    'MAMBA_AVAILABLE',
    'CombinedLoss',
    'DiceLoss',
    'FocalLoss',
    'MetricCalculator',
    'count_parameters',
    'CrackGradCAM',
    'GradCAM'
]