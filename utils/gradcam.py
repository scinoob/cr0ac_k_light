"""
Grad-CAM可视化实现
用于分析裂缝分割网络各层的注意力区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)
    
    通过计算目标类别相对于特征图的梯度，生成类别激活热力图，
    用于可视化网络关注的区域。
    
    参数：
        model: 目标模型
        target_layers: 目标层名称列表
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str]
    ):
        self.model = model
        self.target_layers = target_layers
        
        # 存储特征和梯度
        self.features: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        
        # 注册钩子
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和后向钩子"""
        def forward_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # 遍历模型，找到目标层并注册钩子
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(backward_hook(name)))
                print(f"已注册钩子: {name}")
    
    def __call__(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成Grad-CAM热力图
        
        参数：
            x: 输入图像 (B, 3, H, W)
            target: 目标掩码，用于计算目标分数（可选）
            
        返回：
            各层的Grad-CAM热力图字典
        """
        return self.generate(x, target)
    
    def generate(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成Grad-CAM热力图
        
        参数：
            x: 输入图像 (B, 3, H, W)
            target: 目标掩码（可选）
            
        返回：
            各层的Grad-CAM热力图字典
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(x)
        
        # 计算目标分数
        if target is not None:
            # 使用目标区域的平均预测值作为分数
            score = (output * target).sum() / (target.sum() + 1e-6)
        else:
            # 使用整体预测的平均值
            score = output.mean()
        
        # 反向传播
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # 生成各层的Grad-CAM
        cam_dict = {}
        for name in self.target_layers:
            if name in self.features and name in self.gradients:
                cam = self._compute_cam(
                    self.features[name],
                    self.gradients[name]
                )
                cam_dict[name] = cam
        
        return cam_dict
    
    def _compute_cam(
        self,
        features: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        计算单层的Grad-CAM
        
        参数：
            features: 特征图 (B, C, H, W)
            gradients: 梯度图 (B, C, H, W)
            
        返回：
            CAM热力图 (B, 1, H, W)
        """
        # 全局平均池化梯度，得到各通道权重
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # 加权求和
        cam = (weights * features).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # ReLU激活
        cam = F.relu(cam)
        
        # 归一化
        cam = cam / (cam.max() + 1e-6)
        
        return cam
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ 改进版本
    
    使用二阶和三阶梯度信息，提供更精确的定位
    
    参数：
        model: 目标模型
        target_layers: 目标层名称列表
    """
    
    def _compute_cam(
        self,
        features: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Grad-CAM++热力图
        
        使用加权组合公式：
        α = ReLU(∂y/∂A) / (ReLU(∂y/∂A) + ∂²y/∂A² * A + 0.5 * ∂³y/∂A³ * A²)
        """
        # 计算alpha权重
        grad_pow2 = gradients ** 2
        grad_pow3 = gradients ** 3
        
        # 计算分母
        sum_features = features.sum(dim=(2, 3), keepdim=True)
        alpha_numer = grad_pow2
        alpha_denom = 2 * grad_pow2 + sum_features * grad_pow3 + 1e-6
        alpha = alpha_numer / alpha_denom
        
        # ReLU
        alpha = F.relu(gradients) * alpha
        
        # 全局平均池化alpha
        weights = alpha.mean(dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = (weights * features).sum(dim=1, keepdim=True)
        
        # ReLU激活
        cam = F.relu(cam)
        
        # 归一化
        cam = cam / (cam.max() + 1e-6)
        
        return cam


class CrackGradCAM:
    """
    针对裂缝分割网络的Grad-CAM封装
    
    提供便捷的接口来可视化编码器和解码器各层的注意力
    
    参数：
        model: 裂缝分割模型
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        # 自动检测可用的目标层
        self.encoder_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4']
        self.decoder_layers = ['decoder_final']
        
        # 实际可用的层
        self.available_layers = self._find_available_layers()
        
        # 创建Grad-CAM实例
        self.gradcam = None
    
    def _find_available_layers(self) -> List[str]:
        """查找模型中可用的层"""
        available = []
        for name, module in self.model.named_modules():
            if 'encoder' in name or 'decoder' in name or 'dual_branch' in name:
                if len(name.split('.')) <= 2:  # 避免太深的子模块
                    available.append(name)
        return available
    
    def generate_cams(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成指定层的Grad-CAM
        
        参数：
            x: 输入图像
            target: 目标掩码
            layers: 指定层列表，默认使用所有编码器层
            
        返回：
            各层的Grad-CAM字典
        """
        if layers is None:
            layers = self.available_layers[:4]  # 默认使用前4层
        
        # 创建Grad-CAM实例
        if self.gradcam is not None:
            self.gradcam.remove_hooks()
        
        self.gradcam = GradCAM(self.model, layers)
        
        # 生成CAM
        cams = self.gradcam.generate(x, target)
        
        return cams
    
    def visualize(
        self,
        image: torch.Tensor,
        cams: Dict[str, torch.Tensor],
        target: Optional[torch.Tensor] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 10)
    ) -> np.ndarray:
        """
        可视化Grad-CAM结果
        
        参数：
            image: 原始图像 (3, H, W)
            cams: 各层的CAM字典
            target: 目标掩码（可选）
            output_path: 输出路径（可选）
            figsize: 图像大小
            
        返回：
            可视化结果图像
        """
        # 准备原始图像
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image.transpose(1, 2, 0)
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # 创建子图
        n_layers = len(cams)
        n_cols = min(4, n_layers + 1)
        n_rows = (n_layers + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # 显示原图
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 显示目标掩码
        if target is not None:
            if isinstance(target, torch.Tensor):
                target = target.detach().cpu().numpy()
            if target.ndim == 3:
                target = target.squeeze(0)
            axes[0, 1].imshow(target, cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
        
        # 显示各层CAM
        for idx, (name, cam) in enumerate(cams.items()):
            row = (idx + 2) // n_cols
            col = (idx + 2) % n_cols
            
            if row < n_rows and col < n_cols:
                # 获取CAM
                if isinstance(cam, torch.Tensor):
                    cam = cam.detach().cpu().numpy()
                if cam.ndim == 4:
                    cam = cam.squeeze(0).squeeze(0)
                elif cam.ndim == 3:
                    cam = cam.squeeze(0)
                
                # 上采样到原图尺寸
                h, w = image.shape[:2]
                cam_resized = cv2.resize(cam, (w, h))
                
                # 转换为热力图
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                
                # 叠加到原图
                overlay = 0.5 * image + 0.5 * heatmap
                
                axes[row, col].imshow(overlay)
                axes[row, col].set_title(f'{name}')
                axes[row, col].axis('off')
        
        # 隐藏空白子图
        for idx in range(len(cams) + 2, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"保存Grad-CAM可视化到: {output_path}")
        
        # 转换为numpy数组
        fig.canvas.draw()
        result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return result
    
    def cleanup(self):
        """清理资源"""
        if self.gradcam is not None:
            self.gradcam.remove_hooks()
            self.gradcam = None


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    将CAM叠加到图像上
    
    参数：
        image: 原始图像 (H, W, 3)，值域[0, 1]
        cam: CAM热力图 (H, W)，值域[0, 1]
        alpha: 叠加权重
        colormap: OpenCV colormap
        
    返回：
        叠加后的图像
    """
    # 转换为uint8
    cam_uint8 = np.uint8(255 * cam)
    
    # 应用colormap
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # 叠加
    overlay = (1 - alpha) * image + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_multi_layer_cam(
    image: np.ndarray,
    cams: Dict[str, np.ndarray],
    titles: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8)
) -> np.ndarray:
    """
    多层CAM可视化
    
    参数：
        image: 原始图像
        cams: 各层CAM字典
        titles: 标题列表
        output_path: 输出路径
        figsize: 图像大小
        
    返回：
        可视化结果
    """
    n_cams = len(cams)
    n_cols = n_cams + 1
    n_rows = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 显示原图
    axes[0].imshow(image)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    # 显示各层CAM
    for idx, (name, cam) in enumerate(cams.items()):
        overlay = overlay_cam_on_image(image, cam)
        axes[idx + 1].imshow(overlay)
        axes[idx + 1].set_title(name if titles is None else titles[idx])
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return result


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试Grad-CAM可视化模块")
    print("=" * 60)
    
    # 创建简单模型进行测试
    print("\n[1] 创建测试模型...")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU()
            )
            self.encoder2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x1 = self.encoder1(x)
            x2 = self.encoder2(x1)
            out = self.decoder(x2)
            return out
    
    model = SimpleModel()
    print(f"  模型创建成功")
    
    # 测试GradCAM
    print("\n[2] 测试GradCAM...")
    target_layers = ['encoder1', 'encoder2']
    gradcam = GradCAM(model, target_layers)
    
    # 创建测试输入
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    cams = gradcam.generate(x)
    
    print(f"  生成的CAM层: {list(cams.keys())}")
    for name, cam in cams.items():
        print(f"    {name}: {cam.shape}")
    
    # 清理
    gradcam.remove_hooks()
    
    # 测试GradCAM++
    print("\n[3] 测试GradCAM++...")
    gradcam_pp = GradCAMPlusPlus(model, target_layers)
    cams_pp = gradcam_pp.generate(x)
    
    print(f"  生成的CAM++层: {list(cams_pp.keys())}")
    
    # 清理
    gradcam_pp.remove_hooks()
    
    # 测试可视化函数
    print("\n[4] 测试可视化函数...")
    
    # 创建模拟图像和CAM
    mock_image = np.random.rand(64, 64, 3)
    mock_cam = np.random.rand(64, 64)
    
    # 测试叠加函数
    overlay = overlay_cam_on_image(mock_image, mock_cam)
    print(f"  叠加后图像形状: {overlay.shape}")
    print(f"  叠加后图像值域: [{overlay.min():.3f}, {overlay.max():.3f}]")
    
    # 测试多层可视化
    mock_cams = {
        'layer1': np.random.rand(64, 64),
        'layer2': np.random.rand(64, 64),
        'layer3': np.random.rand(64, 64)
    }
    
    result = visualize_multi_layer_cam(mock_image, mock_cams)
    print(f"  多层可视化结果形状: {result.shape}")
    
    print("\n" + "=" * 60)
    print("Grad-CAM可视化模块测试完成！")
    print("=" * 60)