"""
裂缝分割网络特征热力图可视化脚本 (Grad-CAM)
融合了 vm-unet 的 matplotlib 叠加渲染风格
专门适配 Crack500 (320*640*3) 三通道 Mask
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from models.crack_net import CrackSegmentationNetV2
from datasets.dataset import get_transforms

class SegmentationGradCAM:
    def __init__(self, model: nn.Module, target_layers: list):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(layer_name):
            def hook(module, input, output):
                self.activations[layer_name] = output.detach()
            return hook

        def backward_hook(layer_name):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_name] = grad_output[0].detach()
            return hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_full_backward_hook(backward_hook(name)))

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

    def generate_cams(self, input_tensor: torch.Tensor, target_mask: torch.Tensor = None):
        self.model.zero_grad()
        
        # 1. 前向传播
        output = self.model(input_tensor) # (1, 1, H, W)
        
        # 2. 反向传播目标：计算裂缝预测区域的分数
        if target_mask is not None:
            score = (output * target_mask).sum()
        else:
            crack_region = (output > 0.5).float()
            score = (output * crack_region).sum()
            if score == 0:
                score = output.sum()
                
        score.backward()
        
        # 3. 计算各层的 Grad-CAM
        cams = {}
        for name in self.target_layers:
            if name not in self.activations or name not in self.gradients:
                continue
                
            act = self.activations[name]  # (1, C, H', W')
            grad = self.gradients[name]   # (1, C, H', W')
            
            # 计算梯度权重
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            
            # 特征图加权求和并 ReLU 去除负值
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # 归一化热力图到 0 ~ 1
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-8)
            
            cams[name] = cam.squeeze().cpu().numpy()
            
        return cams, output.detach().squeeze().cpu().numpy()


def save_gradcam_imgs(img, msk, msk_pred, cams_dict, save_path, threshold=0.5):
    """
    使用 matplotlib 直接叠加显示热力图 (兼容任意维度的输入防报错)
    """
    # 处理原图 (反归一化)
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    
    # 安全处理真实的 mask，去除多余维度
    if msk is not None:
        if msk.ndim >= 3:
            msk = np.squeeze(msk)
        msk_bin = np.where(msk > 0.5, 1, 0)
    else:
        msk_bin = None

    # 处理预测概率图
    if msk_pred.ndim >= 3:
        msk_pred = np.squeeze(msk_pred)
    msk_pred_bin = np.where(msk_pred > threshold, 1, 0)

    # 布局计算：原图 + GT(如果存在) + Pred + 各个层的 CAM
    num_plots = 2 + len(cams_dict) + (1 if msk_bin is not None else 0)
    
    plt.figure(figsize=(10, 5 * num_plots))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)

    plot_idx = 1
    
    # 1. 原图
    plt.subplot(num_plots, 1, plot_idx)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    plot_idx += 1

    # 2. Ground Truth 掩码
    if msk_bin is not None:
        plt.subplot(num_plots, 1, plot_idx)
        plt.imshow(msk_bin, cmap='gray')
        plt.axis('off')
        plt.title('Ground Truth Mask')
        plot_idx += 1

    # 3. 预测掩码
    plt.subplot(num_plots, 1, plot_idx)
    plt.imshow(msk_pred_bin, cmap='gray')
    plt.axis('off')
    plt.title('Predicted Mask')
    plot_idx += 1

    # 4. 各层 Grad-CAM 热力图叠加 (直接用 jet cmap + alpha)
    for layer_name, cam in cams_dict.items():
        plt.subplot(num_plots, 1, plot_idx)
        plt.imshow(img)
        
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title(f'Attention Map Overlay ({layer_name})')
        plot_idx += 1

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="裂缝分割网络 Grad-CAM (适配 Crack500 三通道)")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--image_path', type=str, required=True, help='测试图像路径')
    parser.add_argument('--mask_path', type=str, default=None, help='(可选) Ground Truth 掩码路径')
    parser.add_argument('--base_channels', type=int, default=96)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--save_dir', type=str, default='./gradcam_outputs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("加载模型...")
    model = CrackSegmentationNetV2(
        in_channels=3, num_classes=1, 
        base_channels=args.base_channels, 
        input_size=args.input_size, 
        d_state=args.d_state,
        use_gbc=True
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device)
    model.eval()

    # 指定要观察的网络层名称
    target_layers = [
        'encoder1', 
        'encoder2', 
        'encoder3', 
        'encoder4', 
        'decoder.bottleneck',
    ]
    gradcam = SegmentationGradCAM(model, target_layers)

    print(f"处理图像: {args.image_path}")
    orig_img = cv2.imread(args.image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    transform = get_transforms(mode='val', target_size=args.input_size)
    
    orig_mask_bin = None
    target_mask_tensor = None
    
    # ========= 处理 Crack500 320*640*3 Mask 的核心逻辑 =========
    if args.mask_path and os.path.exists(args.mask_path):
        mask_raw = cv2.imread(args.mask_path, cv2.IMREAD_UNCHANGED)
        
        # 判断如果读出来是三通道 (例如 Crack500 的 320*640*3)
        if len(mask_raw.shape) == 3:
            # 安全降维：提取第0通道（对于黑白二值Mask，RGB三个通道的值完全一样）
            orig_mask = mask_raw[:, :, 0] 
        else:
            orig_mask = mask_raw
            
        # 标准化为 0 和 1
        _, orig_mask_bin = cv2.threshold(orig_mask, 127, 1, cv2.THRESH_BINARY)
        
        # 将原图与单通道的 Mask 同时送入数据增强
        augmented = transform(image=orig_img, mask=orig_mask_bin)
        input_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # 此时 augmented['mask'] 是 2D 张量，增加维度以匹配网络输出 (1, 1, H, W)
        target_mask_tensor = augmented['mask'].unsqueeze(0).unsqueeze(0).to(device).float()
    else:
        dummy_mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
        augmented = transform(image=orig_img, mask=dummy_mask)
        input_tensor = augmented['image'].unsqueeze(0).to(device)

    print("生成并绘制热力图...")
    cams_dict, pred_prob = gradcam.generate_cams(input_tensor, target_mask=target_mask_tensor)
    gradcam.remove_hooks()

    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.save_dir, f'{img_name}_gradcam.png')
    
    # 提取转换后的 Mask 传递给画图函数
    mask_to_plot = augmented['mask'].numpy() if orig_mask_bin is not None else None
    
    save_gradcam_imgs(
        img=input_tensor, 
        msk=mask_to_plot, 
        msk_pred=pred_prob, 
        cams_dict=cams_dict, 
        save_path=save_path
    )
    
    print(f"可视化结果已保存至: {save_path}")

if __name__ == '__main__':
    main()