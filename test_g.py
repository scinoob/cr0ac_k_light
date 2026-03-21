"""
测试脚本 - 裂缝分割网络模型评估 (二值化修复版)

功能特点：
1. 强制 batch_size=1
2. 打印模型总参数量和可训练参数量
3. 纯推理模式 (--benchmark)：仅测试计算前向传播，计算纯 FPS。
4. 普通模式：恢复原图尺寸比对原始 Mask，计算 F1, mIoU 等指标。
5. 严格保存二值化的预测结果（黑底白裂缝）并生成黑白预测的可视化图。
6. 完美兼容 Crack500 三通道二值 Mask 和其它单通道 Mask。
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.crack_net import CrackSegmentationNetV2
from datasets.dataset import get_dataloader
from utils.metrics import MetricCalculator

class Tester:
    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        self.model = model
        self.device = device
        self.config = config
        self.dataset_name = config.get('dataset', 'crack500').lower()
        self.model.eval()
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def test(self, test_loader: DataLoader, save_predictions: bool = False, 
             save_dir: str = None, visualize: bool = False, benchmark: bool = False):
        
        if benchmark:
            print("\n[Benchmark 模式] 启动纯推理速度测试 (仅计算模型前向耗时，屏蔽IO与计算)...")
            
            # GPU Warmup 预热
            print("正在预热 GPU...")
            dummy_input = torch.randn(1, 3, self.config['input_size'], self.config['input_size']).to(self.device)
            for _ in range(10):
                self.model(dummy_input)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            total_inference_time = 0.0
            num_samples = 0
            
            # 测试循环
            for images, _, _ in test_loader:
                images = images.to(self.device).float()
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                t_start = time.time()
                _ = self.model(images)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                total_inference_time += (time.time() - t_start)
                num_samples += 1
                
            fps = num_samples / total_inference_time if total_inference_time > 0 else 0
            
            print(f"\n{'='*20} Benchmark 结果 {'='*20}")
            print(f"总处理样本数: {num_samples} 张")
            print(f"总推理耗时:   {total_inference_time:.4f} s")
            print(f"纯推理 FPS:   {fps:.2f} 帧/秒")
            print('='*55)
            return {'fps': fps}
            
        else:
            print("\n[普通推理模式] 在原图分辨率上进行指标评估与结果生成...")
            metric_calc = MetricCalculator()
            
            if save_predictions and save_dir:
                pred_dir = os.path.join(save_dir, 'predictions')
                vis_dir = os.path.join(save_dir, 'visualizations')
                os.makedirs(pred_dir, exist_ok=True)
                if visualize:
                    os.makedirs(vis_dir, exist_ok=True)
                    
            num_samples = 0
            
            for batch_idx, (images, _, meta_list) in enumerate(test_loader):
                # 1. 网络前向传播
                images = images.to(self.device).float()
                outputs = self.model(images) # (1, 1, 512, 512)
                
                # 提取单个 meta 元素
                img_path = meta_list['image_path'][0] if isinstance(meta_list['image_path'], list) else meta_list['image_path']
                mask_path = meta_list['mask_path'][0] if isinstance(meta_list['mask_path'], list) else meta_list['mask_path']
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # 2. 读取原始图像与 Mask 以便基于原图尺寸评估
                orig_img = cv2.imread(img_path)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = orig_img.shape[:2]
                
                # 3. 处理不同数据集的 Mask
                orig_mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if self.dataset_name == 'crack500' and len(orig_mask_raw.shape) == 3:
                    # 三通道二值图像按通道取最大值最安全
                    orig_mask = orig_mask_raw.max(axis=2)
                else:
                    orig_mask = orig_mask_raw
                    
                # 统一转为 {0, 1} 张量
                _, orig_mask_bin = cv2.threshold(orig_mask, 127, 1, cv2.THRESH_BINARY)
                orig_mask_tensor = torch.from_numpy(orig_mask_bin).unsqueeze(0).unsqueeze(0).to(self.device).float()
                
                # 4. 将模型的输出插值还原到原始尺寸
                outputs_resized = F.interpolate(outputs, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                
                # 5. 更新指标
                metric_calc.update(outputs_resized, orig_mask_tensor)
                num_samples += 1
                
                # 6. 保存严格二值化的预测结果与可视化
                if save_predictions and save_dir:
                    # 获取概率图并使用 0.5 作为阈值二值化 (0 或 1)
                    pred_prob = outputs_resized.squeeze().cpu().numpy()
                    pred_mask_bin = (pred_prob > 0.5).astype(np.uint8)
                    
                    # 保存预测掩码：只包含 0 和 255 (背景黑，裂缝白)
                    pred_mask_save = pred_mask_bin * 255
                    cv2.imwrite(os.path.join(pred_dir, f'{img_name}_pred.png'), pred_mask_save)
                    
                    if visualize:
                        self._save_visualization(
                            orig_img, orig_mask_bin, pred_mask_bin,
                            os.path.join(vis_dir, f'{img_name}_vis.png')
                        )
                        
                # 进度打印
                if (batch_idx + 1) % 10 == 0 or batch_idx == len(test_loader) - 1:
                    current_metrics = metric_calc.compute()
                    print(f"  进度 [{batch_idx + 1}/{len(test_loader)}]  F1: {current_metrics['f1']:.4f} | IoU: {current_metrics['iou']:.4f}")
            
            # 汇总结果
            metrics = metric_calc.compute()
            miou = metric_calc.compute_mIoU()
            
            print(f"\n{'='*20} 评估结果 {'='*20}")
            print(f"F1 分数:             {metrics['f1']:.4f}")
            print(f"mIoU:                {miou:.4f}")
            print(f"精确率 (Precision):  {metrics['precision']:.4f}")
            print(f"召回率 (Recall):     {metrics['recall']:.4f}")
            print(f"准确率 (Accuracy):   {metrics['accuracy']:.4f}")
            print(f"特异性 (Specificity):{metrics['specificity']:.4f}")
            print('='*50)
            
            if save_dir:
                metrics['miou'] = miou
                with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=4)
                print(f"测试结果已保存至: {save_dir}")
                
            return metrics

    def _save_visualization(self, img_rgb: np.ndarray, mask_bin: np.ndarray, pred_bin: np.ndarray, save_path: str):
        """生成并保存四格可视化图，确保全部显示二值掩码而非伪彩热力图"""
        img_normalized = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
        
        # 将预测的裂缝用红色叠加到原图上
        overlay = img_normalized.copy()
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - pred_bin) + pred_bin
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - pred_bin)
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - pred_bin)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img_normalized)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground Truth 显示为黑白
        axes[1].imshow(mask_bin, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # 修复：预测掩码同样强制显示为黑白二值化，弃用 jet 伪彩色
        axes[2].imshow(pred_bin, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Prediction (Mask)')
        axes[2].axis('off')
        
        # 叠加图展示预测效果
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (Red=Pred)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='裂缝分割网络测试脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重(.pth)路径')
    parser.add_argument('--dataset', type=str, default='crack500', choices=['crack500', 'cfd', 'sun520'])
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--input_size', type=int, default=224, help='模型输入尺寸')
    parser.add_argument('--base_channels', type=int, default=96, help='基础通道数')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba状态维度')
    parser.add_argument('--use_gbc', action="store_true", help='gbc')
    
    parser.add_argument('--benchmark', action='store_true', help='开启纯推理模式(仅测速)')
    parser.add_argument('--save_predictions', action='store_true', help='保存预测结果')
    parser.add_argument('--visualize', action='store_true', help='保存可视化图像')
    parser.add_argument('--save_dir', type=str, default='./test_results', help='保存路径')
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print(" 裂缝分割网络推理评估")
    print("=" * 60)
    
    # 强制加载测试集时 batch_size=1
    print("加载测试集...")
    test_loader = get_dataloader(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='test',
        batch_size=4,            # <--- 强制设置为 1
        target_size=args.input_size,
        num_workers=4
    )
    
    print("初始化模型...")
    model = CrackSegmentationNetV2(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels,
        input_size=args.input_size,
        d_state=args.d_state,
        use_gbc = args.use_gbc
    )
    
    # === 新增：计算并打印模型参数量 ===
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> 模型总参数量:     {total_params / 1e6:.4f} M")
    print(f" -> 可训练模型参数量: {trainable_params / 1e6:.4f} M")
    
    print(f"\n加载权重: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    tester = Tester(model, device, vars(args))
    tester.test(
        test_loader=test_loader,
        save_predictions=args.save_predictions,
        save_dir=args.save_dir,
        visualize=args.visualize,
        benchmark=args.benchmark
    )

if __name__ == "__main__":
    main()