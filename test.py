"""
测试脚本 - 裂缝分割网络模型评估

用于在测试集上评估模型性能，支持：
- 加载训练好的模型检查点
- 计算各项评估指标（F1, IoU, mIoU, Precision, Recall, FPS）
- 保存预测掩码
- 生成可视化结果
- 输出详细测试报告

使用方法:
    python test.py \
        --checkpoint ./checkpoints/best_model.pth \
        --dataset crack500 \
        --data_root /mnt/d/dev/data/crack500 \
        --save_predictions \
        --visualize \
        --num_vis 20 \
        --save_dir ./test_results
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.crack_net import CrackSegmentationNet
from datasets.dataset import get_dataloader
from utils.metrics import MetricCalculator, AverageMeter
from utils.losses import BCEDiceLoss, CombinedLoss, DiceLoss, FocalLoss


class Tester:
    """
    裂缝分割网络测试器
    
    参数：
        model: 模型
        device: 计算设备
        criterion: 损失函数
        config: 配置字典
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        criterion: nn.Module = None,
        config: Dict = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # 默认损失函数
        self.criterion = criterion or BCEDiceLoss()
        
        # 配置
        self.config = config or {}
        self.use_amp = self.config.get('use_amp', False)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # CuDNN优化
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # TF32加速（Ampere及以上GPU）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @torch.inference_mode()
    def bak_test(
        self,
        test_loader: DataLoader,
        save_predictions: bool = False,
        save_dir: str = None,
        visualize: bool = False,
        num_vis: int = None
    ) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        参数：
            test_loader: 测试数据加载器
            save_predictions: 是否保存预测结果
            save_dir: 预测结果保存目录
            visualize: 是否保存可视化结果
            num_vis: 限制可视化数量（None表示全部）
            
        返回：
            测试指标字典
        """
        self.model.eval()
        test_loss_meter = AverageMeter()
        metric_calc = MetricCalculator()
        
        # 创建保存目录
        if save_predictions and save_dir:
            pred_dir = os.path.join(save_dir, 'predictions')
            vis_dir = os.path.join(save_dir, 'visualizations')
            os.makedirs(pred_dir, exist_ok=True)
            if visualize:
                os.makedirs(vis_dir, exist_ok=True)
        
        all_results = []
        vis_count = 0
        
        print(f"\n开始测试...")
        print(f"测试样本数: {len(test_loader.dataset)}")
        print(f"批次大小: {test_loader.batch_size}")
        print("=" * 60)
        
        start_time = time.time()
        
        for batch_idx, (images, masks, meta_list) in enumerate(test_loader):
            images = images.to(self.device).float()
            masks = masks.to(self.device)
            batch_size = images.size(0)
            
            # 前向传播（自动混合精度）
            if self.use_amp and self.device.type == 'cuda':
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # 更新指标
            test_loss_meter.update(loss.item(), batch_size)
            metric_calc.update(outputs, masks)
            
            # 保存预测结果
            if save_predictions and save_dir:
                predictions = torch.sigmoid(outputs) > 0.5
                # predictions = outputs.cpu().float().numpy()


                
                # 处理元数据
                processed_meta_list = self._process_meta_list(meta_list, batch_size)
                
                for i in range(batch_size):
                    # 获取元数据
                    meta = processed_meta_list[i] if i < len(processed_meta_list) else {}
                    img_path = meta.get('image_path', f'img_{batch_idx * batch_size + i}')
                    img_name = os.path.splitext(os.path.basename(str(img_path)))[0]
                    
                    # 保存预测mask
                    pred_mask = predictions[i].squeeze().cpu().numpy().astype(np.uint8) * 255
                    # pred_mask = predictions[i].squeeze()
                    # _,pred_mask = cv2.threshold(pred_mask,127,255,cv2.THRESH_BINARY)
                    # _,pred_mask = cv2.threshold(pred_mask,127,1,cv2.THRESH_BINARY)

                    pred_path = os.path.join(pred_dir, f'{img_name}_pred.png')
                    cv2.imwrite(pred_path, pred_mask)
                    
                    # 保存可视化结果
                    if visualize and (num_vis is None or vis_count < num_vis):
                        self._save_visualization(
                            images[i], masks[i], outputs[i],
                            os.path.join(vis_dir, f'{img_name}_vis.png')
                        )
                        vis_count += 1
                    
                    # 记录结果
                    all_results.append({
                        'image_path': str(img_path),
                        'pred_path': pred_path,
                        'image_name': img_name
                    })
            
            # 显示进度
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(test_loader) - 1:
                current_metrics = metric_calc.compute()
                print(f"  [{batch_idx + 1}/{len(test_loader)}] "
                      f"Loss: {test_loss_meter.avg:.4f}, "
                      f"F1: {current_metrics['f1']:.4f}")
        
        total_time = time.time() - start_time
        
        # 计算最终指标
        metrics = metric_calc.compute()
        miou = metric_calc.compute_mIoU()
        
        # 计算FPS
        fps = len(test_loader.dataset) / total_time if total_time > 0 else 0.0
        
        test_results = {
            'loss': test_loss_meter.avg,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'iou': metrics['iou'],
            'miou': miou,
            'accuracy': metrics['accuracy'],
            'specificity': metrics['specificity'],
            'fps': fps,
            'total_time': total_time,
            'num_samples': len(test_loader.dataset)
        }
        
        # 保存测试结果
        if save_dir:
            test_results['checkpoint_path'] = self.config.get('checkpoint_path', 'N/A')
            self._save_test_results(test_results, all_results, save_dir)
        
        return test_results
    
    @torch.inference_mode()   
    def test(
        self,
        test_loader: DataLoader,
        save_predictions: bool = False,
        save_dir: str = None,
        visualize: bool = False,
        num_vis: int = None,
        benchmark: bool = False  # [新增] 是否仅测试速度
    ) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        参数：
            benchmark: 如果为 True，仅测量模型前向传播速度，忽略指标计算和 IO 耗时
        """
        self.model.eval()
        test_loss_meter = AverageMeter()
        metric_calc = MetricCalculator()
        
        # 创建保存目录
        if save_predictions and save_dir and not benchmark:
            pred_dir = os.path.join(save_dir, 'predictions')
            vis_dir = os.path.join(save_dir, 'visualizations')
            os.makedirs(pred_dir, exist_ok=True)
            if visualize:
                os.makedirs(vis_dir, exist_ok=True)
        
        all_results = []
        vis_count = 0
        
        print(f"\n开始测试... ")
        print(f"测试样本数：{len(test_loader.dataset)} ")
        print(f"批次大小：{test_loader.batch_size} ")
        print(f"模式：{'Benchmark (纯推理速度)' if benchmark else 'Full Evaluation (完整评估)'}")
        print("=" * 60)
        
        # --- [新增] Warmup 预热 (消除 GPU 初始化开销) ---
        if self.device.type == 'cuda' and benchmark:
            print("正在进行 GPU 预热 (Warmup)...")
            for _ in range(10):
                dummy_input = torch.randn(1, 3, self.config.get('input_size', 512), self.config.get('input_size', 512)).to(self.device)
                _ = self.model(dummy_input)
            torch.cuda.synchronize()
        # ---------------------------------------------
        
        total_inference_time = 0.0  # [修改] 仅累计推理耗时
        start_time = time.time()    # 用于计算总流程时间（非 benchmark 模式）
        
        for batch_idx, (images, masks, meta_list) in enumerate(test_loader):
            images = images.to(self.device).float()
            masks = masks.to(self.device)
            batch_size = images.size(0)
            
            # --- [核心修改] 精确计时仅包含 Forward Pass ---
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # 等待 GPU 完成之前任务
            
            t_start = time.time()
            
            # 前向传播（自动混合精度）
            if self.use_amp and self.device.type == 'cuda':
                with autocast():
                    outputs = self.model(images)
                    # Benchmark 模式下不需要计算 loss
                    if not benchmark:
                        loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                if not benchmark:
                    loss = self.criterion(outputs, masks)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # 确保 GPU 完成当前任务
            
            t_end = time.time()
            total_inference_time += (t_end - t_start)
            # ---------------------------------------------
            
            # 以下操作不计入推理 FPS (除非是非 benchmark 模式的总耗时)
            if not benchmark:
                test_loss_meter.update(loss.item(), batch_size)
                metric_calc.update(outputs, masks)
                
                # 保存预测结果
                if save_predictions and save_dir:
                    predictions = torch.sigmoid(outputs) > 0.5
                    processed_meta_list = self._process_meta_list(meta_list, batch_size)
                    
                    for i in range(batch_size):
                        meta = processed_meta_list[i] if i < len(processed_meta_list) else {}
                        img_path = meta.get('image_path', f'img_{batch_idx * batch_size + i}')
                        img_name = os.path.splitext(os.path.basename(str(img_path)))[0]
                        
                        pred_mask = predictions[i].squeeze().cpu().numpy().astype(np.uint8) * 255
                        pred_path = os.path.join(pred_dir, f'{img_name}_pred.png')
                        cv2.imwrite(pred_path, pred_mask)
                        
                        if visualize and (num_vis is None or vis_count < num_vis):
                            self._save_visualization(
                                images[i], masks[i], outputs[i],
                                os.path.join(vis_dir, f'{img_name}_vis.png')
                            )
                            vis_count += 1
                        
                        all_results.append({
                            'image_path': str(img_path),
                            'pred_path': pred_path,
                            'image_name': img_name
                        })
            
            # 显示进度
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(test_loader) - 1:
                if not benchmark:
                    current_metrics = metric_calc.compute()
                    print(f"  [{batch_idx + 1}/{len(test_loader)}]  "
                          f"Loss: {test_loss_meter.avg:.4f},  "
                          f"F1: {current_metrics['f1']:.4f} ")
                else:
                    current_fps = (batch_idx + 1) * batch_size / total_inference_time if total_inference_time > 0 else 0
                    print(f"  [{batch_idx + 1}/{len(test_loader)}]  当前推理 FPS: {current_fps:.2f} ")
        
        # 计算 FPS
        # Benchmark 模式：使用纯推理时间
        # 普通模式：使用总流逝时间（包含 IO 和指标计算）
        if benchmark:
            fps = len(test_loader.dataset) / total_inference_time if total_inference_time > 0 else 0.0
            total_time = total_inference_time
        else:
            total_time = time.time() - start_time
            fps = len(test_loader.dataset) / total_time if total_time > 0 else 0.0
        
        # 计算最终指标
        if not benchmark:
            metrics = metric_calc.compute()
            miou = metric_calc.compute_mIoU()
            test_results = {
                'loss': test_loss_meter.avg,
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'iou': metrics['iou'],
                'miou': miou,
                'accuracy': metrics['accuracy'],
                'specificity': metrics['specificity'],
                'fps': fps,
                'total_time': total_time,
                'num_samples': len(test_loader.dataset)
            }
            if save_dir:
                test_results['checkpoint_path'] = self.config.get('checkpoint_path', 'N/A')
                self._save_test_results(test_results, all_results, save_dir)
        else:
            # Benchmark 模式只返回速度相关结果
            test_results = {
                'fps': fps,
                'total_inference_time': total_inference_time,
                'num_samples': len(test_loader.dataset),
                'batch_size': test_loader.batch_size,
                'mode': 'benchmark'
            }
            print(f"\n[Benchmark 结果] 平均 FPS: {fps:.2f} | 总推理耗时：{total_inference_time:.4f}s")
        
        return test_results


    def _process_meta_list(self, meta_list, batch_size: int) -> List[Dict]:
        """处理元数据列表"""
        processed_meta_list = []
        
        if isinstance(meta_list, dict):
            # 如果是batch后的dict，每个key对应一个列表
            meta_keys = list(meta_list.keys())
            for i in range(batch_size):
                item = {}
                for k in meta_keys:
                    val = meta_list[k]
                    if isinstance(val, list) and i < len(val):
                        item[k] = val[i]
                    elif isinstance(val, torch.Tensor) and i < len(val):
                        item[k] = val[i].item() if val[i].numel() == 1 else val[i]
                    else:
                        item[k] = None
                processed_meta_list.append(item)
        else:
            processed_meta_list = meta_list
        
        return processed_meta_list
    
    def _save_visualization(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        output: torch.Tensor,
        save_path: str
    ):
        """
        保存可视化结果
        
        创建四格图：原图、真值、预测概率图、叠加图
        """
        # 处理张量
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        mask = mask.squeeze().cpu().numpy()
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # 创建叠加图（红色表示真值）
        overlay = img.copy()
        mask_binary = (mask > 0.5).astype(np.float32)
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - mask_binary * 0.5) + mask_binary * 0.5
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - mask_binary * 0.5)
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - mask_binary * 0.5)
        
        # 创建图形
        _, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Prediction (Prob)')
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (Red=GT)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_test_results(
        self,
        metrics: Dict[str, float],
        results: List[Dict],
        save_dir: str
    ):
        """保存测试结果到文件"""
        
        # 保存指标（JSON格式）
        metrics_path = os.path.join(save_dir, 'test_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存结果列表
        results_path = os.path.join(save_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存详细报告
        report_path = os.path.join(save_dir, 'test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(" " * 20 + "裂缝分割网络 - 测试报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("【模型配置】\n")
            f.write(f"  检查点路径: {self.config.get('checkpoint_path', 'N/A')}\n")
            f.write(f"  设备: {self.device}\n")
            f.write(f"  混合精度: {self.use_amp}\n\n")
            
            f.write("【数据集信息】\n")
            f.write(f"  数据集: {self.config.get('dataset_name', 'N/A')}\n")
            f.write(f"  数据路径: {self.config.get('data_root', 'N/A')}\n")
            f.write(f"  测试样本数: {metrics['num_samples']}\n")
            f.write(f"  输入尺寸: {self.config.get('input_size', 'N/A')}\n\n")
            
            f.write("【性能指标】\n")
            f.write(f"  总推理时间:     {metrics['total_time']:.2f} s\n")
            f.write(f"  推理速度 (FPS): {metrics['fps']:.2f}\n\n")
            
            f.write("【分割指标】\n")
            f.write(f"  损失 (Loss):        {metrics['loss']:.4f}\n")
            f.write(f"  F1 分数:            {metrics['f1']:.4f}\n")
            f.write(f"  IoU:                {metrics['iou']:.4f}\n")
            f.write(f"  mIoU:               {metrics['miou']:.4f}\n")
            f.write(f"  精确率 (Precision): {metrics['precision']:.4f}\n")
            f.write(f"  召回率 (Recall):    {metrics['recall']:.4f}\n")
            f.write(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}\n")
            f.write(f"  特异性 (Specificity): {metrics['specificity']:.4f}\n\n")
            
            f.write("【文件输出】\n")
            f.write(f"  预测结果数量: {len(results)} 张\n")
            f.write(f"  预测目录: {os.path.join(save_dir, 'predictions')}\n")
            if os.path.exists(os.path.join(save_dir, 'visualizations')):
                vis_count = len([f for f in os.listdir(os.path.join(save_dir, 'visualizations')) 
                                if f.endswith(('.png', '.jpg'))])
                f.write(f"  可视化目录: {os.path.join(save_dir, 'visualizations')} ({vis_count} 张)\n")
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"\n测试结果已保存到: {save_dir}")
        print(f"  - 指标文件: {metrics_path}")
        print(f"  - 报告文件: {report_path}")
        if results:
            print(f"  - 预测结果: {len(results)} 张")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    """
    加载模型检查点
    
    参数：
        model: 模型
        checkpoint_path: 检查点路径
        device: 设备
        
    返回：
        检查点信息字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 成功加载模型权重")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ 成功加载模型（完整模型）")
    
    # 打印检查点信息
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  验证指标: {checkpoint['metrics']}")
    
    return checkpoint


def get_loss_function(loss_type: str = 'bce_dice', **kwargs):
    """
    获取损失函数
    
    参数：
        loss_type: 损失函数类型
        **kwargs: 额外参数
        
    返回：
        损失函数实例
    """
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 0.25),
            gamma=kwargs.get('focal_gamma', 2.0)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            dice_weight=kwargs.get('dice_weight', 1.0),
            focal_weight=kwargs.get('focal_weight', 1.0)
        )
    elif loss_type == 'tversky':
        from utils.losses import TverskyLoss
        return TverskyLoss()
    else:  # bce_dice
        return BCEDiceLoss(
            bce_weight=kwargs.get('bce_weight', 0.5),
            dice_weight=kwargs.get('dice_weight', 0.5)
        )


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(
        description='裂缝分割网络测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本测试
  python test.py --checkpoint ./checkpoints/best_model.pth --dataset crack500
  
  # 保存预测结果和可视化
  python test.py --checkpoint ./checkpoints/best_model.pth \\
      --dataset crack500 --data_root /path/to/data \\
      --save_predictions --visualize --save_dir ./test_results
  
  # 限制可视化数量
  python test.py --checkpoint ./checkpoints/best_model.pth \\
      --dataset cfd --visualize --num_vis 10
        """
    )
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径（必需）')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='crack500',
                        choices=['crack500', 'cfd', 'sun520'],
                        help='数据集名称 (默认: crack500)')
    parser.add_argument('--data_root', type=str, default='/mnt/d/dev/data/crack500',
                        help='数据根目录（默认根据数据集自动推断）')
    parser.add_argument('--input_size', type=int, default=512,
                        help='输入图像尺寸 (默认: 512)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小 (默认: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数 (默认: 4)')
    
    # 模型参数
    parser.add_argument('--base_channels', type=int, default=32,
                        help='基础通道数 (默认: 32)')
    parser.add_argument('--d_state', type=int, default=16,
                        help='Mamba状态维度 (默认: 16)')
    parser.add_argument('--use_gbc', action='store_true',
                        help='使用GBC模块')
    parser.add_argument('--use_aspp', action='store_true',
                        help='使用ASPP模块')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='bce_dice',
                        choices=['bce_dice', 'dice', 'focal', 'combined', 'tversky'],
                        help='损失函数类型 (默认: bce_dice)')
    
    # 输出参数
    parser.add_argument('--save_predictions', action='store_true',
                        help='保存预测掩码')
    parser.add_argument('--visualize', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--num_vis', type=int, default=None,
                        help='限制可视化数量（默认不限制）')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='测试结果保存目录 (默认: ./test_results)')
    
    # 其他参数
    parser.add_argument('--use_amp', action='store_true',
                        help='使用自动混合精度（推荐用于推理加速）')
    parser.add_argument('--device', type=str, default=None,
                        help='指定设备 (cuda/cpu，默认自动选择)')
    # [新增] 性能测试模式
    parser.add_argument('--benchmark', action='store_true',
                        help='仅评估推理速度（不包含指标计算和 IO，用于论文报告）')

    
    args = parser.parse_args()
    return args


def main():
    """主测试函数"""
    args = get_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print(" " * 20 + "裂缝分割网络 - 测试")
    print("=" * 70)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    # 推断数据路径
    if args.data_root is None:
        dataset_paths = {
            'crack500': '/mnt/d/dev/data/crack500',
            'cfd': './data/cfd',
            'sun520': './data/sun520'
        }
        args.data_root = dataset_paths.get(args.dataset, f'./data/{args.dataset}')
        print(f"使用默认数据路径: {args.data_root}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"测试结果保存路径: {args.save_dir}")
    
    # 创建模型
    print("\n【1/4】创建模型...")
    model = CrackSegmentationNet(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels,
        input_size=args.input_size,
        d_state=args.d_state,
        use_gbc=args.use_gbc,
        use_aspp=args.use_aspp
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 加载检查点
    print(f"\n【2/4】加载检查点...")
    print(f"  路径: {args.checkpoint}")
    try:
        checkpoint = load_checkpoint(model, args.checkpoint, device)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return
    
    # 创建损失函数
    criterion = get_loss_function(args.loss_type)
    print(f"\n【3/4】配置完成")
    print(f"  损失函数: {args.loss_type}")
    print(f"  混合精度: {args.use_amp}")
    
    # 创建配置
    config = {
        'checkpoint_path': args.checkpoint,
        'dataset_name': args.dataset,
        'data_root': args.data_root,
        'input_size': args.input_size,
        'use_amp': args.use_amp
    }
    
    # 创建测试器
    tester = Tester(
        model=model,
        device=device,
        criterion=criterion,
        config=config
    )
    
    # 加载测试数据
    print(f"\n【4/4】加载测试数据...")
    try:
        test_loader = get_dataloader(
            dataset_name=args.dataset,
            root_dir=args.data_root,
            split='test',
            batch_size=args.batch_size,
            target_size=args.input_size,
            num_workers=args.num_workers
        )
        print(f"  测试集大小: {len(test_loader.dataset)} 张")
    except Exception as e:
        print(f"  警告: 加载测试集失败 ({e})")
        print("  尝试使用验证集...")
        try:
            val_split = 'validation' if args.dataset == 'crack500' else 'val'
            test_loader = get_dataloader(
                dataset_name=args.dataset,
                root_dir=args.data_root,
                split=val_split,
                batch_size=args.batch_size,
                target_size=args.input_size,
                num_workers=args.num_workers
            )
            print(f"  验证集大小: {len(test_loader.dataset)} 张")
        except Exception as e2:
            print(f"错误: 无法加载数据 ({e2})")
            return
    
    # 运行测试
    print("\n" + "=" * 70)
    print("开始测试")
    print("=" * 70)
    
    test_results = tester.test(
        test_loader=test_loader,
        save_predictions=args.save_predictions,
        save_dir=args.save_dir,
        visualize=args.visualize,
        num_vis=args.num_vis,
        benchmark=args.benchmark  # [新增] 传递 benchmark 参数
    )
    
    # 打印最终结果
    print("\n" + "=" * 70)
    print(" " * 25 + "测试结果")
    print("=" * 70)
    print(f"  损失 (Loss):        {test_results['loss']:.4f}")
    print(f"  F1 分数:            {test_results['f1']:.4f}")
    print(f"  IoU:                {test_results['iou']:.4f}")
    print(f"  mIoU:               {test_results['miou']:.4f}")
    print(f"  精确率 (Precision): {test_results['precision']:.4f}")
    print(f"  召回率 (Recall):    {test_results['recall']:.4f}")
    print(f"  准确率 (Accuracy):  {test_results['accuracy']:.4f}")
    print(f"  特异性 (Specificity): {test_results['specificity']:.4f}")
    print(f"  推理速度 (FPS):     {test_results['fps']:.2f}")
    print(f"  总推理时间:         {test_results['total_time']:.2f} s")
    print("=" * 70)
    
    print(f"\n✓ 测试完成！结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
