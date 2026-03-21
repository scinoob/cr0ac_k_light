"""
训练器和训练脚本实现
包含完整的训练流程、验证、保存检查点等
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
import cv2
import time
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import MetricCalculator, AverageMeter, RigorousCalculator
from utils.losses import CombinedLoss, DiceLoss, FocalLoss, BCEDiceLoss, TverskyLoss
from models.crack_net import CrackSegmentationNet, CrackSegmentationNetV2
from datasets.dataset import get_dataloader


class Trainer:
    """
    裂缝分割网络训练器
    
    参数：
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        config: 配置字典
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module = None,
            optimizer: optim.Optimizer = None,
            scheduler: optim.lr_scheduler._LRScheduler = None,
            device: torch.device = None,
            config: Dict = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 默认损失函数
        self.criterion = criterion or BCEDiceLoss()

        # 默认优化器
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4) if config else 1e-4,
            weight_decay=config.get('weight_decay', 1e-4) if config else 1e-4
        )

        # 默认调度器
        self.scheduler = scheduler

        # 设备
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # 配置
        self.config = config or {}
        self.epochs = self.config.get('epochs', 100)
        self.save_dir = self.config.get('save_dir', './checkpoints')
        self.log_interval = self.config.get('log_interval', 10)
        self.use_amp = self.config.get('use_amp', False)
        self.grad_clip = self.config.get('grad_clip', 1.0)
        # self.grad_clip = None

        # 混合精度
        self.scaler = GradScaler() if self.use_amp else None

        # 指标
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        # self.metric_calc = MetricCalculator()
        self.metric_calc = RigorousCalculator()

        # 最佳模型
        self.best_f1 = 0.0
        self.best_epoch = 0

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)

        # TensorBoard日志目录
        self.log_dir = self.config.get('log_dir', os.path.join(self.save_dir, 'log'))
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'val_miou': [],
            'lr': []
        }

    @staticmethod
    def get_gradient_norm(model) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        参数：
            epoch: 当前epoch
            
        返回：
            训练指标字典
        """
        self.model.train()
        self.train_loss_meter.reset()
        self.metric_calc.reset()

        # 图片可视化计数器
        sample_counter = 0
        start_time = time.time()

        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device).float()
            masks = masks.to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # 反向传播
                self.scaler.scale(loss).backward()

                total_norm = self.get_gradient_norm(self.model)
                self.writer.add_scalar('train/Gradient', total_norm, epoch)
                # print(f"Gradient norm: {total_norm:.4f}", end="\r")

                # 梯度裁剪
                if self.grad_clip is not None and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    # 打印梯度裁剪后的梯度范数
                    total_norm = self.get_gradient_norm(self.model)
                    self.writer.add_scalar('train/Gradient_norm_after_clip', total_norm, epoch)
                    # print(f"Gradient norm after clip: {total_norm:.4f}", end="\r")

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()

                total_norm = self.get_gradient_norm(self.model)
                self.writer.add_scalar('train/Gradient', total_norm, epoch)
                # print(f"Gradient norm: {total_norm:.4f}", end="\r")

                # 梯度裁剪
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    # 打印梯度裁剪后的梯度范数
                    total_norm = self.get_gradient_norm(self.model)
                    self.writer.add_scalar('train/Gradient_norm_after_clip', total_norm, epoch)
                    # print(f"Gradient norm after clip: {total_norm:.4f}", end="\r")

                self.optimizer.step()

            # 更新指标
            self.train_loss_meter.update(loss.item(), batch_size)

            # 每处理20张图片，记录第2张图片的可视化
            prev_counter = sample_counter
            sample_counter += batch_size

            if prev_counter // 20 < sample_counter // 20 and batch_size >= 2:
                # 计算第2张图片在当前batch中的索引
                target_idx = 1  # 第2张图片

                with torch.no_grad():
                    # 获取图片、预测值和真值
                    img = images[target_idx]  # (C, H, W)
                    pred = outputs[target_idx]  # (1, H, W) or (C, H, W)
                    mask = masks[target_idx]  # (1, H, W)

                    # 处理预测值：转换为二值可视化格式
                    # 【修复代码】将预测值转换为概率，再映射到 0-255
                    pred = torch.sigmoid(pred).cpu().numpy() * 255
                    _, pred = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
                    _, pred = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)

                    # 创建叠加图：原图 + mask边界
                    img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    overlay = img_normalized.clone()
                    # 在mask区域添加红色高亮
                    mask_binary = (mask.unsqueeze(0) > 0.5).float()

                    # 记录到TensorBoard
                    self.writer.add_image('Train/Image', img_normalized, epoch)
                    self.writer.add_image('Train/Prediction', pred, epoch)
                    self.writer.add_image("Train/GroundTruth", mask_binary, epoch)

            # 日志
            if (batch_idx + 1) % self.log_interval == 0:
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} ({self.train_loss_meter.avg:.4f})")

        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

        epoch_time = time.time() - start_time

        return {
            'loss': self.train_loss_meter.avg,
            'time': epoch_time
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        验证
        
        参数：
            epoch: 当前epoch
            
        返回：
            验证指标字典
        """
        self.model.eval()
        self.val_loss_meter.reset()
        self.metric_calc.reset()

        for images, masks, _ in self.val_loader:
            images = images.to(self.device).float()
            masks = masks.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # 更新指标
            self.val_loss_meter.update(loss.item(), images.size(0))
            # 【修复代码】计算指标前，必须将 Logits 转换为概率！
            self.metric_calc.update(torch.sigmoid(outputs), masks)

        # 计算指标
        metrics = self.metric_calc.compute()
        miou = self.metric_calc.compute_mIoU()

        result =  {
            'loss': self.val_loss_meter.avg,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'iou': metrics['iou'],
            'miou': miou
        }

        # 【关键修复】将丢掉的新指标强行补回来！
        result['ois_f1'] = metrics.get('ois_f1', 0.0)
        result['best_threshold'] = metrics.get('best_threshold', 0.5)

        return result

    @torch.no_grad()
    def test(
            self,
            test_loader: DataLoader,
            save_predictions: bool = False,
            save_dir: str = None,
            visualize: bool = False
    ) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        参数：
            test_loader: 测试数据加载器
            save_predictions: 是否保存预测结果
            save_dir: 预测结果保存目录
            visualize: 是否保存可视化结果
            
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

        print(f"\n开始测试...")
        print(f"测试样本数: {len(test_loader.dataset)}")

        for batch_idx, (images, masks, meta_list) in enumerate(test_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            batch_size = images.size(0)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # 更新指标
            test_loss_meter.update(loss.item(), batch_size)
            metric_calc.update(outputs, masks)

            # 保存预测结果
            if save_predictions and save_dir:
                predictions = torch.sigmoid(outputs) > 0.5

                # 处理元数据 - DataLoader可能会将dict的列表转换为dict的列表
                processed_meta_list = []
                if isinstance(meta_list, dict):
                    # 如果是batch后的dict，每个key对应一个列表
                    meta_keys = list(meta_list.keys())
                    # 使用batch_size，但每个key需要单独处理（可能是list或tensor）
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

                for i in range(batch_size):
                    # 获取元数据
                    meta = processed_meta_list[i] if i < len(processed_meta_list) else {}
                    img_path = meta.get('image_path', f'img_{batch_idx * batch_size + i}')
                    img_name = os.path.splitext(os.path.basename(str(img_path)))[0]

                    # 保存预测mask
                    pred_mask = predictions[i].squeeze().cpu().numpy().astype(np.uint8) * 255
                    pred_path = os.path.join(pred_dir, f'{img_name}_pred.png')
                    cv2.imwrite(pred_path, pred_mask)

                    # 保存可视化结果
                    if visualize:
                        self._save_visualization(
                            images[i], masks[i], outputs[i],
                            os.path.join(vis_dir, f'{img_name}_vis.png')
                        )

                    # 记录结果
                    all_results.append({
                        'image_path': img_path,
                        'pred_path': pred_path,
                        'image_name': img_name
                    })

            # 显示进度
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(test_loader) - 1:
                print(f"  处理进度: [{batch_idx + 1}/{len(test_loader)}] "
                      f"Loss: {test_loss_meter.avg:.4f}")

        # 计算最终指标
        metrics = metric_calc.compute()
        miou = metric_calc.compute_mIoU()

        # 计算FPS
        fps = self._compute_fps()

        test_results = {
            'loss': test_loss_meter.avg,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'iou': metrics['iou'],
            'miou': miou,
            'fps': fps
        }

        # 保存测试结果
        if save_dir:
            # 在配置中记录检查点路径
            test_results['checkpoint_path'] = self.config.get('checkpoint_path', 'N/A')
            self._save_test_results(test_results, all_results, save_dir)

        return test_results

    def _compute_fps(self, input_size=(512, 512), num_warmup=10, num_iters=100) -> float:
        """计算模型推理速度"""
        dummy_input = torch.randn(1, 3, *input_size).to(self.device)

        try:
            # 预热
            for _ in range(num_warmup):
                _ = self.model(dummy_input)

            # 同步GPU
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # 计时测试
            start_time = time.time()
            for _ in range(num_iters):
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / num_iters
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
            return fps
        except RuntimeError as e:
            # 如果尺寸不匹配，返回0并打印警告
            print(f"  警告: 无法计算FPS (尺寸不匹配: {input_size}): {e}")
            return 0.0

    def _save_visualization(
            self,
            image: torch.Tensor,
            mask: torch.Tensor,
            output: torch.Tensor,
            save_path: str
    ):
        """保存可视化结果（原图、真值、预测、叠加）"""
        import matplotlib.pyplot as plt

        # 处理张量
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        mask = mask.squeeze().cpu().numpy()
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)

        # 创建叠加图
        overlay = img.copy()
        mask_binary = (mask > 0.5).astype(np.float32)
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - mask_binary) + mask_binary
        overlay[:, :, 1] = overlay[:, :, 1] * (1 - mask_binary)
        overlay[:, :, 2] = overlay[:, :, 2] * (1 - mask_binary)

        # 创建图形
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title(f'Prediction\n(Prob)')
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

        # 保存指标
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
            f.write("=" * 60 + "\n")
            f.write("裂缝分割网络 - 测试报告\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            f.write("【模型配置】\n")
            f.write(f"  检查点: {self.config.get('checkpoint_path', 'N/A')}\n")
            f.write(f"  设备: {self.device}\n\n")

            f.write("【测试指标】\n")
            f.write(f"  损失 (Loss):      {metrics['loss']:.4f}\n")
            f.write(f"  F1 分数:          {metrics['f1']:.4f}\n")
            f.write(f"  IoU:              {metrics['iou']:.4f}\n")
            f.write(f"  mIoU:             {metrics['miou']:.4f}\n")
            f.write(f"  精确率 (Precision): {metrics['precision']:.4f}\n")
            f.write(f"  召回率 (Recall):    {metrics['recall']:.4f}\n")
            f.write(f"  FPS:              {metrics['fps']:.2f}\n\n")

            f.write("【文件输出】\n")
            f.write(f"  预测结果保存: {len(results)} 张\n")
            f.write(f"  预测目录: {os.path.join(save_dir, 'predictions')}\n")
            f.write(f"  可视化目录: {os.path.join(save_dir, 'visualizations')}\n\n")

            f.write("=" * 60 + "\n")

        print(f"\n测试结果已保存到: {save_dir}")
        print(f"  - 指标文件: {metrics_path}")
        print(f"  - 报告文件: {report_path}")

    def train(self) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        返回：
            训练历史字典
        """
        print("=" * 60)
        print(f"开始训练")
        print(f"设备: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")

            # 训练
            train_metrics = self.train_one_epoch(epoch)
            print(f"  训练损失: {train_metrics['loss']:.4f}, 时间: {train_metrics['time']:.2f}s")

            # 验证
            val_metrics = self.validate(epoch)
            print(f"  验证损失: {val_metrics['loss']:.4f}")
            # print(f"  F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}, mIoU: {val_metrics['miou']:.4f}")
            # print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            # 【新日志格式】：直接展示 ODS, OIS 以及搜出来的最适阈值
            # 【修改点】：打印 ODS F1, OIS F1, 和最佳阈值
            print(
                f"  ODS F1: {val_metrics['f1']:.4f}, OIS F1: {val_metrics.get('ois_f1', 0):.4f}, "
                f"最优阈值: {val_metrics.get('best_threshold', 0.5):.2f}")
            print(f"  IoU: {val_metrics['iou']:.4f}, mIoU: {val_metrics['miou']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_miou'].append(val_metrics['miou'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('Metrics/f1_score', val_metrics['f1'], epoch)
            self.writer.add_scalar('Metrics/iou', val_metrics['iou'], epoch)
            self.writer.add_scalar('Metrics/miou', val_metrics['miou'], epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型，除非f1>0.55，节省空间
            if val_metrics['f1'] > self.best_f1 and val_metrics['f1'] > 0.55:
                self.best_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  保存最佳模型 (F1: {self.best_f1:.4f})")

            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        # 关闭TensorBoard writer
        self.writer.close()

        print("\n" + "=" * 60)
        print(f"训练完成!")
        print(f"最佳F1: {self.best_f1:.4f} (Epoch {self.best_epoch})")
        print(f"TensorBoard日志保存在: {self.log_dir}")
        print("=" * 60)

        return self.model, self.history

    def save_checkpoint(
            self,
            epoch: int,
            metrics: Dict[str, float],
            is_best: bool = False
    ):
        """
        保存检查点
        
        参数：
            epoch: 当前epoch
            metrics: 指标字典
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存路径
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        参数：
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"加载检查点: Epoch {checkpoint['epoch']}")
        return checkpoint


def train_crack_net(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict
) -> Tuple[nn.Module, Dict]:
    """
    训练裂缝分割网络的便捷函数
    
    参数：
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
        
    返回：
        训练后的模型和训练历史
    """
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 训练
    history = trainer.train()

    return model, history


def get_loss_function(loss_config: Dict):
    """
    根据配置获取损失函数
    
    参数：
        loss_config: 损失函数配置
        
    返回：
        损失函数实例
    """
    loss_type = loss_config.get('loss_type', 'bce_dice')

    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('focal_alpha', 0.25),
            gamma=loss_config.get('focal_gamma', 2.0)
        )
    elif loss_type == 'bce_dice':
        return BCEDiceLoss()
    elif loss_type == 'tversky':
        return TverskyLoss()
    else:  # combined
        return CombinedLoss(
            dice_weight=loss_config.get('dice_weight', 1.0),
            focal_weight=loss_config.get('focal_weight', 1.0),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0)
        )


def get_scheduler(optimizer, scheduler_type: str, epochs: int, warmup_epochs: int = 0):
    """
    获取学习率调度器
    
    参数：
        optimizer: 优化器
        scheduler_type: 调度器类型 ('step', 'cosine', 'plateau')
        epochs: 总训练轮数
        warmup_epochs: 预热轮数
        base_lr: 基础学习率（用于warmup）
        
    返回：
        学习率调度器或调度器列表（含warmup）
    """
    # 主调度器
    if scheduler_type == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5, last_epoch=-1)
    elif scheduler_type == 'plateau':
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        raise ValueError("scheduler_type must be 'step', 'cosine' or 'plateau'")
    return main_scheduler


def get_args():
    parser = argparse.ArgumentParser(description='裂缝分割网络训练')

    # 数据参数
    parser.add_argument('--dataset', type=str, default='cfd',
                        choices=['crack500', 'cfd', 'sun520'],
                        help='数据集名称')
    parser.add_argument('--data_root', type=str, default='/mnt/d/dev/data/CFD',
                        help='数据根目录')
    parser.add_argument('--input_size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载工作进程数')

    # 模型参数
    parser.add_argument('--base_channels', type=int, default=96,
                        help='基础通道数')
    parser.add_argument('--d_state', type=int, default=16,
                        help='Mamba状态维度')
    parser.add_argument('--use_gbc', default=True,
                        help='使用GBC模块')
    parser.add_argument('--use_aspp', action='store_true',
                        help='使用ASPP模块')
    # @kimi 新增: drop_path_rate命令行参数
    parser.add_argument('--drop_path_rate', type=float, default=0.2,
                        help='随机深度丢弃概率 (默认: 0.0，建议: 0.1-0.2)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='学习率调度器')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='预热轮数')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--grad_clip', type=float, default=2.0,
                        help='梯度裁剪阈值')

    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='bce_dice',
                        choices=['dice', 'focal', 'combined', 'bce_dice', 'tversky'],
                        help='损失函数类型')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Dice损失权重')
    parser.add_argument('--bce_weight', type=float, default=0.5, help='Bce损失权重')
    parser.add_argument('--focal_weight', type=float, default=0.5,
                        help='Focal损失权重')

    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志记录间隔')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--test_only', action='store_true',
                        help='仅进行测试（在验证集上）')
    parser.add_argument('--eval_test', action='store_true',
                        help='在测试集上评估模型')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='测试时使用的模型检查点路径')
    parser.add_argument('--save_predictions', action='store_true',
                        help='保存测试集的预测结果')
    parser.add_argument('--visualize', action='store_true',
                        help='保存测试集的可视化结果')
    parser.add_argument('--test_save_dir', type=str, default='./test_results',
                        help='测试结果保存目录')

    args = parser.parse_args()
    return args


def main():
    """主训练函数"""
    args = get_args()

    # 处理混合精度参数
    use_amp = args.use_amp

    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # @kimi 修改: 根据数据集名称和日期生成保存路径
    # 原代码直接使用 args.save_dir，现在改为: output_train/数据集名称_年_月_日
    # 理由: 便于组织和管理不同数据集、不同日期的训练结果
    if args.save_dir == './checkpoints':
        current_date = datetime.now().strftime('%Y_%m_%d_%H_%M')
        args.save_dir = f'output_train/{args.dataset}_{current_date}'
        print(f"训练结果保存路径: {args.save_dir}")

    # 创建配置
    config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_dir': args.save_dir,
        'log_interval': args.log_interval,
        'use_amp': use_amp,
        'grad_clip': args.grad_clip,
        'checkpoint_path': args.checkpoint if args.checkpoint else None,
    }

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"配置已保存到: {config_path}")

    # 创建数据加载器
    print("\n加载数据...")

    train_loader = get_dataloader(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='train',
        batch_size=args.batch_size,
        target_size=args.input_size,
        num_workers=args.num_workers
    )

    # 根据数据集确定验证集名称
    val_split = 'validation' if args.dataset == 'crack500' else 'val'
    val_loader = get_dataloader(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split=val_split,
        batch_size=args.batch_size,
        target_size=args.input_size,
        num_workers=args.num_workers
    )

    print(f"训练集大小: {len(train_loader)}")
    print(f"验证集大小: {len(val_loader)}")

    # 创建模型
    print("\n创建模型...")

    # @kimi 新增: 根据base_channels动态调整num_heads_list
    # 目标: 保持head_dim在32-128之间，确保注意力机制稳定
    if args.base_channels <= 32:
        num_heads_list = [2, 2, 4, 4]
    elif args.base_channels <= 64:
        num_heads_list = [4, 4, 8, 8]
    elif args.base_channels <= 128:
        num_heads_list = [8, 8, 12, 12]
    else:
        num_heads_list = [12, 12, 16, 16]

    # 计算预期的head_dim用于检查
    encoder_channels = [
        args.base_channels,
        args.base_channels * 2,
        args.base_channels * 4,
        args.base_channels * 8
    ]
    print(f"  自适应注意力头配置: {num_heads_list}")
    for i, (ch, heads) in enumerate(zip(encoder_channels, num_heads_list)):
        head_dim = ch // heads
        print(f"    Stage {i + 1}: dim={ch}, heads={heads}, head_dim={head_dim}")

    # @kimi 修改: 添加drop_path_rate参数和num_heads_list
    # model = CrackSegmentationNet(
    #     in_channels=3,
    #     num_classes=1,
    #     base_channels=args.base_channels,
    #     input_size=args.input_size,
    #     d_state=args.d_state,
    #     use_gbc=args.use_gbc,
    #     use_aspp=args.use_aspp,
    #     drop_path_rate=args.drop_path_rate,  # @kimi 新增: 传递drop_path_rate参数
    #     num_heads_list=num_heads_list  # @kimi 新增: 传递自适应num_heads_list
    # )
    model = CrackSegmentationNetV2(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels,
        input_size=args.input_size,
        d_state=args.d_state,
        use_gbc=args.use_gbc,
        drop_path_rate=args.drop_path_rate,  # @kimi 新增: 传递drop_path_rate参数
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建损失函数
    loss_config = {
        'loss_type': args.loss_type,
        'dice_weight': args.dice_weight,
        'focal_weight': args.focal_weight,
        'bce_weight': args.bce_weight
    }
    criterion = get_loss_function(loss_config)
    print(f"使用损失函数: {args.loss_type}")

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 创建学习率调度器
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)
    if scheduler:
        print(f"使用学习率调度器: {args.scheduler}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )

    # 恢复训练
    if args.resume:
        print(f"\n恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 仅测试模式（在验证集上）
    if args.test_only:
        if args.checkpoint:
            print(f"\n加载检查点: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        print("\n开始在验证集上测试...")
        test_metrics = trainer.validate(epoch=0)
        print("\n验证集测试结果:")
        print(f"  损失: {test_metrics['loss']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  IoU: {test_metrics['iou']:.4f}")
        print(f"  mIoU: {test_metrics['miou']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        return

    # 在测试集上评估
    if args.eval_test:
        if not args.checkpoint:
            print("错误: 使用 --eval_test 时必须指定 --checkpoint")
            return

        print(f"\n加载检查点: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

        # 加载测试集
        print("\n加载测试集...")
        try:
            test_loader = get_dataloader(
                dataset_name=args.dataset,
                root_dir=args.data_root,
                split='test',
                batch_size=args.batch_size,
                target_size=args.input_size,
                num_workers=args.num_workers
            )
            print(f"测试集大小: {len(test_loader.dataset)}")
        except Exception as e:
            print(f"测试集加载失败: {e}")
            print("某些数据集可能没有独立的测试集，将使用验证集代替")
            test_loader = val_loader

        # 在测试集上评估
        test_results = trainer.test(
            test_loader=test_loader,
            save_predictions=args.save_predictions,
            save_dir=args.test_save_dir,
            visualize=args.visualize
        )

        print("\n" + "=" * 60)
        print("测试集评估结果")
        print("=" * 60)
        print(f"  损失 (Loss):      {test_results['loss']:.4f}")
        print(f"  F1 分数:          {test_results['f1']:.4f}")
        print(f"  IoU:              {test_results['iou']:.4f}")
        print(f"  mIoU:             {test_results['miou']:.4f}")
        print(f"  精确率 (Precision): {test_results['precision']:.4f}")
        print(f"  召回率 (Recall):    {test_results['recall']:.4f}")
        print(f"  FPS:              {test_results['fps']:.2f}")
        print("=" * 60)
        return

    # 训练
    print("\n开始训练...")
    history = None
    try:
        model, history = trainer.train()
    except KeyboardInterrupt:
        print("\n\n========================================")
        print("训练被用户中断 (Ctrl+C)")
        print("========================================")
        # 获取当前训练历史（即使中断了也保存已训练的记录）
        history = trainer.history
    finally:
        # 保存训练历史（无论正常结束还是中断都会执行）
        if history is not None:
            history_path = os.path.join(args.save_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"训练历史已保存到: {history_path}")


# =======开始训练============
if __name__ == "__main__":
    main()
