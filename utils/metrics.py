"""
评估指标实现
包含Precision, Recall, F1-score, mIoU, FPS等
"""

import torch
from typing import Dict, List, Tuple, Optional


class RigorousCalculator:
    """
    裂缝分割指标计算器 (支持 ODS, OIS 全自动阈值搜索)

    采用 GPU 极速并行计算，瞬间遍历 0.01~0.99 的 99 个阈值。
    无需等待，每个 Epoch 实时输出最佳阈值。
    """

    def __init__(self, threshold: float = 0.5):
        # 参数 threshold 仅为兼容老代码保留，实际将被动态阈值取代
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 生成 0.01 到 0.99 的 99 个阈值
        self.thresholds = torch.linspace(0.01, 0.99, 99, device=self.device)
        self.num_th = len(self.thresholds)
        self.reset()

    def reset(self):
        """重置所有阈值下的计数器"""
        # 为 99 个阈值分别记录 TP, FP, FN, TN
        self.tp = torch.zeros(self.num_th, device=self.device)
        self.fp = torch.zeros(self.num_th, device=self.device)
        self.fn = torch.zeros(self.num_th, device=self.device)
        self.tn = torch.zeros(self.num_th, device=self.device)

        # 用于 OIS 统计
        self.ois_f1_sum = 0.0
        self.img_count = 0

    def update(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> Dict[str, float]:
        """更新 99 个阈值下的计数器"""
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        B = pred.shape[0]

        # 展平图像以加速运算，形如 (B, N)
        pred = pred.view(B, -1)
        target = target.view(B, -1).bool()  # 使用 bool 加速逻辑运算

        f1_table = []

        # 遍历 99 个阈值。在 GPU 上做此级联位运算耗时极短，且不占用额外显存
        for i, th in enumerate(self.thresholds):
            p_bin = pred > th

            # 按样本(dim=1)分别计算 TP, FP, FN, TN，以便给 OIS 使用
            tp = (p_bin & target).sum(dim=1)
            fp = (p_bin & ~target).sum(dim=1)
            fn = (~p_bin & target).sum(dim=1)
            tn = (~p_bin & ~target).sum(dim=1)

            # 累加到全局 (用于 ODS)
            self.tp[i] += tp.sum()
            self.fp[i] += fp.sum()
            self.fn[i] += fn.sum()
            self.tn[i] += tn.sum()

            # 计算此阈值下的逐图片 F1 (用于 OIS)
            eps = 1e-6
            prec = tp.float() / (tp + fp + eps)
            rec = tp.float() / (tp + fn + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)
            f1_table.append(f1.unsqueeze(1))  # (B, 1)

        # 统计 OIS (每张图在 99 个结果中挑一个最大的 F1)
        f1_table = torch.cat(f1_table, dim=1)  # (B, 99)
        best_f1, _ = f1_table.max(dim=1)
        self.ois_f1_sum += best_f1.sum().item()
        self.img_count += B

        return {'f1': 0.0}  # 批量更新时不返回实际值，以最后 compute() 为准

    def compute(self) -> Dict[str, float]:
        """计算最终的最佳指标"""
        eps = 1e-6
        # 计算 99 个阈值下的全局指标
        prec = self.tp / (self.tp + self.fp + eps)
        rec = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)

        # --- ODS F1 寻优 ---
        best_idx = torch.argmax(f1)
        ods_f1 = f1[best_idx].item()
        ods_prec = prec[best_idx].item()
        ods_rec = rec[best_idx].item()
        ods_iou = iou[best_idx].item()
        best_th = self.thresholds[best_idx].item()

        # --- OIS F1 ---
        ois_f1 = self.ois_f1_sum / max(self.img_count, 1)

        # 强制将 ODS F1 映射到 'f1' 键值，这样 train.py 就会自动拿 ODS 指标去保存最佳模型！
        return {
            'f1': ods_f1,
            'precision': ods_prec,
            'recall': ods_rec,
            'iou': ods_iou,
            'ods_f1': ods_f1,
            'ois_f1': ois_f1,
            'best_threshold': best_th,
        }

    def compute_mIoU(self) -> float:
        """根据寻优得到的最佳阈值计算 mIoU"""
        eps = 1e-6
        prec = self.tp / (self.tp + self.fp + eps)
        rec = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        best_idx = torch.argmax(f1)  # 找到使得全局 F1 最大的阈值索引

        tp = self.tp[best_idx].item()
        fp = self.fp[best_idx].item()
        fn = self.fn[best_idx].item()
        tn = self.tn[best_idx].item()

        iou_fg = tp / (tp + fp + fn + eps)
        iou_bg = tn / (tn + fn + fp + eps)

        return (iou_fg + iou_bg) / 2


class MetricCalculator:
    """
    分割指标计算器
    
    计算Precision, Recall, F1-score, mIoU等指标
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """重置所有计数器"""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_samples = 0

    def update(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> Dict[str, float]:
        """
        更新计数器
        
        参数：
            pred: 预测概率图 (B, 1, H, W) 或 (B, H, W)
            target: 目标标签 (B, 1, H, W) 或 (B, H, W)
            
        返回：
            当前批次的指标字典
        """
        # 确保维度正确
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        # 二值化预测
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > self.threshold).float()

        # 计算TP, FP, FN, TN
        tp = (pred_binary * target_binary).sum().item()
        fp = ((1 - target_binary) * pred_binary).sum().item()
        fn = (target_binary * (1 - pred_binary)).sum().item()
        tn = ((1 - target_binary) * (1 - pred_binary)).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.total_samples += pred.shape[0]

        # 返回当前批次指标
        return self._compute_metrics(tp, fp, fn, tn)

    def _compute_metrics(
            self,
            tp: float,
            fp: float,
            fn: float,
            tn: float,
            eps: float = 1e-6
    ) -> Dict[str, float]:
        """计算各项指标"""
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # IoU
        iou = tp / (tp + fp + fn + eps)

        # 特异性
        specificity = tn / (tn + fp + eps)

        # 准确率
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'specificity': specificity,
            'accuracy': accuracy
        }

    def compute(self) -> Dict[str, float]:
        """计算所有累积样本的指标"""
        return self._compute_metrics(self.tp, self.fp, self.fn, self.tn)

    def compute_mIoU(self) -> float:
        """计算平均IoU (mIoU)"""
        # 对于二分类，mIoU = (IoU_背景 + IoU_前景) / 2
        eps = 1e-6

        # 前景IoU
        iou_fg = self.tp / (self.tp + self.fp + self.fn + eps)

        # 背景IoU
        iou_bg = self.tn / (self.tn + self.fn + self.fp + eps)

        return (iou_fg + iou_bg) / 2


class AverageMeter:
    """
    平均值计算器
    用于跟踪训练过程中的损失和指标
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FPSCounter:
    """
    FPS计数器
    用于测量推理速度
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.times = []
        self.start_time = None

    def start(self):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        self.start_time = time.time()

    def stop(self):
        """停止计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        end_time = time.time()
        if self.start_time is not None:
            self.times.append(end_time - self.start_time)

    def compute_fps(self) -> float:
        """计算FPS"""
        if len(self.times) == 0:
            return 0.0
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        if len(self.times) == 0:
            return {'fps': 0, 'avg_time': 0, 'min_time': 0, 'max_time': 0}

        return {
            'fps': self.compute_fps(),
            'avg_time': sum(self.times) / len(self.times),
            'min_time': min(self.times),
            'max_time': max(self.times),
            'num_runs': len(self.times)
        }


def compute_metrics_batch(
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    批量计算指标
    
    参数：
        pred: 预测概率图
        target: 目标标签
        threshold: 二值化阈值
        
    返回：
        指标字典
    """
    # 确保维度正确
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # 二值化
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # 按样本计算
    eps = 1e-6

    tp = (pred_binary * target_binary).sum(dim=(1, 2))
    fp = ((1 - target_binary) * pred_binary).sum(dim=(1, 2))
    fn = (target_binary * (1 - pred_binary)).sum(dim=(1, 2))
    tn = ((1 - target_binary) * (1 - pred_binary)).sum(dim=(1, 2))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean(),
        'iou': iou.mean()
    }


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数量
    
    参数：
        model: 模型
        trainable_only: 是否只计算可训练参数
        
    返回：
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_flops(
        model: torch.nn.Module,
        input_size: Tuple[int, int, int] = (3, 512, 512)
) -> int:
    """
    估算模型FLOPs
    
    注意：这是一个简化估算，实际FLOPs可能有所不同
    
    参数：
        model: 模型
        input_size: 输入尺寸 (C, H, W)
        
    返回：
        估算的FLOPs
    """
    # 创建输入
    x = torch.randn(1, *input_size)

    # 使用thop库如果可用
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,))
        return flops
    except ImportError:
        print("警告: thop库未安装，返回简化估算")
        # 简化估算：参数量 * 2（近似）
        return count_parameters(model) * 2


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试评估指标模块")
    print("=" * 60)

    # 创建测试数据
    batch_size = 4
    height, width = 64, 64

    # 模拟预测和目标
    pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    target = (torch.randn(batch_size, 1, height, width) > 0).float()

    # 测试MetricCalculator
    print("\n[1] 测试MetricCalculator...")
    metric_calc = MetricCalculator(threshold=0.5)

    # 更新并获取批次指标
    batch_metrics = metric_calc.update(pred, target)
    print(f"  批次指标:")
    for name, value in batch_metrics.items():
        print(f"    {name}: {value:.4f}")

    # 再更新一次
    metric_calc.update(pred, target)

    # 获取累积指标
    total_metrics = metric_calc.compute()
    print(f"  累积指标:")
    for name, value in total_metrics.items():
        print(f"    {name}: {value:.4f}")

    # 测试mIoU
    miou = metric_calc.compute_mIoU()
    print(f"  mIoU: {miou:.4f}")

    # 测试AverageMeter
    print("\n[2] 测试AverageMeter...")
    meter = AverageMeter()
    for i in range(10):
        meter.update(i * 0.1)
    print(f"  平均值: {meter.avg:.4f}")
    print(f"  总和: {meter.sum:.4f}")
    print(f"  计数: {meter.count}")

    # 测试批量计算
    print("\n[3] 测试compute_metrics_batch...")
    batch_metrics = compute_metrics_batch(pred, target)
    print(f"  批量指标:")
    for name, value in batch_metrics.items():
        print(f"    {name}: {value.item():.4f}")

    # 测试边界情况
    print("\n[4] 测试边界情况...")

    # 完美预测
    perfect_pred = target.clone()
    metric_calc.reset()
    metrics = metric_calc.update(perfect_pred, target)
    print(f"  完美预测 F1: {metrics['f1']:.4f}")
    print(f"  完美预测 IoU: {metrics['iou']:.4f}")

    # 全背景
    all_bg = torch.zeros_like(target)
    metric_calc.reset()
    metrics = metric_calc.update(all_bg, target)
    print(f"  全背景预测 Precision: {metrics['precision']:.4f}")
    print(f"  全背景预测 Recall: {metrics['recall']:.4f}")

    print("\n" + "=" * 60)
    print("评估指标模块测试完成！")
    print("=" * 60)
