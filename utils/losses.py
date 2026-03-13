"""
损失函数实现
包含Dice Loss、Focal Loss和组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice损失函数
    
    用于处理类别不平衡问题，直接优化分割区域的重叠度
    
    Dice系数 = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice系数
    
    参数：
        smooth: 平滑因子，避免除零
        reduction: 归约方式 ('mean', 'sum', 'none')
        apply_sigmoid: 是否对输入应用sigmoid（输入为logits时设为True）
    """

    def __init__(self, smooth: float = 1., reduction: str = 'mean', apply_sigmoid: bool = True):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.apply_sigmoid = apply_sigmoid

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Dice损失
        
        参数：
            pred: 预测logits或概率图 (B, 1, H, W) 或 (B, H, W)
            target: 目标标签 (B, 1, H, W) 或 (B, H, W)，值为0或1
            
        返回：
            Dice损失值
        """
        # 如果输入是logits，先应用sigmoid
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        # 确保维度正确
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # 展平
        pred_flat = pred.flatten(1)  # (B, H*W)
        target_flat = target.flatten(1)  # (B, H*W)

        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        # Dice系数
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Dice损失
        loss = 1.0 - dice

        # 归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal损失函数
    
    通过降低易分类样本的权重，使模型更关注难分类样本
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    参数：
        alpha: 正样本权重，默认0.25
        gamma: 聚焦参数，默认2.0
        reduction: 归约方式
        apply_sigmoid: 是否对输入应用sigmoid（输入为logits时设为True）
    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = 'mean',
            apply_sigmoid: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.apply_sigmoid = apply_sigmoid

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Focal损失
        
        参数：
            pred: 预测logits或概率图 (B, 1, H, W) 或 (B, H, W)
            target: 目标标签 (B, 1, H, W) 或 (B, H, W)，值为0或1
            
        返回：
            Focal损失值
        """
        # 如果输入是logits，先应用sigmoid
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        # 确保维度正确
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # 避免数值问题
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)

        # 计算交叉熵
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # 计算p_t
        p_t = pred * target + (1 - pred) * (1 - target)

        # 计算alpha权重
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # 计算focal权重
        focal_weight = (1 - p_t) ** self.gamma

        # Focal损失
        loss = alpha_t * focal_weight * bce

        # 归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice组合损失
    
    参数：
        bce_weight: BCE损失权重
        dice_weight: Dice损失权重
    """

    def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss，内部会应用sigmoid
        self.dice = DiceLoss(apply_sigmoid=False)  # 已经在外面应用了sigmoid

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        # 确保维度正确
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        bce_loss = self.bce(pred, target.float())
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    Dice Loss + Focal Loss，用于处理裂缝分割中的类别不平衡问题
    
    L_total = λ_dice * L_dice + λ_focal * L_focal
    
    参数：
        dice_weight: Dice损失权重
        focal_weight: Focal损失权重
        focal_alpha: Focal损失的alpha参数
        focal_gamma: Focal损失的gamma参数
    """

    def __init__(
            self,
            dice_weight: float = 1.0,
            focal_weight: float = 1.0,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # 默认apply_sigmoid=True，因为模型输出的是logits
        self.dice_loss = DiceLoss(apply_sigmoid=True)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, apply_sigmoid=True)

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算组合损失
        
        参数：
            pred: 预测概率图
            target: 目标标签
            
        返回：
            组合损失值
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        return self.dice_weight * dice + self.focal_weight * focal


class TverskyLoss(nn.Module):
    """
    Tversky损失函数
    
    Dice损失的泛化形式，可以调节FP和FN的权重
    
    T = TP / (TP + α*FP + β*FN)
    
    参数：
        alpha: FP权重
        beta: FN权重
        smooth: 平滑因子
        apply_sigmoid: 是否对输入应用sigmoid（输入为logits时设为True）
        
    注意：
        alpha = beta = 0.5 时，等同于Dice损失
        alpha < beta 时，更关注召回率（减少FN）
        alpha > beta 时，更关注精确率（减少FP）
    """

    def __init__(
            self,
            alpha: float = 0.3,
            beta: float = 0.7,
            smooth: float = 1e-6,
            apply_sigmoid: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Tversky损失
        
        参数：
            pred: 预测logits或概率图 (B, 1, H, W) 或 (B, H, W)
            target: 目标标签 (B, 1, H, W) 或 (B, H, W)，值为0或1
            
        返回：
            Tversky损失值
        """
        # 如果输入是logits，先应用sigmoid
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        # 确保维度正确
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # 展平
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        # 计算TP, FP, FN
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = ((1 - target_flat) * pred_flat).sum(dim=1)
        fn = (target_flat * (1 - pred_flat)).sum(dim=1)

        # Tversky指数
        tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1.0 - tversky.mean()


class EdgeLoss(nn.Module):
    """
    边缘损失函数
    
    用于边缘监督分支，强调裂缝边缘的准确性
    
    参数：
        edge_width: 边缘宽度
    """

    def __init__(self, edge_width: int = 3):
        super().__init__()
        self.edge_width = edge_width
        self.dice_loss = DiceLoss()

    def get_edge_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        从分割掩码提取边缘
        
        参数：
            mask: 分割掩码
            
        返回：
            边缘掩码
        """
        # 使用形态学操作提取边缘
        kernel_size = self.edge_width * 2 + 1

        # 膨胀
        dilated = F.max_pool2d(
            mask.unsqueeze(1).float(),
            kernel_size=kernel_size,
            stride=1,
            padding=self.edge_width
        )

        # 腐蚀
        eroded = -F.max_pool2d(
            -mask.unsqueeze(1).float(),
            kernel_size=kernel_size,
            stride=1,
            padding=self.edge_width
        )

        # 边缘 = 膨胀 - 腐蚀
        edge = dilated - eroded

        return edge.squeeze(1)

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算边缘损失
        
        参数：
            pred: 预测概率图
            target: 目标标签
            
        返回：
            边缘损失值
        """
        # 提取边缘
        edge_target = self.get_edge_mask(target)

        # 计算边缘上的损失
        edge_loss = self.dice_loss(pred * edge_target, target * edge_target)

        return edge_loss


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试损失函数")
    print("=" * 60)

    # 创建测试数据
    batch_size = 4
    height, width = 64, 64

    # 模拟预测和目标（使用logits作为输入，更符合实际训练场景）
    pred_logits = torch.randn(batch_size, 1, height, width)
    pred = torch.sigmoid(pred_logits)  # 用于打印范围
    target = (torch.randn(batch_size, 1, height, width) > 0).float()

    # 注意：现在损失函数默认 apply_sigmoid=True，可以直接接受 logits 输入

    print(f"\n预测形状: {pred.shape}")
    print(f"目标形状: {target.shape}")
    print(f"预测值范围: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"目标正样本比例: {target.mean():.4f}")

    # 测试DiceLoss（使用logits输入）
    print("\n[1] 测试DiceLoss...")
    dice_loss = DiceLoss()  # 默认 apply_sigmoid=True
    loss = dice_loss(pred_logits, target)
    print(f"  Dice Loss (logits输入): {loss.item():.4f}")

    # 验证与概率输入的一致性
    dice_loss_no_sigmoid = DiceLoss(apply_sigmoid=False)
    loss_prob = dice_loss_no_sigmoid(pred, target)
    print(f"  Dice Loss (概率输入): {loss_prob.item():.4f}")

    # 测试FocalLoss（使用logits输入）
    print("\n[2] 测试FocalLoss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)  # 默认 apply_sigmoid=True
    loss = focal_loss(pred_logits, target)
    print(f"  Focal Loss (logits输入): {loss.item():.4f}")

    # 验证与概率输入的一致性
    focal_loss_no_sigmoid = FocalLoss(alpha=0.25, gamma=2.0, apply_sigmoid=False)
    loss_prob = focal_loss_no_sigmoid(pred, target)
    print(f"  Focal Loss (概率输入): {loss_prob.item():.4f}")

    # 测试不同gamma值
    for gamma in [0.5, 1.0, 2.0, 3.0]:
        fl = FocalLoss(gamma=gamma)
        loss = fl(pred_logits, target)
        print(f"  Focal Loss (gamma={gamma}): {loss.item():.4f}")

    # 测试BCEDiceLoss（使用logits输入）
    print("\n[3] 测试BCEDiceLoss...")
    bce_dice = BCEDiceLoss()
    loss = bce_dice(pred_logits, target)
    print(f"  BCE+Dice Loss (logits输入): {loss.item():.4f}")

    # 测试CombinedLoss（使用logits输入）
    print("\n[4] 测试CombinedLoss...")
    combined = CombinedLoss(dice_weight=1.0, focal_weight=1.0)
    loss = combined(pred_logits, target)
    print(f"  Combined Loss (logits输入): {loss.item():.4f}")

    # 测试TverskyLoss（使用logits输入）
    print("\n[5] 测试TverskyLoss...")
    tversky = TverskyLoss(alpha=0.3, beta=0.7)
    loss = tversky(pred_logits, target)
    print(f"  Tversky Loss (logits输入): {loss.item():.4f}")

    # 测试不同alpha/beta组合
    for alpha, beta in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]:
        tl = TverskyLoss(alpha=alpha, beta=beta)
        loss = tl(pred_logits, target)
        print(f"  Tversky (α={alpha}, β={beta}): {loss.item():.4f}")

    # 测试EdgeLoss（使用logits输入）
    print("\n[6] 测试EdgeLoss...")
    edge_loss = EdgeLoss(edge_width=2)
    loss = edge_loss(pred_logits, target)
    print(f"  Edge Loss (logits输入): {loss.item():.4f}")

    # 测试梯度（使用logits输入）
    print("\n[7] 测试梯度反向传播...")
    pred_grad = torch.randn(batch_size, 1, height, width, requires_grad=True)
    loss = combined(pred_grad, target)  # 直接传入logits
    loss.backward()
    print("  梯度计算成功 (logits输入)")

    # 测试边界情况
    print("\n[8] 测试边界情况...")

    # 完美预测（使用概率输入，并设置apply_sigmoid=False）
    dice_loss_no_sigmoid = DiceLoss(apply_sigmoid=False)
    perfect_pred = target.clone()
    loss = dice_loss_no_sigmoid(perfect_pred, target)
    print(f"  完美预测 Dice Loss: {loss.item():.6f}")

    # 完全错误预测
    wrong_pred = 1 - target
    loss = dice_loss_no_sigmoid(wrong_pred, target)
    print(f"  完全错误 Dice Loss: {loss.item():.4f}")

    # 全部预测为背景
    all_bg = torch.zeros_like(pred)
    loss = dice_loss_no_sigmoid(all_bg, target)
    print(f"  全背景预测 Dice Loss: {loss.item():.4f}")

    # 测试负数logits（模拟训练初期的大负数输出）
    print("\n[9] 测试极端logits输入...")
    extreme_logits = torch.full_like(pred_logits, -10.0)  # 很大的负数
    loss = dice_loss(extreme_logits, target)
    print(f"  大负数logits Dice Loss: {loss.item():.4f} (应该接近1.0)")

    extreme_logits = torch.full_like(pred_logits, 10.0)  # 很大的正数
    loss = dice_loss(extreme_logits, target)
    print(f"  大正数logits Dice Loss: {loss.item():.4f} (应该接近目标比例)")

    print("\n" + "=" * 60)
    print("所有损失函数测试通过！")
    print("=" * 60)
