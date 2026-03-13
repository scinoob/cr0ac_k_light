"""
双分支融合模块和编码器实现
包含自适应加权融合、双分支模块A和完整编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np  # @kimi 新增: 用于计算drop_path调度
# from timm.models._efficientnet_blocks import DepthwiseSeparableConv

from .base_modules import PConv, PConvBlock, GBC, ConvBNReLU, DepthwiseSeparableConv
from .mamba import MambaBranch
from .vit import ViTBranch, LightSwinBlock


class AdaptiveWeightedFusion(nn.Module):
    """
    自适应加权融合模块
    
    通过通道注意力机制自适应融合Mamba分支和ViT分支的特征
    
    流程：
    1. 对两个分支输出进行全局平均池化
    2. 拼接后通过两层全连接和Softmax生成权重
    3. 加权融合
    
    参数：
        channels: 通道数
        reduction: 中间层缩减比例
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        self.channels = channels
        mid_channels = channels // reduction

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels * 2),
            nn.Softmax(dim=-1)
        )

        # 现在改成1*1卷积，提高效率
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(channels * 2, mid_channels, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, channels*2, kernel_size=1),
        #     nn.Softmax(dim=-1)
        # )

    def forward(self, x_mamba: torch.Tensor, x_vit: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x_mamba: Mamba分支输出 (B, C, H, W)
            x_vit: ViT分支输出 (B, C, H, W)
            
        返回：
            融合特征 (B, C, H, W)
        """
        B, C, H, W = x_mamba.shape

        # 全局平均池化
        z_mamba = F.adaptive_avg_pool2d(x_mamba, 1).view(B, C)  # (B, C)
        z_vit = F.adaptive_avg_pool2d(x_vit, 1).view(B, C)  # (B, C)

        # 拼接
        z = torch.cat([z_mamba, z_vit], dim=-1)  # (B, 2C)

        # 生成权重
        weights = self.fusion(z)  # (B, 2C)
        alpha, beta = weights.chunk(2, dim=-1)  # 各 (B, C)

        # 加权融合
        alpha = alpha.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        out = alpha * x_mamba + beta * x_vit

        return out


class SobelEdgeBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # 初始化sobel核
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)
        self.pwconv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.pwconv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 假设x是RGB，先转灰度
        gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        edge_x = self.sobel_x(gray)
        edge_y = self.sobel_y(gray)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        # 将单通道边缘扩展到in_channels
        edge_expanded = edge.expand(-1, x.shape[1], -1, -1)
        # 与原始特征融合（如相加）然后PWConv
        fused = x + edge_expanded
        return self.pwconv2(self.relu(self.pwconv1(fused)))


class SpatialAttentionFusion(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.conv_reduce = nn.Conv2d(2 * channels, channels // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv_alpha = nn.Conv2d(channels // reduction, 1, 3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_mamba, x_vit):
        cat = torch.cat([x_mamba, x_vit], dim=1)
        x = self.conv_reduce(cat)
        x = self.bn(x)
        x = self.relu(x)
        alpha = self.conv_alpha(x)
        alpha = self.sigmoid(alpha)
        return alpha * x_mamba + (1 - alpha) * x_vit


class DualBranchModuleA(nn.Module):
    """
    双分支模块A
    
    核心编码器模块，包含：
    1. LayerNorm
    2. Mamba分支（蛇形扫描SASS）
    3. ViT分支（轻量Swin）
    4. 自适应加权融合
    5. 特征细化（PConv）
    6. 残差连接
    
    参数：
        channels: 通道数
        input_resolution: 输入分辨率 (H, W)
        d_state: Mamba状态维度
        d_conv: Mamba卷积核大小
        expand: Mamba扩展因子
        num_heads: ViT注意力头数
        window_size: ViT窗口大小
        use_gbc: 是否使用GBC模块
        use_light_vit: 是否使用轻量ViT块
        drop_path_mamba: Mamba分支的随机深度丢弃概率 (新增)
        drop_path_vit: ViT分支的随机深度丢弃概率 (新增)
    """

    def __init__(
            self,
            channels: int,
            input_resolution: Tuple[int, int],
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            num_heads: int = 4,
            window_size: int = 7,
            use_gbc: bool = False,
            use_light_vit: bool = True,
            drop_path_mamba: float = 0.0,  # @kimi 新增: Mamba分支的drop_path概率
            drop_path_vit: float = 0.0,  # @kimi 新增: ViT分支的drop_path概率
    ):
        super().__init__()

        self.channels = channels
        self.input_resolution = input_resolution
        self.use_gbc = use_gbc

        # 输入归一化
        self.norm = nn.LayerNorm(channels)

        # Mamba分支
        self.mamba_branch = MambaBranch(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path=drop_path_mamba  # @kimi 新增: 传递drop_path概率
        )

        # 关闭ViT分支，直接残差连接
        # ViT分支
        # if use_light_vit:
        #     self.vit_branch = LightSwinBlock(
        #         dim=channels,
        #         input_resolution=input_resolution,
        #         num_heads=num_heads,
        #         window_size=min(window_size, min(input_resolution)),
        #         drop_path=drop_path_vit  # @kimi 新增: 传递drop_path概率
        #     )
        # else:
        #     # @kimi 注意: ViTBranch有两个块，需要分配不同的drop_path概率
        #     # 第一个块概率较小，第二个块概率较大（深度增加）
        #     drop_path_vit1 = drop_path_vit * 0.5
        #     drop_path_vit2 = drop_path_vit
        #     self.vit_branch = ViTBranch(
        #         dim=channels,
        #         input_resolution=input_resolution,
        #         num_heads=num_heads,
        #         window_size=min(window_size, min(input_resolution)),
        #         drop_path=(drop_path_vit1, drop_path_vit2)  # @kimi 新增: 传递元组形式的drop_path
        #     )

        # 自适应加权融合
        self.fusion = SpatialAttentionFusion(channels)

        # 特征细化（PConv）
        self.refine = PConvBlock(channels)

        # 可选GBC模块
        if use_gbc:
            self.gbc = GBC(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量 (B, C, H, W)
            
        返回：
            输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        identity = x

        # 可选GBC
        if self.use_gbc:
            x = self.gbc(x)

        # 输入归一化
        x_norm = x.flatten(2).transpose(1, 2)  # (B, L, C)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)

        '''
        观察单独实验vit的效果
        '''
        # 双分支并行
        x_mamba = self.mamba_branch(x_norm)
        # x_vit = self.vit_branch(x_norm)

        # 自适应加权融合
        # x_fused = self.fusion(x_mamba, x_vit)
        # x_fused = x_mamba +  x_vit
        x_fused = self.fusion(x_mamba, x_norm)

        # 特征细化
        x_out = self.refine(x_fused)
        # x_out = self.refine(x_vit)

        # 残差连接
        return x_out + identity


class EncoderStage(nn.Module):
    """
    编码器阶段
    
    包含下采样（可选）和双分支模块
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        input_resolution: 输入分辨率 (H, W)
        downsample: 是否下采样
        **kwargs: 双分支模块参数
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            input_resolution: Tuple[int, int],
            downsample: bool = True,
            **kwargs
    ):
        super().__init__()

        self.downsample = downsample

        # 下采样
        if downsample:
            # self.down = DownSample(in_channels, out_channels)
            # 使用深度可分离卷积进行下采样
            self.down = DepthwiseSeparableConv(in_channels, out_channels, 3, stride=2)
            # 下采样后的分辨率
            self.resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        else:
            self.down = None
            self.resolution = input_resolution
            # 如果通道数不同，使用1x1卷积调整
            if in_channels != out_channels:
                self.down = nn.Conv2d(in_channels, out_channels, 1)

        # 双分支模块
        self.dual_branch = DualBranchModuleA(
            channels=out_channels,
            input_resolution=self.resolution,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量 (B, C_in, H, W)
            
        返回：
            输出张量 (B, C_out, H', W')
        """
        if self.downsample or self.down is not None:
            x = self.down(x)
        x = self.dual_branch(x)
        return x


class Encoder(nn.Module):
    """
    完整编码器
    
    包含Patch Embedding和4个编码器阶段
    
    参数：
        in_channels: 输入通道数（通常为3）
        base_channels: 基础通道数
        input_size: 输入图像尺寸
        d_state: Mamba状态维度
        d_conv: Mamba卷积核大小
        expand: Mamba扩展因子
        num_heads_list: 各阶段注意力头数列表
        window_size: ViT窗口大小
        use_gbc: 是否使用GBC模块
        drop_path_rate: 最大随机深度丢弃概率 (新增，默认0.0)
    """

    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 32,
            input_size: int = 512,
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            num_heads_list: List[int] = [2, 2, 4, 4],
            window_size: int = 7,
            use_gbc: bool = False,
            drop_path_rate: float = 0.0,  # @kimi 新增: 最大随机深度丢弃概率
    ):
        super().__init__()

        self.input_size = input_size
        self.base_channels = base_channels
        self.drop_path_rate = drop_path_rate  # @kimi 新增: 保存drop_path_rate

        # @kimi 新增: 计算各阶段的drop_path概率（线性调度）
        # 理由: 随着网络深度增加，丢弃概率线性增大，有助于深层网络的训练稳定性
        # 共有4个阶段，使用np.linspace从0到drop_path_rate生成调度
        self.num_stages = 4
        # dpr = [0, drop_path_rate/3, 2*drop_path_rate/3, drop_path_rate]
        self.dpr = np.linspace(0, drop_path_rate, self.num_stages).tolist()
        # print(f"[Encoder] Drop path rate schedule: {self.dpr}")  # 调试用

        # Patch Embedding: 3x3卷积，步长2
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 各阶段通道数
        channels = [
            base_channels,  # 32
            base_channels * 2,  # 64
            base_channels * 4,  # 128
            base_channels * 8  # 256
        ]

        # 各阶段分辨率
        resolutions = [
            (input_size // 2, input_size // 2),  # 256x256
            (input_size // 4, input_size // 4),  # 128x128
            (input_size // 8, input_size // 8),  # 64x64
            (input_size // 16, input_size // 16)  # 32x32
        ]

        # @kimi 修改: 为每个阶段传递对应的drop_path概率
        # 编码器阶段1：无下采样，仅特征细化
        self.stage1 = EncoderStage(
            in_channels=channels[0],
            out_channels=channels[0],
            input_resolution=resolutions[0],
            downsample=False,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[0],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=self.dpr[0],  # @kimi 新增: 阶段1的Mamba分支drop_path概率
            drop_path_vit=self.dpr[0]  # @kimi 新增: 阶段1的ViT分支drop_path概率
        )

        # 编码器阶段2-4（深度增加，drop_path概率增大）
        self.stage2 = EncoderStage(
            in_channels=channels[0],
            out_channels=channels[1],
            input_resolution=resolutions[0],
            downsample=True,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[1],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=self.dpr[1],  # @kimi 新增: 阶段2的Mamba分支drop_path概率
            drop_path_vit=self.dpr[1]  # @kimi 新增: 阶段2的ViT分支drop_path概率
        )

        self.stage3 = EncoderStage(
            in_channels=channels[1],
            out_channels=channels[2],
            input_resolution=resolutions[1],
            downsample=True,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[2],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=self.dpr[2],  # @kimi 新增: 阶段3的Mamba分支drop_path概率
            drop_path_vit=self.dpr[2]  # @kimi 新增: 阶段3的ViT分支drop_path概率
        )

        self.stage4 = EncoderStage(
            in_channels=channels[2],
            out_channels=channels[3],
            input_resolution=resolutions[2],
            downsample=True,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[3],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=self.dpr[3],  # @kimi 新增: 阶段4的Mamba分支drop_path概率
            drop_path_vit=self.dpr[3]  # @kimi 新增: 阶段4的ViT分支drop_path概率
        )

        # 保存各阶段通道数
        self.channels = channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        参数：
            x: 输入图像 (B, 3, H, W)
            
        返回：
            各阶段特征列表 [f1, f2, f3, f4]
            f1: (B, 32, H/2, W/2)
            f2: (B, 64, H/4, W/4)
            f3: (B, 128, H/8, W/8)
            f4: (B, 256, H/16, W/16)
        """
        # Patch Embedding
        x = self.patch_embed(x)  # (B, 32, H/2, W/2)

        # 各阶段编码
        f1 = self.stage1(x)  # (B, 32, H/2, W/2)
        f2 = self.stage2(f1)  # (B, 64, H/4, W/4)
        f3 = self.stage3(f2)  # (B, 128, H/8, W/8)
        f4 = self.stage4(f3)  # (B, 256, H/16, W/16)

        return [f1, f2, f3, f4]


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试双分支融合模块和编码器")
    print("=" * 60)

    # 测试参数
    batch_size = 2
    in_channels = 64
    height, width = 32, 32

    # 创建测试输入
    x = torch.randn(batch_size, in_channels, height, width)

    # 测试自适应加权融合
    print("\n[1] 测试AdaptiveWeightedFusion...")
    fusion = AdaptiveWeightedFusion(in_channels)
    x_mamba = torch.randn(batch_size, in_channels, height, width)
    x_vit = torch.randn(batch_size, in_channels, height, width)
    out = fusion(x_mamba, x_vit)
    print(f"  Mamba分支输入: {x_mamba.shape}")
    print(f"  ViT分支输入: {x_vit.shape}")
    print(f"  融合输出: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion.parameters()):,}")

    # 测试双分支模块A
    print("\n[2] 测试DualBranchModuleA...")
    dual_branch = DualBranchModuleA(
        channels=in_channels,
        input_resolution=(height, width),
        num_heads=4,
        window_size=7
    )
    out = dual_branch(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in dual_branch.parameters()):,}")

    # 测试编码器阶段
    print("\n[3] 测试EncoderStage...")
    encoder_stage = EncoderStage(
        in_channels=in_channels,
        out_channels=in_channels * 2,
        input_resolution=(height, width),
        downsample=True,
        num_heads=4
    )
    out = encoder_stage(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in encoder_stage.parameters()):,}")

    # 测试完整编码器
    print("\n[4] 测试完整Encoder...")
    encoder = Encoder(
        in_channels=3,
        base_channels=32,
        input_size=512,
        num_heads_list=[2, 2, 4, 4]
    )

    # 输入512x512图像
    x_img = torch.randn(batch_size, 3, 512, 512)
    features = encoder(x_img)

    print(f"  输入图像: {x_img.shape}")
    for i, f in enumerate(features):
        print(f"  阶段{i + 1}输出: {f.shape}")

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  编码器总参数量: {total_params:,}")

    # 测试梯度
    print("\n[5] 测试梯度反向传播...")
    loss = sum(f.sum() for f in features)
    loss.backward()
    print("  梯度计算成功")

    # 测试不同输入尺寸
    print("\n[6] 测试不同输入尺寸...")
    for size in [256, 384, 512]:
        encoder_test = Encoder(
            in_channels=3,
            base_channels=32,
            input_size=size
        )
        x_test = torch.randn(1, 3, size, size)
        features = encoder_test(x_test)
        print(f"  输入尺寸 {size}x{size}:")
        for i, f in enumerate(features):
            print(f"    阶段{i + 1}: {f.shape}")

    print("\n" + "=" * 60)
    print("所有双分支融合模块和编码器测试通过！")
    print("=" * 60)
