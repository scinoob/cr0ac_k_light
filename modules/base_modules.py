"""
基础模块实现：PConv、BottConv、GBC
包含部分卷积、瓶颈卷积和门控瓶颈结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from timm.layers import DropPath


class BottConv(nn.Module):
    """
    瓶颈卷积模块 (Bottleneck Convolution)
    
    将传统卷积分解为：
    1. 逐点卷积 (Pointwise Convolution) - 1x1卷积
    2. 深度卷积 (Depthwise Convolution) - 空间特征提取
    3. 逐点卷积 (Pointwise Convolution) - 通道混合
    
    参数量对比：
    传统卷积参数量 = K1 * K2 * C_in * C_out
    深度卷积参数量 = K1 * K2 * C_mid + C_mid * C_out + C_in * C_mid
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        mid_channels: 中间通道数（瓶颈维度）
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        bias: 是否使用偏置
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True
    ):
        super().__init__()

        # 逐点卷积1：通道扩展/压缩
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)

        # 深度卷积：对每个通道分别应用卷积，仅捕获空间特征
        self.depthwise = nn.Conv2d(
            mid_channels, mid_channels, kernel_size,
            stride=stride, padding=padding, groups=mid_channels, bias=False
        )

        # 逐点卷积2：通道混合
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量，形状为 (B, C_in, H, W)
            
        返回：
            输出张量，形状为 (B, C_out, H', W')
        """
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class PConv(nn.Module):
    """
    部分卷积模块 (Partial Convolution)
    
    来自FasterNet，仅对部分输入通道进行常规卷积，
    其余通道保持不变，大幅减少计算量。
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        partial_ratio: 部分卷积比例，默认0.25
        kernel_size: 卷积核大小，默认3
        stride: 步长，默认1
        padding: 填充，默认1
        bias: 是否使用偏置，默认False
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            partial_ratio: float = 0.25,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.partial_channels = max(1, int(in_channels * partial_ratio))

        # 仅对部分通道进行卷积
        self.conv = nn.Conv2d(
            self.partial_channels, self.partial_channels,
            kernel_size, stride=stride, padding=padding, bias=bias
        )

        # 记录不变通道数
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 如果输入输出通道不同，需要添加投影层
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量，形状为 (B, C, H, W)
            
        返回：
            输出张量，形状为 (B, C_out, H, W)
        """
        # 分割为卷积部分和不变部分
        x_conv = x[:, :self.partial_channels, :, :]
        x_identity = x[:, self.partial_channels:, :, :]

        # 对部分通道进行卷积
        x_conv = self.conv(x_conv)

        # 拼接回原张量
        x_out = torch.cat([x_conv, x_identity], dim=1)

        # 如果需要通道投影
        if self.proj is not None:
            x_out = self.proj(x_out)

        return x_out


class PWConv(nn.Module):
    """
    逐点卷积模块 (Pointwise Convolution)
    
    使用1x1卷积进行通道混合
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        bias: 是否使用偏置
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PConvBlock(nn.Module):
    """
    PConv块：PConv + PWConv + PWConv
    
    结构：
    1. PConv - 部分卷积处理
    2. PWConv + GELU - 通道混合与激活
    3. PWConv - 通道恢复
    
    参数：
        in_channels: 输入通道数
        expand_ratio: 扩展比例，默认2
        partial_ratio: PConv的部分卷积比例
    """

    def __init__(
            self,
            in_channels: int,
            expand_ratio: float = 2.0,
            partial_ratio: float = 0.25,
            drop_path_rate: float = 0.0,
    ):
        super().__init__()

        mid_channels = int(in_channels * expand_ratio)

        self.pconv = PConv(in_channels, in_channels, partial_ratio)
        self.pwconv1 = PWConv(in_channels, mid_channels)
        self.pwconv2 = PWConv(mid_channels, in_channels)
        self.act = nn.GELU()
        # 新增DropPath层
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，带残差连接
        
        参数：
            x: 输入张量
            
        返回：
            残差连接后的输出张量
        """
        identity = x
        x = self.pconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # 将drop_path作用于残差分支
        return self.drop_path(x) + identity


def get_norm_layer(norm_type: str, channels: int, num_groups: int = None) -> nn.Module:
    """
    获取归一化层
    
    参数：
        norm_type: 归一化类型，'GN' (GroupNorm) 或 'IN' (InstanceNorm)
        channels: 通道数
        num_groups: GroupNorm的组数
        
    返回：
        归一化层
    """
    if norm_type == 'GN':
        num_groups = num_groups or min(32, channels // 4)
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    elif norm_type == 'BN':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


class GBC(nn.Module):
    """
    门控瓶颈结构模块 (Gated Bottleneck Convolution)
    
    结构：
    X -> BottConv1 -> BottConv2 -> X1
    X -> BottConv3 -> X2
    X1 * X2 -> BottConv4 -> Output + X
    
    通过门控机制增强特征表达能力
    
    参数：
        in_channels: 输入通道数
        norm_type: 归一化类型
    """

    def __init__(self, in_channels: int, norm_type: str = 'GN'):
        super().__init__()

        mid_channels = in_channels // 8

        # 第一个分支：两个连续的瓶颈卷积
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(inplace=True)
        )

        # 第二个分支：单个瓶颈卷积（门控分支）
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(inplace=True)
        )

        # 输出卷积
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量
            
        返回：
            门控融合后的输出张量（带残差连接）
        """
        identity = x

        x1 = self.block1(x)
        x1 = self.block2(x1)

        x2 = self.block3(x)

        # 门控融合
        x = x1 * x2
        x = self.block4(x)

        return x + identity


class DownSample(nn.Module):
    """
    下采样模块
    
    使用3x3卷积，步长为2进行下采样
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpSample(nn.Module):
    """
    上采样模块
    
    使用双线性插值进行上采样
    
    参数：
        scale_factor: 上采样因子
        mode: 插值模式
        align_corners: 是否对齐角点
    """

    def __init__(self, scale_factor: int = 2, mode: str = 'bilinear', align_corners: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners
        )


class ConvBNReLU(nn.Module):
    """
    卷积 + BN + ReLU 组合模块
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        bias: 是否使用偏置
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试基础模块")
    print("=" * 60)

    # 测试参数
    batch_size = 2
    in_channels = 64
    height, width = 32, 32

    # 创建测试输入
    x = torch.randn(batch_size, in_channels, height, width)

    # 测试 BottConv
    print("\n[1] 测试 BottConv...")
    bottconv = BottConv(in_channels, in_channels * 2, in_channels // 4, 3, 1, 1)
    out = bottconv(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in bottconv.parameters()):,}")

    # 测试 PConv
    print("\n[2] 测试 PConv...")
    pconv = PConv(in_channels, partial_ratio=0.25)
    out = pconv(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in pconv.parameters()):,}")

    # 测试 PConvBlock
    print("\n[3] 测试 PConvBlock...")
    pconv_block = PConvBlock(in_channels)
    out = pconv_block(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in pconv_block.parameters()):,}")

    # 测试 GBC
    print("\n[4] 测试 GBC...")
    gbc = GBC(in_channels)
    out = gbc(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in gbc.parameters()):,}")

    # 测试 DownSample
    print("\n[5] 测试 DownSample...")
    downsample = DownSample(in_channels, in_channels * 2)
    out = downsample(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")

    # 测试 UpSample
    print("\n[6] 测试 UpSample...")
    upsample = UpSample(scale_factor=2)
    out = upsample(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")

    # 测试 ConvBNReLU
    print("\n[7] 测试 ConvBNReLU...")
    conv_bn_relu = ConvBNReLU(in_channels, in_channels * 2)
    out = conv_bn_relu(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")

    print("\n" + "=" * 60)
    print("所有基础模块测试通过！")
    print("=" * 60)
