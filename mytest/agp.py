import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

# 引入基础轻量化算子 (参考 modules/base_modules.py)
from modules.base_modules import PConvBlock, UpSample, ConvBNReLU, DepthwiseSeparableConv
from modules.SS2D_quard_parallel import SS2D_SASS_Quarter

# 尝试导入官方 mamba_ssm 算子，若不可用则需使用简化版
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class AnisotropicGatedSASS(SS2D_SASS_Quarter):
    """
    AGM 核心组件：带方向门控的 SASS
    通过 Softmax 归一化的权重，动态调整四个扫描方向的重要性。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 引入可学习的方向权重 gamma (对应 4 个扫描方向)
        self.gamma = nn.Parameter(torch.ones(4))

    def forward(self, x):
        batch_size, _, H, W = x.shape
        L = H * W
        hw_shape = (H, W)
        E = self.d_inner

        x = x.flatten(2).transpose(1, 2)
        xz = self.in_proj(x)
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=-1)

        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in
                        direction_Bs]

        # 核心改进：计算各向异性权重
        weights = F.softmax(self.gamma, dim=0)

        # 四方向扫描并应用权重
        y_scan = []
        for i, (o, inv_order, dB) in enumerate(zip(orders, inverse_orders, direction_Bs)):
            y_d = selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt, A, (B + dB).contiguous(), C, self.D,
                z=None, delta_bias=self.dt_proj.bias, delta_softplus=True
            ).permute(0, 2, 1)[:, inv_order, :]
            y_scan.append(y_d * weights[i])  # 应用各向异性门控权重

        y = sum(y_scan) * self.act(z)
        out = self.out_proj(y)
        return out.transpose(1, 2).reshape(batch_size, self.d_model, H, W)


class AGM_Module(nn.Module):
    """
    AGM 模块封装：包含归一化、四路并行各向异性 Mamba 及残差连接。
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 使用四路并行结构以进一步降低参数量
        self.ss2d = nn.ModuleList([AnisotropicGatedSASS(d_model=dim, d_state=d_state, expand=expand) for _ in range(1)])
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x

        # 归一化
        x_n = self.norm(x.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)

        # 为了极简化实现，这里直接调用改进后的 SASS
        # 实际 AGM 会将通道拆分为 4 份并行处理
        x_splits = torch.chunk(x_n, 4, dim=1)
        y_splits = [self.ss2d[0](split) for split in x_splits]
        y = torch.cat(y_splits, dim=1)

        return self.proj(y) + identity


class LightFusionBlock(nn.Module):
    """轻量级跳跃连接融合"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            PConvBlock(out_channels)  # 使用 PConv 细化
        )

    def forward(self, x_up, x_skip):
        x = torch.cat([x_up, x_skip], dim=1)
        return self.conv(x)


class CrackSegmentationNetV2(nn.Module):
    """
    重构后的各向异性门控裂缝分割网络 (AGM-Net)
    1. Stage 1-2 使用 PConvBlock (CNN-only)
    2. Stage 3-4 使用 AGM (Mamba-based)
    3. 参数量更轻，性能更强
    """

    def __init__(self, in_channels=3, base_channels=96, input_size=512):
        super().__init__()

        # Encoder 通道数序列
        chs = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, chs[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(chs[0]),
            nn.ReLU(inplace=True)
        )

        # Stage 1-2: 纯 CNN 提取基础特征
        self.encoder1 = PConvBlock(chs[0])
        self.down1 = DepthwiseSeparableConv(chs[0], chs[1], stride=2)
        self.encoder2 = PConvBlock(chs[1])

        # Stage 3-4: AGM Mamba 捕获拓扑依赖
        self.down2 = DepthwiseSeparableConv(chs[1], chs[2], stride=2)
        self.encoder3 = AGM_Module(chs[2])
        self.down3 = DepthwiseSeparableConv(chs[2], chs[3], stride=2)
        self.encoder4 = AGM_Module(chs[3])

        # Decoder (无 ASPP，轻量化设计)
        self.up4 = UpSample(2)
        self.fuse3 = LightFusionBlock(chs[3] + chs[2], chs[2])
        self.up3 = UpSample(2)
        self.fuse2 = LightFusionBlock(chs[2] + chs[1], chs[1])
        self.up2 = UpSample(2)
        self.fuse1 = LightFusionBlock(chs[1] + chs[0], chs[0])
        self.up1 = UpSample(2)

        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(chs[0], 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.patch_embed(x)  # 1/2
        f1 = self.encoder1(x)  # 256x256
        f2 = self.encoder2(self.down1(f1))  # 128x128
        f3 = self.encoder3(self.down2(f2))  # 64x64
        f4 = self.encoder4(self.down3(f3))  # 32x32

        # Decoder
        x = self.up4(f4)
        x = self.fuse3(x, f3)
        x = self.up3(x)
        x = self.fuse2(x, f2)
        x = self.up2(x)
        x = self.fuse1(x, f1)
        x = self.up1(x)

        return self.seg_head(x)


if __name__ == "__main__":
    # 测试模型
    model = CrackSegmentationNetV2(base_channels=96)
    x = torch.randn(1, 3, 512, 512)
    out = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"AGM-Net 总参数量: {total_params / 1e6:.2f}M")