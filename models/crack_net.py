"""
解码器和完整网络模型实现
包含轻量融合块、解码器和完整的裂缝分割网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict

from modules.base_modules import DepthwiseSeparableConv, PConv, PConvBlock, UpSample, ConvBNReLU, GBC
from modules.dual_branch import DualBranchModuleA


class AttentionGate(nn.Module):
    """
    注意力门控机制 (Attention Gate)
    利用深层特征 (g) 生成空间权重，过滤浅层特征 (x) 中的背景噪声。
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: 深层上采样特征; x: 浅层跳跃连接特征
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class LightFusionBlock(nn.Module):
    """
    轻量融合块
    
    用于解码器中融合上采样特征和跳跃连接特征
    
    流程：
    1. 拼接上采样特征和跳跃连接特征
    2. 1x1卷积降维
    3. BN + ReLU
    4. 可选PConv细化
    
    参数：
        in_channels: 输入通道数（拼接后的通道数）
        out_channels: 输出通道数
        use_pconv: 是否使用PConv细化
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_pconv: bool = True
    ):
        super().__init__()

        self.use_pconv = use_pconv

        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 可选PConv细化
        if use_pconv:
            self.pconv = PConvBlock(out_channels)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x_up: 上采样特征 (B, C_up, H, W)
            x_skip: 跳跃连接特征 (B, C_skip, H, W)
            
        返回：
            融合特征 (B, C_out, H, W)
        """
        # 拼接
        x = torch.cat([x_up, x_skip], dim=1)

        # 降维
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        # 可选细化
        if self.use_pconv:
            x = self.pconv(x)

        return x


class DecoderStage(nn.Module):
    """
    解码器阶段
    
    包含上采样、跳跃连接融合和可选的特征细化
    
    参数：
        in_channels: 输入通道数
        skip_channels: 跳跃连接通道数
        out_channels: 输出通道数
        use_pconv: 是否使用PConv细化
    """

    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            use_pconv: bool = True
    ):
        super().__init__()

        # 上采样
        self.upsample = UpSample(scale_factor=2, mode='bilinear', align_corners=True)

        # 融合块
        self.fusion = LightFusionBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            use_pconv=use_pconv
        )

    def forward(
            self,
            x: torch.Tensor,
            skip: torch.Tensor,
            target_size: Tuple[int, int] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入特征 (B, C_in, H, W)
            skip: 跳跃连接特征 (B, C_skip, H*2, W*2)
            target_size: 目标尺寸（可选，用于处理尺寸不匹配）
            
        返回：
            输出特征 (B, C_out, H*2, W*2)
        """
        # 上采样
        x = self.upsample(x)

        # 尺寸对齐（如果需要）
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

        # 确保尺寸匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        # 融合
        x = self.fusion(x, skip)

        return x


class Bottleneck(nn.Module):
    """
    瓶颈层
    
    使用双分支模块处理最深层特征
    
    参数：
        channels: 通道数
        input_resolution: 输入分辨率
        use_aspp: 是否使用ASPP模块
        **kwargs: 双分支模块参数
    """

    def __init__(
            self,
            channels: int,
            input_resolution: Tuple[int, int],
            use_aspp: bool = False,
            **kwargs
    ):
        super().__init__()

        self.use_aspp = use_aspp

        # 双分支模块
        self.dual_branch = DualBranchModuleA(
            channels=channels,
            input_resolution=input_resolution,
            **kwargs
        )

        # 可选ASPP
        if use_aspp:
            self.aspp = ASPP(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dual_branch(x)
        if self.use_aspp:
            x = self.aspp(x)
        return x


class ASPP(nn.Module):
    """
    空洞空间金字塔池化模块 (Atrous Spatial Pyramid Pooling)
    
    捕获多尺度上下文信息
    
    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        dilations: 空洞率列表
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dilations: List[int] = [1, 6, 12, 18]
    ):
        super().__init__()

        # 各分支
        self.branches = nn.ModuleList()

        # 1x1卷积分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        ))

        # 空洞卷积分支
        for d in dilations[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ))

        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        # @kimi 修复: 计算正确的融合层输入通道数
        # 原代码: nn.Conv2d(out_channels, out_channels, ...)
        # 理由: 有len(dilations)个卷积分支 + 1个全局池化分支 = len(dilations)+1个分支
        # 每个分支输出out_channels//4通道，所以拼接后总通道数为(len(dilations)+1) * (out_channels//4)
        fusion_in_channels = (len(dilations) + 1) * (out_channels // 4)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]

        # 各分支输出
        features = []
        for branch in self.branches:
            features.append(branch(x))

        # 全局池化分支
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)
        features.append(global_feat)

        # 拼接融合
        x = torch.cat(features, dim=1)
        x = self.fusion(x)

        return x


#  scSE 模块对编码器特征进行空间和通道加权，让解码器更关注细节区域。
class SCSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Decoder(nn.Module):
    """
    完整解码器
    
    包含瓶颈层和4个解码器阶段
    
    参数：
        encoder_channels: 编码器各阶段通道数列表
        decoder_channels: 解码器各阶段通道数列表
        input_size: 输入图像尺寸
        use_aspp: 是否在瓶颈层使用ASPP
        **kwargs: 双分支模块参数
    """

    def __init__(
            self,
            encoder_channels: List[int],
            decoder_channels: List[int],
            input_size: int = 512,
            use_aspp: bool = False,
            **kwargs
    ):
        super().__init__()

        self.input_size = input_size

        # 瓶颈层
        bottleneck_resolution = (input_size // 16, input_size // 16)
        self.bottleneck = Bottleneck(
            channels=encoder_channels[-1],
            input_resolution=bottleneck_resolution,
            use_aspp=use_aspp,
            **kwargs
        )

        # 解码器阶段
        # 阶段4: 256 -> 128, 32x32 -> 64x64
        self.stage4 = DecoderStage(
            in_channels=encoder_channels[3],
            skip_channels=encoder_channels[2],
            out_channels=decoder_channels[3]
        )

        # 阶段3: 128 -> 64, 64x64 -> 128x128
        self.stage3 = DecoderStage(
            in_channels=decoder_channels[3],
            skip_channels=encoder_channels[1],
            out_channels=decoder_channels[2]
        )

        # 阶段2: 64 -> 32, 128x128 -> 256x256
        self.stage2 = DecoderStage(
            in_channels=decoder_channels[2],
            skip_channels=encoder_channels[0],
            out_channels=decoder_channels[1]
        )

        # 阶段1: 32 -> 16, 256x256 -> 512x512
        self.upsample1 = UpSample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine1 = nn.Sequential(
            PConv(decoder_channels[1], decoder_channels[0]),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            PConv(decoder_channels[0], decoder_channels[0]),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.decoder_channels = decoder_channels

        # 残差连接，引入SCSE
        # self.res4 = SCSE(decoder_channels[3])
        # self.res3 = SCSE(decoder_channels[2])
        # self.res2 = SCSE(decoder_channels[1])

    def forward(
            self,
            encoder_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        参数：
            encoder_features: 编码器各阶段特征 [f1, f2, f3, f4]
            
        返回：
            解码器输出 (B, C, H, W)
        """
        f1, f2, f3, f4 = encoder_features

        # 瓶颈层
        x = self.bottleneck(f4)

        # 解码器阶段
        x = self.stage4(x, f3)
        x = self.stage3(x, f2)
        x = self.stage2(x, f1)

        # 加入scse
        # x = self.stage4(x, self.res4(f3))
        # x = self.stage3(x, self.res3(f2))
        # x = self.stage2(x, self.res2(f1))

        # 最后上采样
        x = self.upsample1(x)
        x = self.refine1(x)

        return x


class CrackSegmentationNet(nn.Module):
    """
    裂缝分割网络
    
    基于Mamba与ViT双分支的非对称UNet架构
    
    参数：
        in_channels: 输入通道数（默认3）
        num_classes: 输出类别数（默认1，二分类）
        base_channels: 基础通道数（默认32）
        input_size: 输入图像尺寸（默认512）
        d_state: Mamba状态维度
        d_conv: Mamba卷积核大小
        expand: Mamba扩展因子
        num_heads_list: 各阶段ViT注意力头数
        window_size: ViT窗口大小
        use_gbc: 是否使用GBC模块
        use_aspp: 是否使用ASPP模块
        drop_path_rate: 最大随机深度丢弃概率 (新增，默认0.0)
    """

    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 1,
            base_channels: int = 32,
            input_size: int = 512,
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            num_heads_list: List[int] = [2, 2, 4, 4],
            window_size: int = 7,
            use_gbc: bool = False,
            use_aspp: bool = False,
            drop_path_rate: float = 0.0,  # @kimi 新增: 最大随机深度丢弃概率
            use_edge_supervision: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.input_size = input_size
        self.drop_path_rate = drop_path_rate  # @kimi 新增: 保存drop_path_rate

        # 编码器通道数
        encoder_channels = [
            base_channels,  # 32
            base_channels * 2,  # 64
            base_channels * 4,  # 128
            base_channels * 8  # 256
        ]

        # 解码器通道数
        decoder_channels = [
            base_channels // 2,  # 16
            base_channels,  # 32
            base_channels * 2,  # 64
            base_channels * 4  # 128
        ]

        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        # 加入边缘监督分支
        # self.use_edge_supervision = use_edge_supervision
        # if use_edge_supervision:
        #     # 选择从解码器 stage2 引出（128×128，通道64），也可选 stage1（256×256，通道32）
        #     self.edge_head = nn.Sequential(
        #         nn.Conv2d(encoder_channels[1], decoder_channels[1], 3, padding=1),  # 输入为解码器 stage2 的通道数
        #         nn.BatchNorm2d(decoder_channels[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(decoder_channels[1], 1, 1),
        #         nn.Sigmoid()
        #     )

        # 编码器各阶段
        from modules.dual_branch import EncoderStage

        resolutions = [
            (input_size // 2, input_size // 2),
            (input_size // 4, input_size // 4),
            (input_size // 8, input_size // 8),
            (input_size // 16, input_size // 16)
        ]

        # @kimi 新增: 计算各阶段的drop_path概率（线性调度）
        # 理由: 共有4个编码器阶段，使用np.linspace从0到drop_path_rate生成调度
        import numpy as np
        num_stages = 4
        dpr = np.linspace(0, drop_path_rate, num_stages).tolist()

        # 阶段1：无下采样
        self.encoder1 = EncoderStage(
            in_channels=encoder_channels[0],
            out_channels=encoder_channels[0],
            input_resolution=resolutions[0],
            downsample=False,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[0],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=dpr[0],  # @kimi 新增
            drop_path_vit=dpr[0]  # @kimi 新增
        )

        # 阶段2-4（深度增加，drop_path概率增大）
        self.encoder2 = EncoderStage(
            in_channels=encoder_channels[0],
            out_channels=encoder_channels[1],
            input_resolution=resolutions[0],
            downsample=True,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[1],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=dpr[1],  # @kimi 新增
            drop_path_vit=dpr[1]  # @kimi 新增
        )

        self.encoder3 = EncoderStage(
            in_channels=encoder_channels[1],
            out_channels=encoder_channels[2],
            input_resolution=resolutions[1],
            downsample=True,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[2],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=dpr[2],  # @kimi 新增
            drop_path_vit=dpr[2]  # @kimi 新增
        )

        self.encoder4 = EncoderStage(
            in_channels=encoder_channels[2],
            out_channels=encoder_channels[3],
            input_resolution=resolutions[2],
            downsample=True,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[3],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=dpr[3],  # @kimi 新增
            drop_path_vit=dpr[3]  # @kimi 新增
        )

        # @kimi 新增: 瓶颈层使用最大的drop_path概率（最深位置）
        # 理由: 瓶颈层位于编码器最深层，应使用最大的drop_path_rate
        bottleneck_drop_path = dpr[-1]

        # 解码器
        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            input_size=input_size,
            use_aspp=use_aspp,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads_list[3],
            window_size=window_size,
            use_gbc=use_gbc,
            drop_path_mamba=bottleneck_drop_path,  # @kimi 新增: 瓶颈层Mamba分支drop_path
            drop_path_vit=bottleneck_drop_path  # @kimi 新增: 瓶颈层ViT分支drop_path
        )

        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[0], num_classes, 1),
            nn.Sigmoid()
        )

        # 保存中间特征用于Grad-CAM
        self.intermediate_features = {}

        # 保存通道数信息
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False,
            # return_edges: bool = False,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入图像 (B, 3, H, W)
            return_features: 是否返回中间特征（用于Grad-CAM）
            return_edges: 是否返回边缘
            
        返回：
            分割输出 (B, 1, H, W) 或 (分割输出, 中间特征字典)
        """
        # Patch Embedding
        x = self.patch_embed(x)

        # 编码器
        f1 = self.encoder1(x)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = self.encoder4(f3)

        # 保存中间特征
        if return_features:
            self.intermediate_features = {
                'encoder1': f1,
                'encoder2': f2,
                'encoder3': f3,
                'encoder4': f4
            }

        # 解码器
        decoder_out = self.decoder([f1, f2, f3, f4])

        # 保存解码器特征
        if return_features:
            self.intermediate_features['decoder_final'] = decoder_out

        # 分割头
        out = self.seg_head(decoder_out)

        # if self.training and self.use_edge_supervision and return_edges:
        #     # 从 d2 计算边缘预测
        #     edge_pred = self.edge_head(self.decoder.stage3)  # (B,1,128,128)
        #     return out, edge_pred
        if return_features:
            return out, self.intermediate_features

        return out

    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        获取编码器各阶段特征（用于Grad-CAM）
        
        参数：
            x: 输入图像
            
        返回：
            编码器特征列表
        """
        x = self.patch_embed(x)
        f1 = self.encoder1(x)
        f2 = self.encoder2(f1)
        f3 = self.encoder3(f2)
        f4 = self.encoder4(f3)
        return [f1, f2, f3, f4]


# edited by gemini
# --- 在 models/crack_net.py 中添加 ---
from modules.SS2D_quard_parallel import AGM_Module  # 确保导入新定义的 AGM

class MaxPoolDown(nn.Module):
    """
    基于最大池化的下采样模块
    利用 MaxPool 保留区域内的最大激活值（微弱裂缝的峰值特征），
    防止微小目标在跨步卷积（stride=2）中被背景平滑掉导致特征消失。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 使用 MaxPool 进行严格的空间下采样，保留最强信号
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2. 池化后接 3x3 卷积，用于调整通道数并进行局部上下文特征重组
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(self.pool(x))

class AGM_LightFusion(nn.Module):
    """
    [配套模块] 极简融合块
    替代原有的 LightFusionBlock，专注于低参数量的跳跃连接合并。
    """

    def __init__(self, up_channels, skip_channels, out_channels, use_gbc: bool = True):
        super().__init__()
        # 引入注意力门控，中间通道数设为浅层通道数的一半，极大地控制参数量
        self.ag = AttentionGate(F_g=up_channels, F_l=skip_channels, F_int=skip_channels // 2)

        in_channels = skip_channels + up_channels

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 【关键手术 2】解码器细化：开启 GBC 时使用 GBC 提纯，否则用 PConv
        self.refine = GBC(out_channels) if use_gbc else PConvBlock(out_channels)

    def forward(self, x_up, x_skip):
        # 1. 用深层特征 x_up 过滤浅层特征 x_skip
        x_skip_ag = self.ag(g=x_up, x=x_skip)
        # return self.fuse(torch.cat([x_up, x_skip], dim=1))

        # 2. 拼接过滤后的特征与深沉特征
        x = torch.cat([x_up, x_skip_ag], dim=1)

        # 3. 降维并细化
        x = self.reduce(x)
        return self.refine(x)


class CrackSegmentationNetV2(nn.Module):
    """
        [新增网络] 裂缝分割网络 V2
        特点：
        1. 非对称设计：浅层(Stage 1-2)纯 CNN，深层(Stage 3-4) AGM Mamba。
        2. 无 ASPP：减少计算开销，通过 AGM 捕获长程依赖。
        3. 各向异性权重：强化细长裂缝的连通性。
    """

    def __init__(self, in_channels=3, num_classes=1, base_channels=96, input_size=224, use_gbc=True,
                 drop_path_rate: float = 0.0, **kwargs):
        super().__init__()

        chs = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        # 计算各阶段的 drop_path 概率（线性调度，随深度增加）
        dpr = np.linspace(0, drop_path_rate, 4).tolist()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, chs[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(chs[0]),
            nn.ReLU(inplace=True)
        )

        # 【关键手术 3】浅层编码器注入 GBC
        # 在高分辨率下，GBC 能有效剥离水渍和阴影，为主干减负
        self.stage1_gbc = GBC(chs[0]) if use_gbc else nn.Identity()
        self.stage1 = PConvBlock(chs[0], drop_path_rate=dpr[0])
        # self.down1 = DepthwiseSeparableConv(chs[0], chs[1], stride=2)
        # 替换为池化提取特征
        self.down1 = MaxPoolDown(chs[0], chs[1])

        self.stage2_gbc = GBC(chs[1]) if use_gbc else nn.Identity()
        self.stage2 = PConvBlock(chs[1], drop_path_rate=dpr[1])
        # self.down2 = DepthwiseSeparableConv(chs[1], chs[2], stride=2)
        self.down2 = MaxPoolDown(chs[1], chs[2])

        # 深层保持 AGM Mamba，提取拓扑连通性
        self.stage3 = AGM_Module(chs[2], drop_path_rate=dpr[2])
        self.down3 = DepthwiseSeparableConv(chs[2], chs[3], stride=2)
        self.stage4 = AGM_Module(chs[3], drop_path_rate=dpr[3])

        # 解码器：传入 use_gbc 进行高保真重建
        # self.up4 = UpSample(2)
        # self.fuse3 = AGM_LightFusion(chs[3] + chs[2], chs[2], use_gbc)
        # self.up3 = UpSample(2)
        # self.fuse2 = AGM_LightFusion(chs[2] + chs[1], chs[1], use_gbc)
        # self.up2 = UpSample(2)
        # self.fuse1 = AGM_LightFusion(chs[1] + chs[0], chs[0], use_gbc)
        # self.up1 = UpSample(2)

        # 解码器：传入 use_gbc 进行高保真重建 (修改为分别传入 up_channels 和 skip_channels)
        self.up4 = UpSample(2)
        self.fuse3 = AGM_LightFusion(up_channels=chs[3], skip_channels=chs[2], out_channels=chs[2], use_gbc=use_gbc)
        self.up3 = UpSample(2)
        self.fuse2 = AGM_LightFusion(up_channels=chs[2], skip_channels=chs[1], out_channels=chs[1], use_gbc=use_gbc)
        self.up2 = UpSample(2)
        self.fuse1 = AGM_LightFusion(up_channels=chs[1], skip_channels=chs[0], out_channels=chs[0], use_gbc=use_gbc)
        self.up1 = UpSample(2)

        self.seg_head = nn.Sequential(
            nn.Conv2d(chs[0], num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        p = self.patch_embed(x)

        f1 = self.stage1(self.stage1_gbc(p))
        f2 = self.stage2(self.stage2_gbc(self.down1(f1)))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(self.down3(f3))

        x = self.fuse3(self.up4(f4), f3)
        x = self.fuse2(self.up3(x), f2)
        x = self.fuse1(self.up2(x), f1)
        x = self.up1(x)

        return self.seg_head(x)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试解码器和完整网络模型")
    print("=" * 60)

    # 测试参数
    batch_size = 2
    input_size = 512

    # 测试LightFusionBlock
    print("\n[1] 测试LightFusionBlock...")
    fusion = LightFusionBlock(in_channels=128, out_channels=64)
    x_up = torch.randn(batch_size, 64, 32, 32)
    x_skip = torch.randn(batch_size, 64, 32, 32)
    out = fusion(x_up, x_skip)
    print(f"  上采样特征: {x_up.shape}")
    print(f"  跳跃连接特征: {x_skip.shape}")
    print(f"  融合输出: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in fusion.parameters()):,}")

    # 测试DecoderStage
    print("\n[2] 测试DecoderStage...")
    decoder_stage = DecoderStage(in_channels=256, skip_channels=128, out_channels=128)
    x = torch.randn(batch_size, 256, 16, 16)
    skip = torch.randn(batch_size, 128, 32, 32)
    out = decoder_stage(x, skip)
    print(f"  输入特征: {x.shape}")
    print(f"  跳跃连接: {skip.shape}")
    print(f"  输出特征: {out.shape}")

    # 测试ASPP
    print("\n[3] 测试ASPP...")
    aspp = ASPP(256, 256)
    x = torch.randn(batch_size, 256, 32, 32)
    out = aspp(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in aspp.parameters()):,}")

    # 测试完整网络
    print("\n[4] 测试CrackSegmentationNet...")
    model = CrackSegmentationNet(
        in_channels=3,
        num_classes=1,
        base_channels=32,
        input_size=input_size
    )

    # 输入图像
    x = torch.randn(batch_size, 3, input_size, input_size)

    # 前向传播
    out = model(x)
    print(f"  输入图像: {x.shape}")
    print(f"  分割输出: {out.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 测试返回中间特征
    print("\n[5] 测试返回中间特征...")
    out, features = model(x, return_features=True)
    print(f"  分割输出: {out.shape}")
    print("  中间特征:")
    for name, feat in features.items():
        print(f"    {name}: {feat.shape}")

    # 测试梯度
    print("\n[6] 测试梯度反向传播...")
    loss = out.sum()
    loss.backward()
    print("  梯度计算成功")

    # 测试不同输入尺寸
    print("\n[7] 测试不同输入尺寸...")
    for size in [256, 384]:
        model_test = CrackSegmentationNet(
            in_channels=3,
            base_channels=32,
            input_size=size
        )
        x_test = torch.randn(1, 3, size, size)
        out = model_test(x_test)
        print(f"  输入 {size}x{size} -> 输出 {out.shape}")

    print("\n" + "=" * 60)
    print("所有解码器和网络模型测试通过！")
    print("=" * 60)
