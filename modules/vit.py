"""
ViT分支实现：轻量级Swin Transformer
包含窗口多头自注意力(W-MSA)和移位窗口(SW-MSA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

# @kimi 新增: 导入DropPath用于随机深度正则化
# 理由: 实现深度网络的随机深度(Stochastic Depth)机制，提高泛化能力
try:
    from timm.layers import DropPath
except ImportError:
    # 如果timm不可用，使用简化实现
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample."""
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output


class WindowAttention(nn.Module):
    """
    窗口多头自注意力模块 (Window Multi-Head Self Attention)
    
    在局部窗口内计算自注意力，复杂度与图像尺寸线性相关
    
    参数：
        dim: 输入维度
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        attn_drop: 注意力dropout
        proj_drop: 投影dropout
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 计算相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # 偏移到非负
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量 (B*num_windows, Wh*Ww, C)
            mask: 注意力掩码 (num_windows, Wh*Ww, Wh*Ww) 或 None
            
        返回：
            输出张量 (B*num_windows, Wh*Ww, C)
        """
        B_, N, C = x.shape
        
        # QKV投影
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 (B_, num_heads, N, head_dim)
        
        # 注意力计算
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )  # (Wh*Ww, Wh*Ww, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # 应用掩码（用于移位窗口）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图划分为窗口
    
    参数：
        x: 输入张量 (B, H, W, C)
        window_size: 窗口大小
        
    返回：
        窗口张量 (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # 如果 H 或 W 不能被 window_size 整除，使用 cyclic padding
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode='circular')
    
    H_padded, W_padded = x.shape[1], x.shape[2]
    x = x.view(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (H, W), (H_padded, W_padded)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, orig_HW: tuple = None) -> torch.Tensor:
    """
    将窗口合并回特征图
    
    参数：
        windows: 窗口张量 (B*num_windows, window_size, window_size, C)
        window_size: 窗口大小
        H: 高度（padded 后的高度）
        W: 宽度（padded 后的宽度）
        orig_HW: 原始 (H, W)，用于裁剪回原始尺寸
        
    返回：
        特征图 (B, H, W, C) 或 (B, orig_H, orig_W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    
    # 如果需要，裁剪回原始尺寸
    if orig_HW is not None:
        orig_H, orig_W = orig_HW
        if H > orig_H or W > orig_W:
            x = x[:, :orig_H, :orig_W, :].contiguous()
    
    return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer块
    
    包含：
    1. 窗口多头自注意力 (W-MSA) 或 移位窗口多头自注意力 (SW-MSA)
    2. MLP层
    3. 层归一化
    4. 残差连接
    
    参数：
        dim: 输入维度
        input_resolution: 输入分辨率 (H, W)
        num_heads: 注意力头数
        window_size: 窗口大小
        shift_size: 移位大小（0表示W-MSA，>0表示SW-MSA）
        mlp_ratio: MLP扩展比例
        qkv_bias: 是否使用偏置
        drop: dropout率
        attn_drop: 注意力dropout率
        drop_path: 随机深度丢弃概率 (新增)
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 4,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,  # @kimi 新增: 随机深度丢弃概率
    ):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 如果输入尺寸小于窗口大小，则退化为全局自注意力
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # @kimi 新增: DropPath层用于随机深度
        # 理由: 在残差连接上应用随机深度，以一定概率丢弃整个块的变换
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # 计算注意力掩码（用于移位窗口）
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows, _, _ = window_partition(img_mask, self.window_size)  # (nW, window_size, window_size, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量 (B, L, C)，其中 L = H*W
            
        返回：
            输出张量 (B, L, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 窗口划分
        x_windows, orig_HW, padded_HW = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, window_size*window_size, C)
        
        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (nW*B, window_size*window_size, C)
        
        # 窗口合并
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, 
                                    padded_HW[0], padded_HW[1], orig_HW)  # (B, H, W, C)
        
        # 逆向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # @kimi 修改: 应用DropPath到残差连接
        # 原代码: x = shortcut + x
        # 理由: 使用DropPath实现随机深度，以drop_path概率丢弃当前块的贡献
        x = shortcut + self.drop_path(x)
        
        # @kimi 修改: MLP残差连接也应用DropPath（可选，这里保持简单不应用）
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTBranch(nn.Module):
    """
    ViT分支模块
    
    包含两个连续的Swin块（W-MSA + SW-MSA）
    
    参数：
        dim: 输入维度（通道数）
        input_resolution: 输入分辨率 (H, W)
        num_heads: 注意力头数
        window_size: 窗口大小
        mlp_ratio: MLP扩展比例
        drop: dropout率
        drop_path: 随机深度丢弃概率 (新增，两个块使用不同概率)
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        drop_path: Tuple[float, float] = (0.0, 0.0),  # @kimi 新增: 两个块的drop_path概率
    ):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        
        # @kimi 修改: 解析drop_path为元组，支持为两个块设置不同概率
        # 理由: 深度增加，丢弃概率应增大，遵循线性调度策略
        if isinstance(drop_path, (list, tuple)):
            drop_path1, drop_path2 = drop_path[0], drop_path[1]
        else:
            drop_path1 = drop_path2 = drop_path
        
        # 第一个块：W-MSA
        self.block1 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path1  # @kimi 新增: 传递drop_path概率
        )
        
        # 第二个块：SW-MSA
        self.block2 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path2  # @kimi 新增: 传递drop_path概率（通常更大）
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量 (B, C, H, W)
            
        返回：
            输出张量 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 转换为序列格式
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        
        # 两个Swin块
        x = self.block1(x)
        x = self.block2(x)
        
        # 转换回图像格式
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        
        return x


class LightSwinBlock(nn.Module):
    """
    轻量级Swin块
    
    仅使用一个Swin块（W-MSA），用于控制参数量
    
    参数：
        dim: 输入维度
        input_resolution: 输入分辨率
        num_heads: 注意力头数
        window_size: 窗口大小
        mlp_ratio: MLP扩展比例
        drop_path: 随机深度丢弃概率 (新增)
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,  # @kimi 新增: 随机深度丢弃概率
    ):
        super().__init__()
        
        self.block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path  # @kimi 新增: 传递drop_path概率
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.block(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试ViT分支模块")
    print("=" * 60)
    
    # 测试参数
    batch_size = 2
    in_channels = 64
    height, width = 32, 32
    
    # 创建测试输入
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 测试窗口划分与合并
    print("\n[1] 测试窗口划分与合并...")
    x_test = torch.randn(1, 16, 16, 64)  # (B, H, W, C)
    windows, orig_HW, padded_HW = window_partition(x_test, window_size=4)
    print(f"  输入形状: {x_test.shape}")
    print(f"  窗口形状: {windows.shape}")
    x_reversed = window_reverse(windows, window_size=4, H=padded_HW[0], W=padded_HW[1], orig_HW=orig_HW)
    print(f"  还原形状: {x_reversed.shape}")
    print(f"  还原正确: {torch.allclose(x_test, x_reversed)}")
    
    # 测试窗口注意力
    print("\n[2] 测试WindowAttention...")
    window_attn = WindowAttention(dim=64, window_size=(7, 7), num_heads=4)
    x_window = torch.randn(10, 49, 64)  # (num_windows*B, window_size^2, C)
    out = window_attn(x_window)
    print(f"  输入形状: {x_window.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in window_attn.parameters()):,}")
    
    # 测试SwinTransformerBlock
    print("\n[3] 测试SwinTransformerBlock...")
    swin_block = SwinTransformerBlock(
        dim=in_channels,
        input_resolution=(height, width),
        num_heads=4,
        window_size=7,
        shift_size=0
    )
    x_seq = x.flatten(2).transpose(1, 2)  # (B, L, C)
    out = swin_block(x_seq)
    print(f"  输入形状: {x_seq.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in swin_block.parameters()):,}")
    
    # 测试ViTBranch
    print("\n[4] 测试ViTBranch...")
    vit_branch = ViTBranch(
        dim=in_channels,
        input_resolution=(height, width),
        num_heads=4,
        window_size=7
    )
    out = vit_branch(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in vit_branch.parameters()):,}")
    
    # 测试LightSwinBlock
    print("\n[5] 测试LightSwinBlock...")
    light_swin = LightSwinBlock(
        dim=in_channels,
        input_resolution=(height, width),
        num_heads=4,
        window_size=7
    )
    out = light_swin(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in light_swin.parameters()):,}")
    
    # 测试梯度
    print("\n[6] 测试梯度反向传播...")
    loss = out.sum()
    loss.backward()
    print("  梯度计算成功")
    
    # 测试不同尺寸
    print("\n[7] 测试不同输入尺寸...")
    for h, w in [(16, 16), (32, 64), (64, 32)]:
        x_test = torch.randn(1, in_channels, h, w)
        vit_test = ViTBranch(
            dim=in_channels,
            input_resolution=(h, w),
            num_heads=4,
            window_size=min(7, min(h, w))
        )
        out = vit_test(x_test)
        print(f"  输入 ({h}, {w}) -> 输出 {out.shape}")
    
    print("\n" + "=" * 60)
    print("所有ViT分支模块测试通过！")
    print("=" * 60)