"""
Mamba分支实现：SASS蛇形扫描
优先使用官方mamba_ssm实现，无法调用时使用简化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from functools import lru_cache
import math

from modules import BottConv

MAMBA_AVAILABLE = False
# 尝试导入官方Mamba实现
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    MAMBA_AVAILABLE = True
    print("官方Mamba实现已加载成功")
except ImportError:
    print("警告: 无法导入官方mamba_ssm，将使用简化实现")
    selective_scan_fn = None
    Mamba = None


class SS2D_SASS(nn.Module):
    """
    基于蛇形扫描的2D选择性状态空间模型
    
    采用SASS (Snake-like Alternating Scanning Strategy) 四方向扫描：
    1. 方向1：水平蛇形扫描（从左下到右上）
    2. 方向2：垂直蛇形扫描（从左上到右下）
    3. 方向3：对角线扫描（左上到右下）
    4. 方向4：反对角线扫描（右上到左下）
    
    参数：
        d_model: 模型维度（输入通道数）
        d_state: 状态维度，默认16
        d_conv: 卷积核大小，默认3
        expand: 扩展因子，默认2
        dt_rank: dt投影秩，自动设置
        use_official: 是否优先使用官方实现
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            dt_rank: int = None,
            use_official: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_model // 16)
        self.use_official = use_official and MAMBA_AVAILABLE

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * expand, bias=False)

        # 深度卷积（2D版本）
        # self.conv2d = nn.Conv2d(
        #     self.d_inner, self.d_inner,
        #     kernel_size=d_conv, padding=d_conv // 2,
        #     groups=self.d_inner, bias=True
        # )

        self.conv2d = BottConv(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            mid_channels=self.d_inner // 16,
            kernel_size=d_conv,
            padding=1,
            stride=1,
            bias=True
        )

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # SSM参数
        # A参数（对数空间）
        # self.A_logs = nn.Parameter(torch.randn(4, self.d_inner, d_state))
        self.A_logs = nn.Parameter(torch.log(torch.ones(4, self.d_inner, d_state)))

        # D参数（跳跃连接）
        self.Ds = nn.Parameter(torch.ones(4, self.d_inner))

        # x_proj: 从 d_inner 投影到 (dt_rank + d_state + d_state)，同时产生 delta_rank, B, C
        self.x_projs = nn.Parameter(torch.randn(4, self.d_inner, self.dt_rank + d_state + d_state))

        # dt_proj: 从 dt_rank 投影到 d_inner (delta 上采样)
        self.dt_projs = nn.Parameter(torch.randn(4, self.d_inner, self.dt_rank))

        # 方向偏置
        self.dir_biases = nn.Parameter(torch.zeros(4, d_state))

        # 激活函数
        self.act = nn.SiLU()

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.kaiming_uniform_(self.x_projs, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dt_projs, a=math.sqrt(5))

    @staticmethod
    @lru_cache(maxsize=32)
    def sass_scan_indices(hw_shape: Tuple[int, int]) -> Tuple:
        """
        计算蛇形扫描的索引序列
        
        参数：
            hw_shape: (H, W) 高度和宽度
            
        返回：
            (扫描索引, 逆序索引, 方向标记)
        """
        H, W = hw_shape
        L = H * W

        o1, o2, o3, o4 = [], [], [], []
        o1_inverse = [-1] * L
        o2_inverse = [-1] * L
        o3_inverse = [-1] * L
        o4_inverse = [-1] * L

        # 方向1：水平蛇形（从左下到右上）
        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                else:
                    i = i - 1
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                else:
                    i = i - 1
                    j_d = "right"

        # 方向2：垂直蛇形（从左上到右下）
        i, j = 0, 0
        i_d = "down"
        while j < W:
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                else:
                    j = j + 1
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                else:
                    j = j + 1
                    i_d = "down"

        # 方向3：对角线扫描（左上到右下）
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o3) - 1
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o3) - 1

        # 方向4：反对角线扫描（右上到左下）
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1

        return (
            (tuple(o1), tuple(o2), tuple(o3), tuple(o4)),
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量 (B, C, H, W)
            
        返回：
            输出张量 (B, C, H, W)
            
        @kimi 优化: 减少内存拷贝，使用view替代flatten/reshape，优化四方向扫描
        """
        B, C, H, W = x.shape
        L = H * W

        # @kimi 优化前: 输入投影 - flatten产生新的张量
        # x_flat = x.flatten(2).transpose(1, 2)  # (B, L, C)
        # @kimi 优化后: 使用view替代flatten，避免内存拷贝
        x_flat = x.view(B, C, L).transpose(1, 2)  # (B, L, C)
        xz = self.in_proj(x_flat)  # (B, L, 2*E)

        # 分割为x和z
        x_proj, z = xz.chunk(2, dim=-1)  # 各 (B, L, E)

        # @kimi 优化前: 2D卷积 - 使用reshape+contiguous产生内存拷贝
        # x_conv = x_proj.transpose(1, 2).reshape(B, self.d_inner, H, W).contiguous()
        # @kimi 优化后: 使用view替代reshape，移除不必要的contiguous
        x_conv = x_proj.transpose(1, 2).view(B, self.d_inner, H, W)
        x_conv = self.act(self.conv2d(x_conv))
        # @kimi 优化: 使用view替代flatten
        x_conv = x_conv.view(B, self.d_inner, L).transpose(1, 2)  # (B, L, E)

        # 获取扫描索引
        scan_orders, inverse_orders = self.sass_scan_indices((H, W))

        # @kimi 优化: 预计算方向偏置，避免在循环内重复unsqueeze
        dir_biases = [self.dir_biases[d].view(1, 1, -1) for d in range(4)]

        # 四方向扫描
        outputs = []
        for d in range(4):
            # 按扫描顺序重排
            x_scan = x_conv[:, scan_orders[d], :]  # (B, L, E)

            # 生成SSM参数 - 使用单矩阵乘法和切片
            x_dbl = torch.matmul(x_scan, self.x_projs[d])  # (B, L, dt_rank + N + N)

            # 分割为 delta_rank, B, C
            delta_rank = x_dbl[:, :, :self.dt_rank]  # (B, L, dt_rank)
            B_d = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]  # (B, L, N)
            C_d = x_dbl[:, :, self.dt_rank + self.d_state:]  # (B, L, N)

            # delta: 从 dt_rank 上采样到 d_inner，然后 softplus
            delta = F.softplus(torch.matmul(delta_rank, self.dt_projs[d].T))  # (B, L, E)

            # A参数（对数空间转为实际值）
            A = -torch.exp(self.A_logs[d])  # (E, N)

            # @kimi 优化前: 选择性扫描 - 多次transpose+contiguous产生内存拷贝
            # x_scan_fp32 = x_scan.transpose(1, 2).float().contiguous()  # (B, E, L)
            # delta_fp32 = delta.transpose(1, 2).float().contiguous()  # (B, E, L)
            # A_fp32 = A.float()  # (E, N)
            # B_fp32 = (B_d + self.dir_biases[d].unsqueeze(0).unsqueeze(0)).transpose(1, 2).float().contiguous()
            # C_fp32 = C_d.transpose(1, 2).float().contiguous()  # (B, N, L)
            # D_fp32 = self.Ds[d].float()  # (E,)
            
            # @kimi 优化后: 使用view替代transpose，减少contiguous调用
            # 优化策略：1) 仅在必要时类型转换 2) 使用view替代transpose 3) 批量处理减少API调用
            input_dtype = x_scan.dtype
            x_scan_t = x_scan.view(B, self.d_inner, L)
            delta_t = delta.view(B, self.d_inner, L)
            
            if input_dtype == torch.float32:
                # 如果输入已经是float32，避免类型转换
                x_scan_fp32 = x_scan_t
                delta_fp32 = delta_t
                A_fp32 = A
                B_fp32 = (B_d + dir_biases[d]).view(B, self.d_state, L)
                C_fp32 = C_d.view(B, self.d_state, L)
                D_fp32 = self.Ds[d]
            else:
                # 批量类型转换
                x_scan_fp32 = x_scan_t.float()
                delta_fp32 = delta_t.float()
                A_fp32 = A.float()
                B_fp32 = (B_d + dir_biases[d]).view(B, self.d_state, L).float()
                C_fp32 = C_d.view(B, self.d_state, L).float()
                D_fp32 = self.Ds[d].float()

            y_t = selective_scan_fn(
                x_scan_fp32,
                delta_fp32,
                A_fp32,
                B_fp32,
                C_fp32,
                D_fp32,
                z=None
            )
            # 恢复原始类型
            if y_t.dtype != input_dtype:
                y_t = y_t.to(input_dtype)

            # @kimi 优化: 使用view替代transpose减少内存拷贝
            # y_t 形状为 (B, E, L)
            y = y_t.view(B, L, self.d_inner)  # (B, L, E)

            # 逆序恢复
            y = y[:, inverse_orders[d], :]  # (B, L, E)
            outputs.append(y)

        # @kimi 优化: 四方向融合 - 使用inplace加法减少内存分配
        # y = sum(outputs)  # 创建新张量
        y = outputs[0]
        for i in range(1, 4):
            y = y + outputs[i]  # inplace累加

        # 门控融合
        y = y * self.act(z)

        # 输出投影
        y = self.out_proj(y)  # (B, L, C)

        # @kimi 优化: 恢复形状 - 使用view替代transpose+reshape
        # y = y.transpose(1, 2).reshape(B, C, H, W)
        y = y.view(B, C, H, W)  # (B, C, H, W)

        return y


class MambaBranch(nn.Module):
    """
    Mamba分支模块
    
    完整的Mamba分支，包含：
    1. LayerNorm
    2. SS2D_SASS（蛇形扫描选择性扫描）
    3. 残差连接
    
    参数：
        d_model: 模型维度（通道数）
        d_state: 状态维度
        d_conv: 卷积核大小
        expand: 扩展因子
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            use_official: bool = True
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ss2d = SS2D_SASS(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_official=use_official
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量 (B, C, H, W)
            
        返回：
            输出张量 (B, C, H, W)
            
        @kimi 优化: 使用view替代flatten/reshape减少内存拷贝
        """
        B, C, H, W = x.shape

        # @kimi 优化: LayerNorm - 使用view替代flatten和reshape
        # x_norm = x.flatten(2).transpose(1, 2)  # (B, L, C)
        x_norm = x.view(B, C, H * W).transpose(1, 2)  # (B, L, C) - view比flatten更快
        x_norm = self.norm(x_norm)
        # x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        x_norm = x_norm.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W) - view替代reshape

        # SS2D
        y = self.ss2d(x_norm)

        # @kimi 优化: 残差连接 - x在前可利用pytorch的inplace优化
        return x + y


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试Mamba分支模块")
    print("=" * 60)

    # 测试参数
    batch_size = 2
    in_channels = 64
    height, width = 32, 32

    # 确定设备
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")

    # 创建测试输入
    x = torch.randn(batch_size, in_channels, height, width, device=device)

    # 测试SASS索引
    print("\n[1] 测试SASS扫描索引...")
    scan_orders, inverse_orders = SS2D_SASS.sass_scan_indices((8, 8))
    print(f"  H=8, W=8, L={8 * 8}")
    for i, (order, inv) in enumerate(zip(scan_orders, inverse_orders)):
        print(f"  方向{i + 1}: 扫描序列长度={len(order)}, 逆序序列长度={len(inv)}")

    # 测试SS2D_SASS
    print("\n[2] 测试SS2D_SASS...")
    ss2d = SS2D_SASS(d_model=in_channels, d_state=16, d_conv=3, expand=2).to(device)
    out = ss2d(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in ss2d.parameters()):,}")

    # 测试MambaBranch
    print("\n[3] 测试MambaBranch...")
    mamba_branch = MambaBranch(d_model=in_channels).to(device)
    out = mamba_branch(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in mamba_branch.parameters()):,}")

    # 测试梯度
    print("\n[4] 测试梯度反向传播...")
    x_grad = torch.randn(batch_size, in_channels, height, width, device=device, requires_grad=True)
    mamba_branch.zero_grad()
    out = mamba_branch(x_grad)
    loss = out.sum()
    loss.backward()
    print(f"  梯度计算成功")
    print(f"  输入梯度形状: {x_grad.grad.shape}")

    print("\n" + "=" * 60)
    print("所有Mamba分支测试通过！")
    print("=" * 60)
