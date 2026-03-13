import math

import torch
import torch.nn as nn
from einops import repeat
from timm.layers import trunc_normal_
from modules import BottConv

MAMBA_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    raise ImportError('Please install mama-ssm first.')


class SS2D_SASS_Quarter(nn.Module):
    """
    四分之一维度的 SS2D SASS 模块

    【功能】处理 d_model//4 维度的输入，执行四方向蛇形扫描

    【相对原 SS2D.py 的改动】
    1. 输入维度从 d_model 改为 d_model//4
    2. 内部维度 d_inner_quarter = expand * (d_model//4)
    3. 用于被 SS2D_QuadParallel 的4个分支共享使用
    4. 保留完整的 sass() 四方向扫描机制
    5. 保留方向偏置 direction_Bs 的学习
    """

    def __init__(
            self,
            d_model: int,  # 原始完整维度
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            dt_rank: str = "auto",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
    ):
        super().__init__()

        self.d_model_full = d_model
        # 【改动】使用四分之一维度
        self.d_model = d_model // 4
        self.d_state = d_state
        self.expand = expand
        # 【改动】内部维度也相应变为四分之一
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4

        # 【改动】输入投影维度改为四分之一
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert d_conv % 2 == 1
        # 【改动】瓶颈卷积维度改为四分之一
        self.conv2d = BottConv(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            mid_channels=self.d_inner // 16,
            kernel_size=d_conv, padding=1, stride=1
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        # 【改动】投影维度改为四分之一
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # dt_proj 权重初始化
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt_proj 偏置初始化
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A 矩阵初始化
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D 参数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 【改动】输出投影维度改为四分之一
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # 方向偏置（保持4个方向）
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    @staticmethod
    def sass(hw_shape):
        """
        【功能】生成四方向蛇形扫描顺序

        【相对原 SS2D.py 的改动】无改动，完全复用原有的 sass 方法

        生成4个方向的扫描顺序：
        - o1: 水平蛇形扫描（从左到右/右到左交替）
        - o2: 垂直蛇形扫描（从上到下/下到上交替）
        - o3: 对角线扫描（从左上到右下）
        - o4: 反对角线扫描（从右上到左下）
        """
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        # 水平蛇形扫描
        while i > -1:
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        # 垂直蛇形扫描
        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d2.append(4)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d2.append(3)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "down"
        d2 = [0] + d2[:-1]

        # 对角线扫描
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(4 if i == diag else 1)
        d3 = [0] + d3[:-1]

        # 反对角线扫描
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(4 if i == diag else 1)
        d4 = [0] + d4[:-1]

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))

    def forward(self, x):
        """
        【功能】前向传播，执行四方向蛇形扫描

        【相对原 SS2D.py 的改动】
        1. 输入维度改为 d_model//4
        2. 内部计算使用四分之一维度
        3. 输出维度为 d_model//4
        """
        batch_size, _, H, W = x.shape
        L = H * W
        hw_shape = (H, W)
        E = self.d_inner

        # 展平为序列形式
        x = x.flatten(2).transpose(1, 2)

        conv_state, ssm_state = None, None
        xz = self.in_proj(x)  # (B, L, 2*E_quarter)
        A = -torch.exp(self.A_log.float())

        # 拆分为 x 和 z
        x, z = xz.chunk(2, dim=-1)

        # 卷积处理
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        # 生成 dt, B, C
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        assert self.activation in ["silu", "swish"]

        # 生成扫描顺序
        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in
                        direction_Bs]

        # 四方向 selective scan
        y_scan = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D,
                z=None,
                delta_bias=self.dt_proj.bias,
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_order, :]
            for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs)
        ]

        # 融合四方向结果
        y = sum(y_scan) * self.act(z)
        out = self.out_proj(y)

        # 恢复为特征图形式
        out = torch.transpose(out, 1, 2).reshape(batch_size, self.d_model, H, W)

        return out


class SS2D_QuadParallel(nn.Module):
    """
    四路并行 SS2D 模块

    【功能】模仿 PVMLayer 的四路并行机制，将输入通道拆分为4份，
           使用共享的 SS2D_SASS_Quarter 模块并行处理，最后拼接结果

    【相对原 SS2D.py 的核心改动】
    1. **通道拆分机制**：输入通道 C 拆分为4份，每份 C/4
    2. **权重共享**：4个分支共享同一个 SS2D_SASS_Quarter 实例
    3. **并行处理**：每个分支独立处理 C/4 通道的特征
    4. **结果拼接**：将4个分支的输出拼接后投影到目标维度
    5. **参数量减少**：通过权重共享，参数量显著降低

    【参数对比】
    - 原始 SS2D：d_model = C, d_inner = expand * C
    - 本模块：每个分支 d_model = C/4, d_inner = expand * C/4
             共享同一个 SS2D 实例，总参数量约为原始的 1/4
    """

    def __init__(
            self,
            input_dim: int,  # 输入通道数 (对应原 d_model)
            output_dim: int,  # 输出通道数
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            dt_rank: str = "auto",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 【新增】验证输入维度可被4整除
        assert input_dim % 4 == 0, f"input_dim must be divisible by 4, got {input_dim}"

        # 【新增】LayerNorm 用于归一化
        self.norm = nn.LayerNorm(input_dim)

        # 【核心改动】创建共享的 SS2D_SASS_Quarter 模块
        # 该模块处理 input_dim//4 维度的输入
        self.ss2d_quarter = SS2D_SASS_Quarter(
            d_model=input_dim,  # 传入完整维度，内部会除以4
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            bias=bias,
        )

        # 【新增】跳跃连接缩放参数，模仿 PVMLayer
        self.skip_scale = nn.Parameter(torch.ones(1))

        # 【新增】输出投影层，将拼接后的特征映射到目标维度
        # 拼接后维度为 input_dim，投影到 output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        【功能】前向传播 - 四路并行处理

        【处理流程】
        1. 输入: (B, C, H, W)，其中 C = input_dim
        2. 展平并归一化: (B, H*W, C)
        3. 通道拆分: 4份，每份 (B, H*W, C/4)
        4. 四路并行: 每个分支 reshape 为 (B, C/4, H, W) 后输入共享 SS2D
        5. 结果拼接: 4个 (B, C/4, H, W) -> (B, C, H, W)
        6. 投影输出: (B, output_dim, H, W)

        【相对原 SS2D.py 的改动】
        - 原 SS2D：直接处理完整维度，单路处理
        - 本模块：拆分通道，四路并行，权重共享
        """

        # 类型处理
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]
        assert C == self.input_dim, f"Input channels {C} != expected {self.input_dim}"

        # 计算空间维度
        n_tokens = x.shape[2:].numel()  # H * W
        img_dims = x.shape[2:]  # (H, W)
        H, W = img_dims

        # 展平空间维度并转置: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        # LayerNorm 归一化
        x_norm = self.norm(x_flat)  # (B, H*W, C)

        # 【核心改动】通道拆分为4份，沿特征维度(dim=2)拆分
        # 每份: (B, H*W, C/4)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        # 【核心改动】将每份 reshape 回特征图形式，输入共享的 SS2D
        # 从 (B, H*W, C/4) -> (B, C/4, H, W)
        quarter_dim = C // 4
        x1_2d = x1.transpose(-1, -2).reshape(B, quarter_dim, H, W)
        x2_2d = x2.transpose(-1, -2).reshape(B, quarter_dim, H, W)
        x3_2d = x3.transpose(-1, -2).reshape(B, quarter_dim, H, W)
        x4_2d = x4.transpose(-1, -2).reshape(B, quarter_dim, H, W)

        # 【核心改动】四路并行处理，共享同一个 SS2D_SASS_Quarter
        # 每个分支输出: (B, C/4, H, W)
        y1 = self.ss2d_quarter(x1_2d)
        y2 = self.ss2d_quarter(x2_2d)
        y3 = self.ss2d_quarter(x3_2d)
        y4 = self.ss2d_quarter(x4_2d)

        # 【新增】跳跃连接（模仿 PVMLayer）
        y1 = y1 + self.skip_scale * x1_2d
        y2 = y2 + self.skip_scale * x2_2d
        y3 = y3 + self.skip_scale * x3_2d
        y4 = y4 + self.skip_scale * x4_2d

        # 【核心改动】将结果展平并拼接
        # 从 (B, C/4, H, W) -> (B, H*W, C/4)
        y1_flat = y1.reshape(B, quarter_dim, n_tokens).transpose(-1, -2)
        y2_flat = y2.reshape(B, quarter_dim, n_tokens).transpose(-1, -2)
        y3_flat = y3.reshape(B, quarter_dim, n_tokens).transpose(-1, -2)
        y4_flat = y4.reshape(B, quarter_dim, n_tokens).transpose(-1, -2)

        # 拼接: 4个 (B, H*W, C/4) -> (B, H*W, C)
        y_mamba = torch.cat([y1_flat, y2_flat, y3_flat, y4_flat], dim=2)

        # 再次归一化
        y_mamba = self.norm(y_mamba)

        # 投影到输出维度
        y_mamba = self.proj(y_mamba)  # (B, H*W, output_dim)

        # 恢复为特征图形式: (B, output_dim, H, W)
        out = y_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        return out


# ============================================================================
# 使用示例和测试代码
# ============================================================================

if __name__ == "__main__":
    """
    【功能】测试四路并行 SS2D 模块
    """
    import torch

    # 测试参数
    batch_size = 2
    input_dim = 32  # 必须是4的倍数
    output_dim = 32
    H, W = 64, 64

    # 创建输入
    x = torch.randn(batch_size, input_dim, H, W)

    print("=" * 60)
    print("SS2D 四路并行模块测试")
    print("=" * 60)
    print(f"输入形状: {x.shape}")
    print(f"输入通道: {input_dim}, 输出通道: {output_dim}")
    print(f"每分支处理通道: {input_dim // 4}")

    # 创建模型
    model = SS2D_QuadParallel(
        input_dim=input_dim,
        output_dim=output_dim,
        d_state=16,
        d_conv=3,
        expand=2,
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params / 1e6:.4f}M")

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"输出形状: {output.shape}")
    print(f"\n测试通过！")

    # 对比：原始 SS2D 的参数量估算
    print("\n" + "=" * 60)
    print("参数量对比")
    print("=" * 60)

    # 原始 SS2D (单路，完整维度)
    # 估算主要参数: in_proj + x_proj + dt_proj + out_proj + 卷积 + A + D + direction_Bs
    d_inner = input_dim * 2  # expand=2
    dt_rank = math.ceil(input_dim / 16)

    orig_params = (
            input_dim * d_inner * 2 +  # in_proj
            d_inner * (dt_rank + 32) +  # x_proj (d_state=16, so 2*d_state=32)
            dt_rank * d_inner + d_inner +  # dt_proj (weight + bias)
            d_inner * input_dim +  # out_proj
            d_inner * (d_inner // 16) + (d_inner // 16) * d_inner +  # BottConv (简化估算)
            d_inner * 16 +  # A_log
            d_inner +  # D
            5 * 16  # direction_Bs
    )

    print(f"原始 SS2D (单路) 估算参数量: ~{orig_params / 1e6:.4f}M")
    print(f"四路并行 SS2D 实际参数量: {total_params / 1e6:.4f}M")
    print(f"参数量比例: {total_params / orig_params:.2%}")
