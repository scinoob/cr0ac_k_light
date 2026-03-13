"""
Mamba分支实现：SASS蛇形扫描
优先使用官方mamba_ssm实现，无法调用时使用简化版本
"""
import torch
import torch.nn as nn
import math

from einops import repeat
from torch.nn.init import trunc_normal_

from modules import BottConv

# @kimi 新增: 导入DropPath用于随机深度正则化
# 理由: 实现深度网络的随机深度(Stochastic Depth)机制，提高泛化能力
from timm.layers import DropPath

from modules.SS2D_quard_parallel import SS2D_QuadParallel

MAMBA_AVAILABLE = False
# 尝试导入官方Mamba实现
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    MAMBA_AVAILABLE = True
    print("官方Mamba实现已加载成功")
except ImportError:
    raise ImportError("警告: 无法导入官方mamba_ssm，请检查安装环境..")


class SS2D_SASS(nn.Module):
    def __init__(
            self,
            d_model: int,
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
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        # 内部空间维度 d_inner = expand * d_model =2*256=512
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4

        # 输入投影：将 d_model 映射到 2*d_inner，以便后面拆分为 x 和 z（门控）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert d_conv % 2 == 1
        # 先学习一遍通道信息，然后在学习空间信息，最后将信息组合
        # <=>nn.conv2d(in_ch,out_ch,k,padding,stride)
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16,
                               kernel_size=d_conv, padding=1, stride=1)
        self.activation = "silu"
        self.act = nn.SiLU()

        # 从卷积输出 x_conv 生成 dt、B、C 的投影
        # dt_rank + d_state*2 维，分别对应 delta、B 矩阵、C 矩阵
        # 512->16+16*2=48
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        # 将 dt 从 dt_rank 投影回 d_inner 维度
        # 16->512
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )
        # 根据 dt_init 初始化 dt_proj 的权重
        # dt_init_std=16^-0.5*1=0.25
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            # Fills the input Tensor with values drawn from the uniform distribution U(a,b)
            # uniform_(tensor,a,b)
            # 限定投影的权重-0.25,0.25
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 特殊初始化 dt_proj 的偏置，使 delta 大致落在 [dt_min, dt_max] 区间
        # dt_init_floor=1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # 构造 A 矩阵的对数形式，A 是状态转移矩阵（对角阵的简化形式）
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True  # 优化时不对 A_log 进行权重衰减

        # D 参数：跳跃连接（skip connection）系数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 输出投影：将内部维度映射回 d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # 方向偏置：为四个扫描方向分别学习一个 d_state 维的向量，用于调制 B 矩阵
        # 多一个维度可能用于占位或未使用（索引 0 可能对应无方向？具体由 sass 返回的方向值决定）
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    @staticmethod
    def sass(hw_shape):
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

        #
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
        # 注意：输入的x为B,C,H,W，不能直接使用，要转化为B,L,d_model

        # x: (B, L, d_model), L = H*W
        batch_size, _, H, W = x.shape
        L = H * W
        # 原始特征图的高和宽
        # (512 // 8, 512 // 8)
        hw_shape = (H, W)
        E = self.d_inner
        x = x.flatten(2).transpose(1, 2)

        conv_state, ssm_state = None, None
        xz = self.in_proj(x)  # (B, L, 2*E)
        A = -torch.exp(self.A_log.float())  # 从对数恢复 A 并取负，得到 (E, d_state)

        # 将 xz 拆分为 x 和 z（门控），形状均为 (B, L, E)
        x, z = xz.chunk(2, dim=-1)

        # 将 x 重塑为 2D 特征图，经过瓶颈卷积，再还原为序列形式
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        # 从卷积输出生成 dt, B, C
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank+2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, L, E) -> 再转置
        dt = dt.permute(0, 2, 1).contiguous()  # (B, E, L)
        B = B.permute(0, 2, 1).contiguous()  # (B, d_state, L)
        C = C.permute(0, 2, 1).contiguous()  # (B, d_state, L)

        assert self.activation in ["silu", "swish"]

        # 生成四个扫描顺序、逆顺序和方向编码
        orders, inverse_orders, directions = self.sass(hw_shape)
        # 为每个位置添加方向偏置 dB，形状 (B, d_state, 1) 以便与 B 广播相加
        direction_Bs = [self.direction_Bs[d, :] for d in directions]  # 每个 dB 是 (d_state,)
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in
                        direction_Bs]  # 最终形状 (B, d_state, 1)

        # 对四个方向分别执行 selective_scan_fn
        y_scan = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),  # (B, E, len(o)) 按顺序重排
                dt,  # (B, E, L)
                A,  # (E, d_state)
                (B + dB).contiguous(),  # (B, d_state, L) 添加方向调制
                C,  # (B, d_state, L)
                self.D,  # (E,)
                z=None,  # 不使用额外门控
                delta_bias=self.dt_proj.bias,  # (E,)  delta 偏置
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_order, :]  # 输出 (B, L, E) 并恢复原始顺序
            for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs)
        ]

        # 四个方向结果相加，再乘以门控 z 的激活值
        y = sum(y_scan) * self.act(z)  # (B, L, E)
        out = self.out_proj(y)  # (B,L,d_model)

        out = torch.transpose(out, 1, 2).reshape(batch_size, self.d_model, H, W)

        return out


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
        drop_path: 随机深度丢弃概率 (新增)
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 3,
            expand: int = 2,
            use_official: bool = True,
            drop_path: float = 0.0,  # @kimi 新增: 随机深度丢弃概率
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        # 原版ss2d
        # self.ss2d = SS2D_SASS(
        #     d_model=d_model,
        #     d_state=d_state,
        #     d_conv=d_conv,
        #     expand=expand,
        #     # use_official=use_official
        # )

        # 新版四向平行ss2d
        self.ss2d = SS2D_QuadParallel(
            input_dim=d_model,
            output_dim=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand)

        # @kimi 新增: DropPath层用于随机深度
        # 理由: 在残差连接上应用随机深度，以一定概率丢弃整个Mamba分支的变换
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

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

        # 在dual_branch进入前已经进行层归一化
        # x_norm = self.norm(x_norm)
        # x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        x_norm = x_norm.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W) - view替代reshape

        # SS2D
        y = self.ss2d(x_norm)

        # @kimi 修改: 应用DropPath到残差连接
        # 原代码: return x + y
        # 理由: 使用DropPath实现随机深度，以drop_path概率丢弃当前Mamba分支的贡献
        return x + self.drop_path(y)


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
