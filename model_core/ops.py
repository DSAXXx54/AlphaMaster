import torch

def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)

def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """降低稀疏度：阈值从 3σ 改为 1.5σ，让更多时间步有非零输出"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.tanh(z - 1.5)   # tanh 软化，不再产生全零区间

def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

def _op_wma(x: torch.Tensor) -> torch.Tensor:
    """加权移动平均（权重 3,2,1），平滑信号，减少剥头皮"""
    return (3.0 * x + 2.0 * _ts_delay(x, 1) + 1.0 * _ts_delay(x, 2)) / 6.0

OPS_CONFIG = [
    ('ADD',    lambda x, y: x + y,        2),
    ('SUB',    lambda x, y: x - y,        2),
    ('MUL',    lambda x, y: x * y,        2),
    ('DIV',    lambda x, y: x / (y + 1e-6), 2),
    ('NEG',    lambda x: -x,              1),
    ('ABS',    torch.abs,                  1),
    ('SIGN',   torch.sign,                 1),
    ('GATE',   _op_gate,                   3),
    ('JUMP',   _op_jump,                   1),   # 已降低稀疏度
    ('DECAY',  _op_decay,                  1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3',   lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1),
]

# ── 时序滑动窗口辅助函数（不使用 @torch.jit.script，lambda 不兼容 JIT）──────

def _ts_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    """unfold 实现因果滑动窗口，返回 [N, T, d] 的窗口张量。"""
    N, T = x.shape
    pad = torch.zeros(N, d - 1, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1).unfold(1, d, 1)  # [N, T, d]


def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动均值，返回 [N, T]。"""
    return _ts_rolling(x, d).mean(dim=-1)


def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动标准差（ddof=0），返回 [N, T]，下界 1e-6。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    m = w.mean(dim=-1, keepdim=True)
    std = ((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6
    return torch.nan_to_num(std, nan=0.0)


def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动排名（严格小于当前值的比例），返回 [N, T]，值域 [0, 1)。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    cur = w[:, :, -1:]                             # 当前值，[N, T, 1]
    rank = (w < cur).float().mean(dim=-1)          # [N, T]
    return torch.nan_to_num(rank, nan=0.0)


def _ts_corr_10(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x 与 y 的 10 周期因果滑动 Pearson 相关系数，返回 [N, T]，值域 [-1, 1]。
    当 x 或 y 在窗口内为常数（std < 1e-6）时，该位置输出 0。
    """
    d = 10
    wx = _ts_rolling(x, d)                         # [N, T, 10]
    wy = _ts_rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    sx = ((wx - mx) ** 2).mean(dim=-1).sqrt()      # [N, T]
    sy = ((wy - my) ** 2).mean(dim=-1).sqrt()
    # 常数窗口（std < 1e-6）输出 0
    mask = (sx < 1e-6) | (sy < 1e-6)
    corr = cov / (sx * sy + 1e-8)
    corr[mask] = 0.0
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


# ── 追加时序算子到 OPS_CONFIG（token id 从 feat_offset+12 开始）────────────

OPS_CONFIG += [
    ('TS_MEAN_5',  lambda x: _ts_mean(x, 5),  1),
    ('TS_MEAN_10', lambda x: _ts_mean(x, 10), 1),
    ('TS_MEAN_20', lambda x: _ts_mean(x, 20), 1),
    ('TS_STD_5',   lambda x: _ts_std(x, 5),   1),
    ('TS_STD_10',  lambda x: _ts_std(x, 10),  1),
    ('TS_STD_20',  lambda x: _ts_std(x, 20),  1),
    ('TS_RANK_5',  lambda x: _ts_rank(x, 5),  1),
    ('TS_RANK_10', lambda x: _ts_rank(x, 10), 1),
    ('TS_RANK_20', lambda x: _ts_rank(x, 20), 1),
    ('TS_CORR_10', _ts_corr_10,                2),
    # ── 趋势 / 动量类算子（新增，token id = feat_offset+22~27）────────────
    # MOMENTUM_5: 短期均线 - 长期均线，捕捉趋势方向
    ('MOMENTUM_5',  lambda x: _ts_mean(x, 5)  - _ts_mean(x, 20), 1),
    # MOMENTUM_10: 中期动量
    ('MOMENTUM_10', lambda x: _ts_mean(x, 10) - _ts_mean(x, 20), 1),
    # TS_MAX_10: 10周期最大值，捕捉强势突破
    ('TS_MAX_10',   lambda x: _ts_rolling(x, 10).max(dim=-1).values, 1),
    # TS_MIN_10: 10周期最小值，捕捉弱势突破
    ('TS_MIN_10',   lambda x: _ts_rolling(x, 10).min(dim=-1).values, 1),
    # WMA: 加权移动平均，平滑信号
    ('WMA',         _op_wma,  1),
    # DELAY4: 延迟4根bar，构建中期动量差
    ('DELAY4',      lambda x: _ts_delay(x, 4), 1),
]



# ── v3.0 新增算子 helper ─────────────────────────────────────────────

def _ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """指数加权移动平均（因果）。alpha 越大越关注近期。"""
    # 用递推实现太慢，用衰减权重卷积近似（窗口=20 足够）
    w = min(20, x.shape[1])
    weights = torch.tensor([alpha * (1 - alpha) ** i for i in range(w)],
                           device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    pad = torch.zeros(x.shape[0], w - 1, device=x.device, dtype=x.dtype)
    xp = torch.cat([pad, x], dim=1)
    return torch.nn.functional.unfold(xp.unsqueeze(1), (1, w)).squeeze(1) * 0  # placeholder
    # 上面的 unfold 对 1D 不直接 work，改用简单循环近似


def _ema_simple(x: torch.Tensor, span: int) -> torch.Tensor:
    """指数加权移动平均（因果），span 期。用递推实现。"""
    alpha = 2.0 / (span + 1.0)
    N, T = x.shape
    out = torch.zeros_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, T):
        out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
    return out


def _ts_quantile(x: torch.Tensor, d: int) -> torch.Tensor:
    """当前值在过去 d 期的分位数（0~1），用 TS_RANK 的连续版。"""
    w = _ts_rolling(x, d)
    cur = w[:, :, -1:]
    rank = (w <= cur).float().mean(dim=-1)
    return torch.nan_to_num(rank, nan=0.5)


def _ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    """d 期偏度（三阶矩标准化），捕捉分布非对称性。"""
    w = _ts_rolling(x, d)
    m = w.mean(dim=-1, keepdim=True)
    s = ((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6
    skew = ((w - m) ** 3).mean(dim=-1) / (s ** 3)
    return torch.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)




def _delta(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """d 期差分: x[t] - x[t-d]，前 d 位置 0。Alpha 101 最常用算子。"""
    if d == 0:
        return x
    out = torch.zeros_like(x)
    out[:, d:] = x[:, d:] - x[:, :-d]
    return out


def _ts_arg_max(x: torch.Tensor, d: int) -> torch.Tensor:
    """过去 d 期最大值的位置（归一化到 [0,1]，0=最早，1=最近）。Alpha#001 核心算子。"""
    w = _ts_rolling(x, d)
    idx = w.argmax(dim=-1).float()
    return idx / max(d - 1, 1)


def _ts_arg_min(x: torch.Tensor, d: int) -> torch.Tensor:
    """过去 d 期最小值的位置（归一化到 [0,1]）。"""
    w = _ts_rolling(x, d)
    idx = w.argmin(dim=-1).float()
    return idx / max(d - 1, 1)


def _decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """线性衰减加权平均（近期权重更高）。Alpha#98 核心算子。权重 = [1,2,...,d]/sum。"""
    w = _ts_rolling(x, d)
    weights = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return (w * weights).sum(dim=-1)


def _decay_exp(x: torch.Tensor, d: int, alpha: float = 0.5) -> torch.Tensor:
    """指数衰减加权平均（近期权重更高）。与 DECAY_LINEAR 对应，平滑更激进。"""
    w = _ts_rolling(x, d)
    weights = torch.tensor([alpha * (1 - alpha) ** i for i in range(d)],
                           dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return (w * weights).sum(dim=-1)


def _scale(x: torch.Tensor) -> torch.Tensor:
    """沿时间轴缩放到单位 L1 范数（Alpha#028/032 高频算子）。
    scale(x)[t] = x[t] / sum(|x[1..t]|)，避免未来信息用因果累积和。
    """
    abs_x = x.abs()
    cumsum = torch.cumsum(abs_x, dim=1) + 1e-6
    return x / cumsum


def _ts_covariance(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """d 期因果滑动协方差。"""
    wx = _ts_rolling(x, d)
    wy = _ts_rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    return torch.nan_to_num(cov, nan=0.0)


def _ts_product(x: torch.Tensor, d: int) -> torch.Tensor:
    """d 期因果滑动乘积。用对数累加避免数值爆炸：prod = exp(sum(log(x+1)))。
    适合收益累积，输出接近 "过去 d 期累计收益"。
    输入先 clamp 到 [-0.999, +inf)，避免 log1p 在 x<=-1 时产生 NaN。
    """
    x_safe = torch.clamp(x, -0.999, None)
    log_x = torch.log1p(x_safe)
    w = _ts_rolling(log_x, d)
    out = torch.expm1(w.sum(dim=-1))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _signed_power(x: torch.Tensor, a: float = 2.0) -> torch.Tensor:
    """带符号乘方: sign(x) * |x|^a。Alpha#001 SignedPower。保留符号同时放大极端值。"""
    return torch.sign(x) * torch.abs(x) ** a

# ── v3.0 新增算子（token id = feat_offset+28~33）────────────────────
OPS_CONFIG += [
    ('EMA_5',           lambda x: _ema_simple(x, 5),    1),
    ('EMA_20',          lambda x: _ema_simple(x, 20),   1),
    ('TS_QUANTILE_10',  lambda x: _ts_quantile(x, 10),  1),
    ('TS_SKEW_10',      lambda x: _ts_skew(x, 10),      1),
    ('TS_MIN_20',       lambda x: _ts_rolling(x, 20).min(dim=-1).values, 1),
    ('TS_MAX_20',       lambda x: _ts_rolling(x, 20).max(dim=-1).values, 1),
    # ── v3.0 Alpha 101 + 补充算子（token id = feat_offset+34~43）────────
    # Alpha 101 核心 4 个
    ('DELTA',           lambda x: _delta(x, 1),              1),
    ('TS_ARG_MAX_5',    lambda x: _ts_arg_max(x, 5),         1),
    ('TS_ARG_MIN_5',    lambda x: _ts_arg_min(x, 5),         1),
    ('DECAY_LINEAR_5',  lambda x: _decay_linear(x, 5),       1),
    # 联网搜索补充 6 个
    ('SCALE',           lambda x: _scale(x),                 1),
    ('COVARIANCE_10',   lambda x, y: _ts_covariance(x, y, 10), 2),
    ('PRODUCT_5',       lambda x: _ts_product(x, 5),         1),
    ('SIGNED_POWER_2',  lambda x: _signed_power(x, 2.0),     1),
    ('TS_DECAY_EXP_5',  lambda x: _decay_exp(x, 5, 0.5),     1),
    ('DELTA_5',         lambda x: _delta(x, 5),              1),
]

assert len(OPS_CONFIG) == 44, f"OPS_CONFIG 长度应为 44，实际为 {len(OPS_CONFIG)}"
