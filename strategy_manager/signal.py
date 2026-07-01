"""
strategy_manager/signal.py — 回测与实盘共享的信号计算模块

提供：
  compute_target_positions(factors)  →  {-1, 0, +1} 目标仓位张量
  reconcile_action(current, target)  →  动作字符串

回测（backtest.py）与实盘（runner.py）共用此模块，
保证「信号→目标仓位」逻辑严格一致，消除回测-实盘偏差。
"""
from __future__ import annotations

import torch
from torch import Tensor


def compute_target_positions(factors: Tensor) -> Tensor:
    """将因子张量转换为目标仓位 {-1, 0, +1}。

    与回测 backtest.py 的逻辑完全一致：
        signal     = tanh(factors)
        target_pos = sign(signal)

    Args:
        factors: [N, T] 或 [N]（实盘最新 bar）的因子张量。

    Returns:
        与 factors 同形状的整数张量，值为 {-1, 0, +1}。
        tanh 输出精确为 0 的概率极低，实践中几乎只出现 ±1。
    """
    return torch.sign(torch.tanh(factors))


# ── 动作常量 ──────────────────────────────────────────────────────────────────
HOLD            = "HOLD"
OPEN_LONG       = "OPEN_LONG"
OPEN_SHORT      = "OPEN_SHORT"
CLOSE           = "CLOSE"
REVERSE_TO_LONG = "REVERSE_TO_LONG"
REVERSE_TO_SHORT = "REVERSE_TO_SHORT"


def reconcile_action(current: int, target: int) -> str:
    """根据当前仓位方向和目标方向，返回应执行的动作。

    Args:
        current: 当前仓位方向，+1（多）/ -1（空）/ 0（空仓）。
        target:  目标仓位方向，+1 / -1 / 0。

    Returns:
        动作字符串，取值为模块级常量之一：
        HOLD / OPEN_LONG / OPEN_SHORT / CLOSE /
        REVERSE_TO_LONG / REVERSE_TO_SHORT
    """
    if current == target:
        return HOLD
    if current == 0:
        return OPEN_LONG if target == 1 else OPEN_SHORT
    if target == 0:
        return CLOSE
    return REVERSE_TO_LONG if target == 1 else REVERSE_TO_SHORT
