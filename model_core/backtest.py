"""
model_core/backtest.py — MT5 回测评估器

使用 Sortino Ratio 作为适应度评分。
信号逻辑委托 strategy_manager.signal.compute_target_positions()，
与实盘 runner.py 共享同一套「因子→仓位」转换，消除回测-实盘偏差。
"""
import math
import torch
from torch import Tensor

from strategy_manager.signal import compute_target_positions


# H1 周期每年 K 线数（约 252 交易日 × 6.5 小时 × 约 1 根/小时）
# 外汇市场 5 天 × 24 小时 × 52 周 = 6240；此处使用 6240 作为默认值
_H1_PERIODS_PER_YEAR = 6240


class MT5Backtest:
    """MT5 品种的回测评估器。

    评分逻辑：
    1. signal = tanh(factors)，position = sign(signal)
    2. turnover = |pos[t] - pos[t-1]|
    3. pnl = position * target_ret - turnover * cost_rate
    4. 前 80% 为 in-sample，后 20% 为 out-of-sample
    5. Sortino Ratio = mean(pnl) / downside_std(pnl) * sqrt(periods_per_year)
    6. 若 turnover.mean() > 0.5，fitness_score -= 1.0
    7. 返回 (in-sample Sortino as scalar Tensor, oos mean as float)
    """

    def __init__(
        self,
        cost_rate: float = 0.0001,
        periods_per_year: int = _H1_PERIODS_PER_YEAR,
    ):
        """初始化 MT5Backtest。

        Args:
            cost_rate: 单边点差+佣金，默认 0.0001（forex/metals）。
            periods_per_year: 每年 K 线数，用于 Sortino 年化，默认 6240（H1）。
        """
        self.cost_rate = cost_rate
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # 内部辅助：Sortino Ratio 计算
    # ------------------------------------------------------------------
    # Sortino 输出截断上下限：±20 在实践中已是极高值，超出则是噪声
    _SORTINO_CLIP: float = 20.0

    def _sortino(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        """计算展平后 pnl 序列的 Sortino Ratio（标量 Tensor）。

        防爆炸处理：
        1. downside_std 取 max(实际std, |mean_pnl|, eps)，保证分母有意义下界，
           使 Sortino 在"全正收益"极端情形下也不超过 sqrt(periods_per_year) ≈ 79。
        2. 最终结果 clamp 到 [-_SORTINO_CLIP, +_SORTINO_CLIP]，截断任何残余极值。

        Args:
            pnl:  任意形状的 PnL 张量，计算前会展平。
            eps:  数值稳定下限，防止绝对除零。

        Returns:
            标量 Tensor，Sortino Ratio，值域 [-100, 100]。
        """
        flat = pnl.reshape(-1)
        mean_pnl = flat.mean()

        downside = flat[flat < 0]
        if downside.numel() > 0:
            raw_std = downside.std(unbiased=False)
        else:
            raw_std = torch.tensor(0.0, dtype=flat.dtype, device=flat.device)

        # 分母下限 = max(实际下行std, |mean_pnl|, eps)
        # 当无负收益时，|mean_pnl| 作为有意义的下限，
        # 保证 Sortino ≤ 1 * sqrt(periods_per_year)，不发散。
        floor = torch.clamp(mean_pnl.abs(), min=eps)
        downside_std = torch.clamp(raw_std, min=floor)

        sortino = mean_pnl / downside_std * math.sqrt(self.periods_per_year)

        # 最终截断，防止任何残余极值污染 reward
        return torch.clamp(sortino, -self._SORTINO_CLIP, self._SORTINO_CLIP)

    # ------------------------------------------------------------------
    # Walk-Forward 辅助接口
    # ------------------------------------------------------------------
    def evaluate_fold(
        self,
        factors: Tensor,
        target_ret: Tensor,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> tuple[Tensor, Tensor]:
        """在指定的训练/验证切片上计算 Sortino。

        用于 Walk-Forward 验证：调用方负责传入每折的时间索引。
        换手率惩罚依然基于全局换手率（完整序列），保持与原 evaluate() 一致。

        Args:
            factors:     因子矩阵，形状 [N, T]。
            target_ret:  目标收益率，形状 [N, T]。
            train_start: 训练段起始索引（含）。
            train_end:   训练段结束索引（不含）。
            val_start:   验证段起始索引（含）。
            val_end:     验证段结束索引（不含）。

        Returns:
            (train_sortino, val_sortino) — 均为标量 Tensor。
        """
        signal = compute_target_positions(factors)   # tanh → sign，与实盘共享
        position = signal

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        pnl = position * target_ret - turnover * self.cost_rate

        pnl_train = pnl[:, train_start:train_end]
        pnl_val   = pnl[:, val_start:val_end]

        train_score = self._sortino(pnl_train)
        val_score   = self._sortino(pnl_val)

        # 换手率惩罚（基于本折训练段）
        if turnover[:, train_start:train_end].mean() > 0.5:
            train_score = train_score - 1.0

        return train_score, val_score

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def evaluate(
        self,
        factors: Tensor,
        raw_dict: dict,
        target_ret: Tensor,
    ) -> tuple[Tensor, float]:
        """评估一组 Alpha 因子的回测表现。

        Args:
            factors:    因子矩阵，形状 [N, T]，StackVM 的输出。
            raw_dict:   原始 OHLCV 字典（当前方法未直接使用，保留供扩展）。
            target_ret: 目标收益率，形状 [N, T]。

        Returns:
            (fitness_score, mean_oos_return)
            - fitness_score:    标量 Tensor，in-sample Sortino（含惩罚项）。
            - mean_oos_return:  float，out-of-sample 均值收益率。
        """
        signal = compute_target_positions(factors)   # tanh → sign，与实盘共享
        position = signal

        # ── 2. 换手率 ──────────────────────────────────────────────
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0                   # 第一个时间步无前仓位
        turnover = torch.abs(position - prev_pos)  # [N, T]

        # ── 3. PnL ──────────────────────────────────────────────────
        pnl = position * target_ret - turnover * self.cost_rate  # [N, T]

        # ── 4. 80/20 分割 ───────────────────────────────────────────
        T = factors.shape[1]
        split = int(math.floor(T * 0.8))       # 等价于 torch.floor(T * 0.8).int()

        pnl_is = pnl[:, :split]                # in-sample:  前 80%
        pnl_oos = pnl[:, split:]               # out-of-sample: 后 20%

        # ── 5. Sortino Ratio（in-sample）────────────────────────────
        score = self._sortino(pnl_is)          # 标量 Tensor

        # ── 6. 换手率惩罚 ─────────────────────────────────────────
        if turnover.mean() > 0.5:
            score = score - 1.0

        # ── 7. OOS 均值收益 ──────────────────────────────────────
        mean_oos = pnl_oos.mean().item()

        return score, mean_oos
