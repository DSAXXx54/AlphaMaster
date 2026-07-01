"""
model_core/features.py — MT5 技术指标特征工程

MT5FeatureEngineer.compute_features(raw_dict) 计算十个技术特征：
  RET, RET5, VOL_RATIO, PRESSURE, DEV, HL_RANGE, ATR, RVOL, RET20, AC1

全部经相应归一化策略处理后，NaN/Inf 替换为 0。

Requirements: F1.1–F1.9, F4.1–F4.6
"""
import torch


class MT5FeatureEngineer:
    """
    MT5 特征工程器。

    compute_features(raw_dict) 接受含 OHLCV 张量的字典，
    返回形状为 [N, 10, T] 的特征张量。
    """

    INPUT_DIM = 10  # 与 Config.INPUT_DIM 一致

    # ── 特征顺序与索引 ──────────────────────────────────────
    # 0: RET       log(close[t] / close[t-1])
    # 1: RET5      log(close[t] / close[t-5])
    # 2: VOL_RATIO volume[t] / MA20(volume)[t]
    # 3: PRESSURE  (close[t] - open[t]) / (high[t] - low[t] + ε)
    # 4: DEV       (close[t] - MA20(close)[t]) / MA20(close)[t]
    # 5: HL_RANGE  (high[t] - low[t]) / close[t]
    # 6: ATR       平均真实波幅（window=14）
    # 7: RVOL      已实现波动率（window=20）
    # 8: RET20     20 周期对数动量
    # 9: AC1       因果一阶自相关（window=20）

    # 归一化 clip 边界
    _CLIP_BOUND: float = 5.0
    # 数值稳定 epsilon
    _EPS: float = 1e-9
    # MA 窗口
    _MA_WINDOW: int = 20

    @staticmethod
    def _rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
        """
        沿 T 维度（dim=1）计算指定窗口大小的简单移动均值（因果）。

        输入 x:      [N, T]
        window:      滑动窗口大小（≥ 1）
        输出:        [N, T]，前 window-1 个位置用可用样本均值填充（causal）
        """
        N, T = x.shape
        # 在 T 维度左端补零，使 unfold 后每个位置对应历史窗口
        pad = torch.zeros(N, window - 1, dtype=x.dtype, device=x.device)
        x_pad = torch.cat([pad, x], dim=1)  # [N, T + window - 1]
        # unfold: [N, T, window]
        unfolded = x_pad.unfold(1, window, 1)
        return unfolded.mean(dim=-1)  # [N, T]

    @classmethod
    def _ma20(cls, x: torch.Tensor) -> torch.Tensor:
        """
        兼容别名：窗口为 20 的简单移动均值（causal）。

        等价于 `_rolling_mean(x, _MA_WINDOW)`。
        输入 x: [N, T]
        输出:   [N, T]
        """
        return cls._rolling_mean(x, cls._MA_WINDOW)

    @staticmethod
    def _atr(close: torch.Tensor, high: torch.Tensor, low: torch.Tensor, window: int = 14) -> torch.Tensor:
        """
        平均真实波幅（Average True Range），窗口默认 14（因果）。

        prev_close: 右移一位，第 0 位用 close[:,0] 填充。
        TR = max(high-low, |high-prev_close|, |low-prev_close|) 逐元素最大值。
        ATR = _rolling_mean(TR, window)

        输入:
            close : [N, T]  收盘价（正值）
            high  : [N, T]  最高价
            low   : [N, T]  最低价
            window: 滑动窗口大小，默认 14
        输出: [N, T]，非负
        需求：F1.3
        """
        # prev_close: 因果右移，第 0 位用 close[:,0] 填充
        prev_close = torch.cat([close[:, :1], close[:, :-1]], dim=1)  # [N, T]
        tr = torch.stack([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], dim=-1).max(dim=-1).values  # [N, T]
        return MT5FeatureEngineer._rolling_mean(tr, window)

    @staticmethod
    def _rvol(close: torch.Tensor, window: int = 20) -> torch.Tensor:
        """
        已实现波动率（Realized Volatility），window=20，ddof=0。

        计算逐步对数收益率的滚动标准差，结果加 1e-9 保证严格正值。

        输入 close: [N, T]（正值）
        输出:       [N, T]，所有值 ≥ 1e-9

        需求：F1.4
        """
        eps = MT5FeatureEngineer._EPS
        ret = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = torch.cat([torch.zeros_like(close[:, :1]), ret], dim=1)  # [N, T]
        N, T = ret.shape
        pad = torch.zeros(N, window - 1, device=ret.device, dtype=ret.dtype)
        unfolded = torch.cat([pad, ret], dim=1).unfold(1, window, 1)  # [N, T, W]
        mean_ = unfolded.mean(dim=-1, keepdim=True)
        rvol = ((unfolded - mean_) ** 2).mean(dim=-1).sqrt() + 1e-9  # [N, T]
        return rvol

    @staticmethod
    def _ret20(close: torch.Tensor) -> torch.Tensor:
        """
        20 周期对数动量。

        输出前 20 个位置补 0，第 20 位起等于 log(close[t]/close[t-20]+ε)。

        输入 close: [N, T]
        输出:       [N, T]
        需求：F1.5
        """
        N, T = close.shape
        eps = MT5FeatureEngineer._EPS
        ret20 = torch.log(close[:, 20:] / (close[:, :-20] + eps))  # [N, T-20]
        pad = torch.zeros(N, 20, device=close.device, dtype=close.dtype)
        return torch.cat([pad, ret20], dim=1)  # [N, T]

    @staticmethod
    def _ac1(close: torch.Tensor, window: int = 20) -> torch.Tensor:
        """
        因果一阶自相关（lag-1 autocorrelation），窗口默认 20。

        对每个时间步 t，计算 ret[t-window:t-1] 与 ret[t-window+1:t] 的
        Pearson 相关系数。不足 window 个样本的位置（unfold 补零区）输出 0，
        常数窗口（std=0）输出 0，结果 NaN/Inf 替换为 0。

        输入 close: [N, T]
        输出:       [N, T]，值域 ∈ [-1, 1]
        需求：F1.6
        """
        N, T = close.shape
        eps = MT5FeatureEngineer._EPS
        ret = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = torch.cat([torch.zeros_like(close[:, :1]), ret], dim=1)  # [N, T]
        pad = torch.zeros(N, window, device=ret.device, dtype=ret.dtype)
        unfolded = torch.cat([pad, ret], dim=1).unfold(1, window + 1, 1)  # [N, T, window+1]
        x = unfolded[:, :, :-1]  # ret[t-window:t-1], [N, T, window]
        y = unfolded[:, :, 1:]   # ret[t-window+1:t], [N, T, window]
        x_m = x.mean(dim=-1, keepdim=True)
        y_m = y.mean(dim=-1, keepdim=True)
        cov = ((x - x_m) * (y - y_m)).mean(dim=-1)
        std_x = ((x - x_m) ** 2).mean(dim=-1).sqrt()
        std_y = ((y - y_m) ** 2).mean(dim=-1).sqrt()
        ac1 = cov / (std_x * std_y + 1e-8)  # [N, T]
        return torch.nan_to_num(ac1, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _robust_norm(x: torch.Tensor) -> torch.Tensor:
        """
        MAD robust 归一化 + clip 到 [-5, 5]。

        x: [N, T]
        """
        # 沿 T 维度计算 median 和 MAD
        median = x.median(dim=1, keepdim=True).values          # [N, 1]
        mad = (x - median).abs().median(dim=1, keepdim=True).values + 1e-6  # [N, 1]
        normed = (x - median) / mad
        return torch.clamp(normed, -MT5FeatureEngineer._CLIP_BOUND, MT5FeatureEngineer._CLIP_BOUND)

    @staticmethod
    def _clean(x: torch.Tensor) -> torch.Tensor:
        """将 NaN 和 ±Inf 替换为 0（Req 4.5）。"""
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def compute_features(raw_dict: dict) -> torch.Tensor:
        """
        计算 MT5 十个技术特征。

        Parameters
        ----------
        raw_dict : dict
            键为 "open", "high", "low", "close", "volume"，
            每个值为形状 [N, T] 的 torch.Tensor（正值，float）。

        Returns
        -------
        torch.Tensor
            形状 [N, 10, T] 的特征张量，所有值在 [-5, 5] 内，无 NaN/Inf。
            特征顺序严格对应 FEATURE_NAMES 索引 0~9：
            RET, RET5, VOL_RATIO, PRESSURE, DEV, HL_RANGE,
            ATR, RVOL, RET20, AC1
        """
        close  = raw_dict["close"].float()   # [N, T]
        open_  = raw_dict["open"].float()    # [N, T]
        high   = raw_dict["high"].float()    # [N, T]
        low    = raw_dict["low"].float()     # [N, T]
        volume = raw_dict["volume"].float()  # [N, T]

        N, T = close.shape
        eps = MT5FeatureEngineer._EPS

        # ── Feature 0: RET ──────────────────────────────────
        # log(close[t] / close[t-1])，第一个位置补 0
        ret_raw = torch.log(close[:, 1:] / (close[:, :-1] + eps))  # [N, T-1]
        ret_pad = torch.zeros(N, 1, dtype=close.dtype, device=close.device)
        ret = torch.cat([ret_pad, ret_raw], dim=1)                   # [N, T]
        ret = MT5FeatureEngineer._clean(ret)
        ret = MT5FeatureEngineer._robust_norm(ret)
        ret = MT5FeatureEngineer._clean(ret)

        # ── Feature 1: RET5 ─────────────────────────────────
        # log(close[t] / close[t-5])，前 5 个位置补 0
        ret5_raw = torch.log(close[:, 5:] / (close[:, :-5] + eps))  # [N, T-5]
        ret5_pad = torch.zeros(N, 5, dtype=close.dtype, device=close.device)
        ret5 = torch.cat([ret5_pad, ret5_raw], dim=1)                 # [N, T]
        ret5 = MT5FeatureEngineer._clean(ret5)
        ret5 = MT5FeatureEngineer._robust_norm(ret5)
        ret5 = MT5FeatureEngineer._clean(ret5)

        # ── Feature 2: VOL_RATIO ────────────────────────────
        ma20_vol = MT5FeatureEngineer._ma20(volume)                   # [N, T]
        vol_ratio = volume / (ma20_vol + eps)                         # [N, T]
        vol_ratio = MT5FeatureEngineer._clean(vol_ratio)
        vol_ratio = MT5FeatureEngineer._robust_norm(vol_ratio)
        vol_ratio = MT5FeatureEngineer._clean(vol_ratio)

        # ── Feature 3: PRESSURE ─────────────────────────────
        # (close - open) / (high - low + ε)，天然有界，直接 clamp[-1,1]（F4.1）
        pressure = (close - open_) / (high - low + eps)               # [N, T]
        pressure = MT5FeatureEngineer._clean(pressure)
        pressure = torch.clamp(pressure, -1.0, 1.0)

        # ── Feature 4: DEV ──────────────────────────────────
        # (close - MA20(close)) / MA20(close)
        ma20_close = MT5FeatureEngineer._ma20(close)                  # [N, T]
        dev = (close - ma20_close) / (ma20_close + eps)               # [N, T]
        dev = MT5FeatureEngineer._clean(dev)
        dev = MT5FeatureEngineer._robust_norm(dev)
        dev = MT5FeatureEngineer._clean(dev)

        # ── Feature 5: HL_RANGE ─────────────────────────────
        # (high - low) / close
        hl_range = (high - low) / (close + eps)                       # [N, T]
        hl_range = MT5FeatureEngineer._clean(hl_range)
        hl_range = MT5FeatureEngineer._robust_norm(hl_range)
        hl_range = MT5FeatureEngineer._clean(hl_range)

        # ── Feature 6: ATR — log1p → MAD norm → clean（F4.2）───
        atr_raw = MT5FeatureEngineer._atr(close, high, low)           # [N, T], ≥ 0
        atr_log = torch.log1p(atr_raw.clamp(min=0.0))                 # log1p 压缩长尾
        atr = MT5FeatureEngineer._robust_norm(atr_log)
        atr = MT5FeatureEngineer._clean(atr)

        # ── Feature 7: RVOL — log1p → MAD norm → clean（F4.2）─
        rvol_raw = MT5FeatureEngineer._rvol(close)                    # [N, T], ≥ 1e-9
        rvol_log = torch.log1p(rvol_raw.clamp(min=0.0))               # log1p 压缩长尾
        rvol = MT5FeatureEngineer._robust_norm(rvol_log)
        rvol = MT5FeatureEngineer._clean(rvol)

        # ── Feature 8: RET20 — MAD norm → clean（F4.3）─────────
        ret20_raw = MT5FeatureEngineer._ret20(close)                  # [N, T]
        ret20 = MT5FeatureEngineer._robust_norm(ret20_raw)
        ret20 = MT5FeatureEngineer._clean(ret20)

        # ── Feature 9: AC1 — clamp[-1,1] → clean（F4.4）────────
        ac1_raw = MT5FeatureEngineer._ac1(close)                      # [N, T], ∈ [-1, 1]
        ac1 = torch.clamp(ac1_raw, -1.0, 1.0)
        ac1 = MT5FeatureEngineer._clean(ac1)

        # ── Stack → [N, 10, T] ──────────────────────────────
        features = torch.stack(
            [ret, ret5, vol_ratio, pressure, dev, hl_range,
             atr, rvol, ret20, ac1],
            dim=1,
        )
        # Final safety pass: ensure no residual NaN/Inf after stacking
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    # ── 备用候选特征（F3 迁移，非当前活跃特征） ──────────────────────────
    # 以下两个方法来自原 AdvancedFactorEngineer，供未来特征扩充时直接启用。

    @staticmethod
    def volatility_clustering(close: torch.Tensor, window: int = 10) -> torch.Tensor:
        """波动率聚集指标。返回 [N, T]，非负。
        备用候选特征（非当前活跃），供未来扩充时启用。
        计算：sqrt(rolling_mean(ret^2, window))，因果滑动窗口。
        注意：迁移时将 torch.roll 替换为 cat+unfold，消除第 0 步的边界污染。
        需求：F3.1
        """
        eps = MT5FeatureEngineer._EPS
        ret = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = torch.cat([torch.zeros_like(close[:, :1]), ret], dim=1)
        ret_sq = ret ** 2
        vol_ma = MT5FeatureEngineer._rolling_mean(ret_sq, window)
        return torch.sqrt(vol_ma + eps)

    @staticmethod
    def relative_strength(
        close: torch.Tensor, high: torch.Tensor, low: torch.Tensor, window: int = 14
    ) -> torch.Tensor:
        """类 RSI 相对强弱指标。返回 [N, T]，范围 [-1, 1]。
        备用候选特征（非当前活跃），供未来扩充时启用。
        计算：(avg_gain - avg_loss) / (avg_gain + avg_loss + ε)。
        注意：迁移时将 torch.roll 替换为 cat+unfold，保证因果性。
        需求：F3.2
        """
        diff = close - torch.cat([close[:, :1], close[:, :-1]], dim=1)
        gains = torch.relu(diff)
        losses = torch.relu(-diff)
        avg_gain = MT5FeatureEngineer._rolling_mean(gains, window)
        avg_loss = MT5FeatureEngineer._rolling_mean(losses, window)
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return (rsi - 50.0) / 50.0  # 归一化到 [-1, 1]
