"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 训练参数 ──────────────────────────────────────────────────────────
    BATCH_SIZE      = 64
    TRAIN_STEPS     = 5000
    MAX_FORMULA_LEN = 12

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控而非线性加权 ────────────────────
    # 多目标 reward = REWARD_ALPHA * sortino + IC_GATE_BONUS（若 IC > 0）
    # 去掉原先 β*IC + γ*ICIR 的线性项，改为：
    #   IC > IC_GATE_THRESH  → reward * IC_GATE_MULT（正向奖励）
    #   IC < -IC_GATE_THRESH → reward * IC_NEG_MULT  （反向惩罚）
    # 量纲无关，不受 IC 绝对值偏小的影响
    REWARD_ALPHA:      float = 1.0    # Sortino 权重（IC 已剥离为独立门控，不再分摊权重）
    IC_GATE_THRESH:    float = 0.005  # IC 绝对值超此阈值才触发门控（过滤噪声）
    IC_GATE_MULT:      float = 1.15   # IC > thresh 时奖励乘数（+15%）
    IC_NEG_MULT:       float = 0.85   # IC < -thresh 时惩罚乘数（-15%）

    # ── 熵保护：更早、更强地阻止策略坍塌 ─────────────────────────────
    # 原参数：系数 0.15/(1+H)，在 H=0.3 时系数≈0.115（太弱）
    # 新参数：系数 ENTROPY_COEFF_MAX/(1+H)^ENTROPY_COEFF_POWER
    #   H=3.0 → coeff≈0.090  （探索期，轻触）
    #   H=1.0 → coeff≈0.250  （收敛中期，中等）
    #   H=0.3 → coeff≈0.385  （坍塌边缘，强保护）
    ENTROPY_COEFF_MAX:   float = 0.50   # 最大 entropy bonus 系数
    ENTROPY_COEFF_POWER: float = 1.3    # 指数，控制系数随熵衰减的曲率
    ENTROPY_COLLAPSE_THRESH: float = 1.0  # 低于此熵值判定为"危险区"（原 0.3 过低）
    ENTROPY_COLLAPSE_STEPS:  int   = 10   # 连续低于阈值多少步才触发重启（原 15）

    # ── Elite Replay：把历史最优公式重放进每步 batch ──────────────────
    # 每步 batch 中预留 ELITE_REPLAY_FRAC 的比例用于重放精英公式
    # 这些公式的 log_prob 会直接参与 REINFORCE 更新，
    # 让梯度持续朝"已知好区域"引导，对抗 on-policy 遗忘
    ELITE_REPLAY_FRAC:  float = 0.10   # 精英占 batch 比例（10% = 约 6 个样本）
    ELITE_POOL_SIZE:    int   = 20     # 保留历史最优公式的数量（val_score 前 K 名）
    ELITE_REWARD_SCALE: float = 1.0    # 重放精英的 reward 缩放（1.0=不额外加权）

    # ── 坍塌重启：更激进的恢复策略 ────────────────────────────────────
    MAX_RESTARTS:   int   = 8     # 最大重启次数（原 3）
    RESTART_NOISE:  float = 0.05  # 重启时 FFN 扰动强度（原 0.02）

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 10
    CORR_THRESHOLD:   float = 0.7
    CORR_PENALTY:     float = 0.5

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    WF_GAP: int = 20
