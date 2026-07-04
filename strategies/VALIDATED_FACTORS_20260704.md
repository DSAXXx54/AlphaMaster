# 已验证有效因子记录

> 生成时间：2026-07-04
> 训练日期：2026-07-03 ~ 2026-07-04（隔夜训练）
> 训练模式：分组训练（forex / metals_comm / index 三组），CPU
> 训练步数：目标 3000 步，实际因熵坍塌 Early Stop（forex=1207, metals_comm=845, index=913）
> 回测模式：离线缓存（D:\K线数据），T=3498 bars H1（10品种时间轴对齐）
> 数据周期：约 1.5 年历史数据
> vocab_version：v6ff4c52e30c1

---

## 一、有效性判定标准

本次记录以如下标准判定"有效"：

- **✅ 确定有效**：Sharpe > 1 且 PnL > 0，回测结果稳健
- **⚠️ 部分有效**：同组内部分品种有效，存在泛化问题
- **❌ 无效**：整体亏损或零交易

---

## 二、本次回测汇总（2026-07-04）

| 品种 | 组 | PnL | Sharpe | Sortino | MaxDD | Calmar | 交易数 | 胜率 | 平均持仓 | 判定 |
|------|-----|-----|--------|---------|-------|--------|--------|------|----------|------|
| EURUSD | forex | -0.019 | -0.29 | -0.37 | 0.066 | -0.51 | 288 | 50.0% | 12.1h | ❌ |
| USDJPY | forex | +0.103 | +1.32 | +1.78 | 0.097 | +1.90 | 286 | 51.7% | 12.2h | ✅ |
| XAUUSD | metals_comm | -0.376 | -1.64 | -2.11 | 0.671 | -1.00 | 5 | 40.0% | 699h | ❌ |
| AAVUSD | metals_comm | +1.318 | +1.57 | +2.16 | 1.031 | +2.28 | 5 | 100% | 699h | ⚠️ |
| COCOA.c | metals_comm | +0.800 | +1.71 | +2.51 | 0.618 | +2.31 | 3 | 100% | 1166h | ⚠️ |
| US30.cash | index | 0.000 | 0.00 | 0.00 | 0.000 | 0.00 | 0 | — | — | ❌ |
| US100.cash | index | +0.300 | +1.47 | +1.75 | 0.296 | +1.81 | 3 | 100% | 1166h | ⚠️ |
| US500.cash | index | +0.201 | +1.26 | +1.48 | 0.236 | +1.52 | 3 | 100% | 1166h | ⚠️ |
| US2000.cash | index | -0.253 | -1.15 | -1.41 | 0.568 | -0.79 | 3 | 66.7% | 1166h | ❌ |
| JP225.cash | index | +0.563 | +2.40 | +3.01 | 0.226 | +4.45 | 3 | 66.7% | 1166h | ⚠️ |
| **Portfolio** | — | **+0.264** | **+2.84** | **+4.22** | **0.088** | **+5.37** | — | — | — | ✅ |

正收益品种：6/10 | Sharpe > 1 品种：6/10

---

## 三、各组公式详情

### 3.1 Forex 组（⚠️ 部分有效）

**公式**
```
RS_VOL -> DONCHIAN_POS_20 -> TS_STD_10 -> TS_RANK_5 -> TS_CORR_10 -> TANH_SQUASH -> MOMENTUM_10 -> TANH_SQUASH
```
Token 序列：`[40, 51, 81, 83, 86, 125, 88, 125]`

**训练结果**：Best Score = 6.636，Early Stop（step 1207/3000，熵坍塌触发上限）

**回测表现**：
- USDJPY：✅ Sharpe +1.32，Sortino +1.78，288 笔，胜率 51.7%，平均持仓 12h
- EURUSD：❌ Sharpe -0.29，小幅亏损

**公式逻辑**：
1. `RS_VOL`：相对波动率（品种波动率 / 截面均值），识别当前品种是否处于高波动状态
2. `DONCHIAN_POS_20`：价格在 20 期唐奇安通道内的相对位置，值域 [0,1]，近期高点附近=1，低点附近=0
3. `TS_STD_10`：10 期时序标准差，衡量近期信号的变化幅度
4. `TS_RANK_5`：5 期时序排名，捕捉短期动量方向
5. `TS_CORR_10`：10 期相关系数，与截面均值的相关性
6. `TANH_SQUASH`：将值压缩到 (-1, 1)，防止极端值
7. `MOMENTUM_10`：10 期动量（对数收益率差）
8. `TANH_SQUASH`：再次压缩，最终输出平滑信号

**本质**：波动率感知 + 通道位置 + 短期动量的组合信号，双重 tanh 压缩平滑。

**与历史 forex 公式对比**：
| 维度 | 历史公式（2026-07-03） | 本次公式（2026-07-04） |
|------|----------------------|----------------------|
| 公式 | `MA_DIFF -> TS_MIN_10 -> NEG -> ...` | `RS_VOL -> DONCHIAN_POS_20 -> ...` |
| EURUSD Sharpe | +1.82 | **-0.29（退步）** |
| USDJPY Sharpe | +1.63 | +1.32（相近） |
| 交易频率 | 83/77 笔 | 288/286 笔（高频化） |
| 持仓时长 | 57-62h | **12h（大幅缩短）** |

⚠️ **结论**：本次公式 USDJPY 仍有效，但 EURUSD 失效，高频化（12h 持仓）可能导致被点差侵蚀。历史 forex 公式（四重平滑）总体优于本次，建议继续沿用历史公式。

---

### 3.2 Metals_comm 组（⚠️ 分化严重）

**公式**
```
STOCH_K_14 -> KELTNER_POS_20 -> SUB -> WILLR_14 -> KELTNER_POS_20 -> DMI_DIFF_14 -> LT -> GATE
```
Token 序列：`[45, 52, 66, 26, 52, 49, 127, 72]`

**训练结果**：Best Score = 8.183，Early Stop（step 845/3000）

**回测表现**：
- AAVUSD：⚠️ Sharpe +1.57，5 笔交易，100% 胜率，持仓 699h（~29天）
- COCOA.c：⚠️ Sharpe +1.71，3 笔交易，100% 胜率，持仓 1166h（~49天）
- XAUUSD：❌ Sharpe -1.64，大幅亏损

**公式逻辑（RPN 执行）**：
```
[1] PUSH STOCH_K_14          → 随机指标K线（超买超卖）
[2] KELTNER_POS_20(STOCH_K) → 凯尔特纳通道位置
[3] PUSH WILLR_14             → 威廉指标（值域 [-1,0]）
[4] KELTNER_POS_20(WILLR_14) → 威廉指标的通道位置
[5] PUSH DMI_DIFF_14          → 方向运动指标差值（+DI - -DI）
[6] LT(WILLR_keltner, DMI)   → 二元比较（WILLR通道位置 < DMI差值？）
[7] SUB(STOCH_keltner, WILLR_keltner) → 差值
    GATE(cond=SUB, x=LT, y=?) → 条件选择
```

> 注：实际 RPN 栈执行顺序较复杂，GATE 弹出最近 3 个值。

**本质**：技术指标组合（随机指标 + 威廉指标 + 方向运动指标）的阈值突破条件门控策略。

**分化原因**：
- XAUUSD 亏损而 AAVUSD/COCOA.c 盈利，说明公式对黄金的价格结构不适配
- 交易次数极少（3~5 笔），100% 胜率在小样本下统计意义不大
- 持仓 700~1200 小时实为超长期趋势跟踪，不是真正的"信号策略"
- **⚠️ 小样本警告**：3~5 笔交易不能作为有效性的可靠证据

---

### 3.3 Index 组（⚠️ 部分有效，泛化不足）

**公式**
```
VOL_REGIME -> TS_MAX_10 -> DELTA -> TS_MIN_10 -> PRESSURE -> MOMENTUM_5 -> WILLR_14 -> GATE
```
Token 序列：`[8, 89, 99, 90, 12, 87, 26, 72]`

**训练结果**：Best Score = 8.094，Early Stop（step 913/3000）

**回测表现**：
- JP225.cash：⚠️ Sharpe +2.40，3 笔，66.7% 胜率，持仓 1166h
- US100.cash：⚠️ Sharpe +1.47，3 笔，100% 胜率，持仓 1166h
- US500.cash：⚠️ Sharpe +1.26，3 笔，100% 胜率，持仓 1166h
- US2000.cash：❌ Sharpe -1.15，亏损
- US30.cash：❌ **零交易**（见下方根因分析）

**US30.cash 零交易根因**：

GATE 的 condition = `TS_MIN_10(DELTA(TS_MAX_10(VOL_REGIME)))` 要 > 0，要求：
> 过去 10 根 bar 内，VOL_REGIME 的 10 期滚动最大值**全部严格递增**

这是极苛刻的条件。大部分时间 DELTA=0（滚动最大在台阶期不变），导致 condition ≤ 0，GATE 全程选择 y 分支 = `WILLR_14`。

`WILLR_14` 值域 [-1, 0]（永远非正），截面 z-score 归一化后 US30 的 WILLR_14 变化幅度不足，归一化后长期落在 neutral band (-0.3, 0) 内，始终不触发 ±0.3 的入场阈值。US100/US500 的波动率结构恰好在几个时点突破了该区间，产生了 3 笔交易。

**本质**：US30 数据无问题，是公式在该品种特征分布上泛化失败。

**⚠️ 小样本警告**：JP225/US100/US500 各仅 3 笔交易（其中 2 笔重叠），100% 胜率不可靠。

---

## 四、综合结论

### 4.1 跨时间段验证汇总

| 公式组 | 2026-07-03 回测 | 2026-07-04 回测 | 综合判定 |
|--------|----------------|----------------|---------|
| forex（旧公式） | EURUSD +1.82, USDJPY +1.63 | — | ✅ **确定有效** |
| forex（本次公式） | — | USDJPY +1.32, EURUSD -0.29 | ⚠️ **部分有效** |
| metals_comm（旧公式） | COCOA.c +2.54 | — | ⚠️ 不稳定 |
| metals_comm（本次公式） | — | AAVUSD +1.57, COCOA.c +1.71, XAUUSD -1.64 | ⚠️ 分化严重 |
| index（旧公式） | 全亏 | — | ❌ 无效 |
| index（本次公式） | — | JP225 +2.40, US100 +1.47, US500 +1.26, US30 零交易 | ⚠️ 部分有效 |

### 4.2 唯一确定有效的公式

**2026-07-03 Forex 公式**（历史最优，两次回测验证）：
```
MA_DIFF -> TS_MIN_10 -> NEG -> TS_MAX_10 -> TS_MEAN_10 -> TS_MEAN_20 -> TS_MEAN_10 -> TS_MEAN_20
```
Token：`[3, 45, 24, 44, 33, 34, 33, 34]`（vocab_version: 2.0）

- EURUSD：Sharpe +1.82，Sortino +2.41，MaxDD 0.071，83 笔，胜率 57.8%
- USDJPY：Sharpe +1.63，Sortino +2.08，MaxDD 0.097，77 笔，胜率 57.1%
- 两次回测均正收益，跨品种一致，逻辑可解释

### 4.3 本次训练的主要问题

1. **Early Stop 过早**：三组均在 28%~40% 进度时因熵坍塌退出，未达到 3000 步目标
2. **index/metals_comm 组交易数极少**（3~5 笔），持仓超长（700~1200h），公式退化为稀疏趋势信号，统计可靠性极低
3. **US30.cash 零交易**：公式泛化失败，条件链在该品种上永不激活
4. **XAUUSD 亏损**：metals_comm 组共用公式，在黄金上反向运作

---

## 五、下一步建议

| 优先级 | 行动 | 说明 |
|--------|------|------|
| P0 | 沿用 forex 旧公式（2026-07-03）用于 EURUSD/USDJPY | 唯一经过两次独立验证的有效因子 |
| P1 | 解决 Early Stop 问题后重训 index 组 | 调大 `ENTROPY_COLLAPSE_STEPS`（15→30）或 `MAX_RESTARTS`（8→15）|
| P1 | index 组改为品种独立训练 | `--single US30.cash` 等，避免共用公式掩盖品种特异性 |
| P2 | metals_comm 公式需更多数据验证 | 3~5 笔样本量不足，需至少 30 笔以上才可信 |
| P3 | 清理 vocab_version 不匹配问题 | 旧公式 vocab_version=2.0，新公式=v6ff4c52e30c1，需统一 |

---

## 六、文件索引

| 文件 | 说明 |
|------|------|
| `strategies/best_group_forex.json` | 本次 forex 组公式（score=6.636） |
| `strategies/best_group_metals_comm.json` | 本次 metals_comm 组公式（score=8.183） |
| `strategies/best_group_index.json` | 本次 index 组公式（score=8.094） |
| `strategies/best_EURUSD.json` 等各品种 | 本次各品种策略文件 |
| `backtest_output/multi_factor_report.json` | 本次完整回测 JSON 报告 |
| `backtest_output/portfolio_equity.png` | 本次组合资金曲线图 |
| `strategies/FOREX_FACTOR_REPORT.md` | 历史 forex 组分析报告（2026-07-03） |
| `strategies/EFFECTIVE_FACTORS_ANALYSIS.md` | 历史综合因子分析（2026-07-03） |
| `training_history_forex.json` | 本次 forex 训练历史（1207 步） |
| `training_history_index.json` | 本次 index 训练历史（913 步） |
| `training_history_metals_comm.json` | 本次 metals_comm 训练历史（845 步） |
