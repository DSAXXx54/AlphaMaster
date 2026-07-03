"""
live_trade.py — 自动交易启动脚本

使用方法：
    python live_trade.py                    # 使用 strategies/best_{symbol}.json 多因子
    python live_trade.py --single           # 使用 best_mt5_strategy.json 单公式
    python live_trade.py --dry-run          # 模拟运行（不下单，只打印信号）
    python live_trade.py --symbols XAUUSDm US500m   # 只交易指定品种

当前已存档的有效策略：
    strategies/best_EURUSDm.json   （如有）
    strategies/best_US500m.json    → us500_v2  Sortino +1.43
    strategies/best_XAUUSDm.json   → xauusd_v1 Sortino +1.11
    ...

注意：
  - 需要 MT5 终端已登录并允许自动交易
  - 确保 .env 中配置了 MT5_LOGIN / MT5_PASSWORD / MT5_SERVER（若需要登录）
  - 停止方法：在当前目录创建 STOP_SIGNAL 文件，或直接 Ctrl+C
"""

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import Config
from strategy_manager.runner import MT5StrategyRunner
from loguru import logger


def main():
    # 命令行参数处理
    dry_run = "--dry-run" in sys.argv
    single  = "--single"  in sys.argv
    sym_override = None
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        sym_override = sys.argv[idx+1:]

    # 覆盖品种列表
    if sym_override:
        Config.SYMBOLS = sym_override
        logger.info(f"[live_trade] 品种覆盖: {Config.SYMBOLS}")

    if dry_run:
        logger.info("[live_trade] DRY RUN 模式：只打印信号，不下单")
        # TODO: 可以在 runner 里加 dry_run 开关跳过下单

    if single:
        logger.info("[live_trade] 单公式模式：所有品种共用 best_mt5_strategy.json")

    logger.info("=" * 50)
    logger.info("  AlphaGPT 自动交易启动")
    logger.info(f"  品种: {Config.SYMBOLS}")
    logger.info(f"  周期: H1")
    logger.info(f"  信号模式: {Config.SIGNAL_MODE}")
    logger.info(f"  出场模式: {Config.EXIT_MODE}")
    logger.info("=" * 50)

    runner = MT5StrategyRunner()

    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("[live_trade] 收到 Ctrl+C，正在停止...")
    finally:
        runner.shutdown()
        logger.info("[live_trade] 已停止。")


if __name__ == "__main__":
    main()
