"""
main.py — 训练入口

使用方法：
    python main.py

流程：
    1. 通过 MT5DataFetcher 上下文管理器连接 MT5 终端
    2. 初始化 MT5DataManager，加载并对齐多品种 OHLCV 历史数据
    3. 初始化 AlphaEngine（注入 data_manager）
    4. 启动 AlphaGPT 强化学习训练，搜索最优 Alpha 因子公式
    5. 训练完成后将最佳公式保存至 best_mt5_strategy.json

Requirements: 5.6, 11.1
"""

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.engine import AlphaEngine

if __name__ == "__main__":
    with MT5DataFetcher() as fetcher:
        data_mgr = MT5DataManager(fetcher)
        data_mgr.load()
        engine = AlphaEngine(data_manager=data_mgr)
        engine.train()
