"""
main.py — 多因子训练入口

使用方法：
    python main.py                     # 每品种单独训练（默认，防过拟合）
    python main.py XAUUSDm             # 只训练指定品种
    python main.py --cross-section     # 截面模式：5品种一起训练，可用跨资产特征

单独训练 vs 截面训练：
  单独训练（默认）：每个品种找自己的最优公式，互相独立，适合差异大的品种
  截面训练：N=5 一起跑，REL_RET5/REL_RET20/REL_VOL 等跨资产特征才有意义，
            能找到"在5个品种里选强做多/选弱做空"的截面排名因子，
            结果保存到 strategies/best_cross_section.json
"""
import sys
import pathlib

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.single_symbol_manager import SingleSymbolDataManager
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig


def main(target_symbols: list[str] | None = None, cross_section: bool = False):
    """
    Args:
        target_symbols: 要训练的品种列表（单独模式）。None = 全部。
        cross_section:  True = 截面模式（所有品种一起），False = 逐品种单独训练。
    """
    print(f"{'='*60}")
    print(f"  AlphaGPT {'截面' if cross_section else '多因子'}训练")
    print(f"  TRAIN_STEPS={ModelConfig.TRAIN_STEPS}  "
          f"MAX_FORMULA_LEN={ModelConfig.MAX_FORMULA_LEN}  "
          f"BATCH_SIZE={ModelConfig.BATCH_SIZE}")
    print(f"{'='*60}")

    with MT5DataFetcher() as fetcher:
        multi_mgr = MT5DataManager(fetcher)
        multi_mgr.load()

        if cross_section:
            # ── 截面模式：所有品种一起训练 ───────────────────────────
            print(f"  截面训练品种: {multi_mgr.symbols}")
            print(f"  特征包含跨资产 REL_RET5/REL_RET20/REL_VOL，N={len(multi_mgr.symbols)}\n")

            engine = AlphaEngine(
                data_manager=multi_mgr,
                target_symbol=None,   # None = 截面模式，策略存 best_mt5_strategy.json
            )
            engine.train()

            print(f"\n{'─'*60}")
            print(f"  截面训练完成")
            print(f"  BestScore={engine.best_score:.4f}")
            print(f"  公式: {engine._decode_formula(engine.best_formula)}")
            print(f"  → 结果已保存到 strategies/best_cross_section.json")

            # 截面模式额外存一份带标识的文件
            import json
            from model_core.vocab import VOCAB_VERSION
            cs_path = pathlib.Path("strategies") / "best_cross_section.json"
            cs_path.parent.mkdir(parents=True, exist_ok=True)
            cs_path.write_text(json.dumps({
                "vocab_version": VOCAB_VERSION,
                "symbol":        "cross_section",
                "formula":       engine.best_formula,
                "best_score":    engine.best_score,
                "mode":          "cross_section_N5",
            }, indent=2))

        else:
            # ── 逐品种单独训练（原有逻辑）──────────────────────────
            symbols_to_train = target_symbols or multi_mgr.symbols
            print(f"  准备训练品种: {symbols_to_train}\n")

            results = {}
            for symbol in symbols_to_train:
                if symbol not in multi_mgr.symbols:
                    print(f"  [跳过] {symbol} 不在已加载数据中")
                    continue

                print(f"\n{'─'*60}")
                print(f"  开始训练: {symbol}")
                print(f"{'─'*60}")

                single_mgr = SingleSymbolDataManager(multi_mgr, symbol)
                engine = AlphaEngine(
                    data_manager=single_mgr,
                    target_symbol=symbol,
                )
                engine.train()

                results[symbol] = {
                    "best_score": engine.best_score,
                    "formula":    engine.best_formula,
                    "readable":   engine._decode_formula(engine.best_formula),
                }

            print(f"\n{'='*60}")
            print(f"  训练完成汇总")
            print(f"{'='*60}")
            for sym, r in results.items():
                print(f"  {sym:12s}  BestScore={r['best_score']:.4f}")
                print(f"             {r['readable']}")
                sp = pathlib.Path("strategies") / f"best_{sym}.json"
                print(f"             → {sp}")
            print()


if __name__ == "__main__":
    cross = "--cross-section" in sys.argv
    cli_symbols = [s for s in sys.argv[1:] if not s.startswith("--")] or None
    main(target_symbols=cli_symbols, cross_section=cross)
