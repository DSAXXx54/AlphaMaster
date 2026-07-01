# Feature: mt5-alphagpt-refactor, Property 12: 策略信号触发买单
"""
Property-based tests for strategy_manager.runner (MT5StrategyRunner).

Property 12 Validates: Requirements 10.4

For any signal score tensor:
  - If score > Config.BUY_THRESHOLD AND symbol not in portfolio.positions
    → trader.buy() MUST be called for that symbol
  - If score <= Config.BUY_THRESHOLD
    → trader.buy() MUST NOT be called for that symbol
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from config import Config
from strategy_manager.runner import MT5StrategyRunner
from strategy_manager.portfolio import MT5PortfolioManager, Position


# ── Helpers ───────────────────────────────────────────────────────────────────

# MT5 BUY_THRESHOLD constant (0.70)
_THRESHOLD = Config.BUY_THRESHOLD


def _make_runner(
    symbols: List[str],
    held_symbols: List[str],
) -> MT5StrategyRunner:
    """
    Construct an MT5StrategyRunner via __new__ (bypass __init__ to avoid
    sys.exit(1) on missing strategy file) and inject all required attributes.

    Args:
        symbols:      List of symbol names aligned with the scores tensor.
        held_symbols: Subset of symbols already in the portfolio (no buy expected).

    Returns:
        A fully wired MT5StrategyRunner instance with mock trader / portfolio.
    """
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)

    # Minimal formula (not actually executed in _scan_for_entries)
    runner.formula = [1, 2, 3]

    # ── Mock trader ──────────────────────────────────────────────────────────
    mock_trader = MagicMock()
    mock_account = {"equity": 10_000.0, "margin_free": 5_000.0}
    mock_trader.get_account_info.return_value = mock_account
    mock_trader.buy.return_value = True
    runner.trader = mock_trader

    # ── Mock portfolio (with pre-held positions) ─────────────────────────────
    mock_portfolio = MagicMock(spec=MT5PortfolioManager)
    mock_portfolio.positions = {sym: MagicMock() for sym in held_symbols}
    mock_portfolio.get_open_count.return_value = len(held_symbols)
    runner.portfolio = mock_portfolio

    # ── Mock risk engine ─────────────────────────────────────────────────────
    mock_risk = MagicMock()
    mock_risk.calculate_lot.return_value = 0.01  # always valid lot
    runner.risk = mock_risk

    # ── Mock data manager ────────────────────────────────────────────────────
    mock_data_manager = MagicMock()
    mock_data_manager.symbols = symbols
    runner._data_manager = mock_data_manager

    # ── Other attributes used by _scan_for_entries ────────────────────────────
    runner._last_refresh = 0.0

    return runner


# ── Hypothesis strategies ─────────────────────────────────────────────────────

# Valid MT5 symbol strings: uppercase alphabetic, 3-8 characters
symbol_strategy = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=3,
    max_size=8,
)

# Scores clearly above threshold (to avoid floating-point boundary ambiguity)
above_threshold_strategy = st.floats(
    min_value=_THRESHOLD + 1e-6,
    max_value=1.0 - 1e-9,
    allow_nan=False,
    allow_infinity=False,
)

# Scores clearly at or below threshold
below_threshold_strategy = st.floats(
    min_value=1e-9,
    max_value=_THRESHOLD,
    allow_nan=False,
    allow_infinity=False,
)


@composite
def score_scenario_strategy(draw):
    """
    Draw a scenario with:
    - n symbols (1–5, unique)
    - for each symbol: a score and whether the symbol is already held

    Returns a dict with keys:
        symbols:        list[str]
        scores:         list[float]
        held_symbols:   list[str]  (already in portfolio)
    """
    n = draw(st.integers(min_value=1, max_value=5))
    symbols = draw(
        st.lists(symbol_strategy, min_size=n, max_size=n, unique=True)
    )

    scores = []
    held_symbols = []

    for sym in symbols:
        # Randomly choose above or below threshold
        use_above = draw(st.booleans())
        if use_above:
            score = draw(above_threshold_strategy)
        else:
            score = draw(below_threshold_strategy)
        scores.append(score)

        # Randomly decide if this symbol is already held
        is_held = draw(st.booleans())
        if is_held:
            held_symbols.append(sym)

    return {
        "symbols": symbols,
        "scores": scores,
        "held_symbols": held_symbols,
    }


# ── Property 12: 策略信号触发买单 ─────────────────────────────────────────────
# Validates: Requirements 10.4


@settings(max_examples=100, deadline=None)
@given(scenario=score_scenario_strategy())
def test_property12_buy_signal_triggers_buy(scenario: dict):
    """
    Property 12: score > BUY_THRESHOLD 且不在仓时必须调用 buy()；
                 score ≤ 阈值时不调用

    For any signal score tensor passed to MT5StrategyRunner._scan_for_entries():
    - WHEN score > Config.BUY_THRESHOLD AND symbol not currently held:
        MT5Trader.buy() MUST be called with that symbol.
    - WHEN score <= Config.BUY_THRESHOLD:
        MT5Trader.buy() MUST NOT be called for that symbol.
    - WHEN symbol is already in portfolio.positions:
        MT5Trader.buy() MUST NOT be called for that symbol,
        regardless of score.

    Validates: Requirements 10.4
    """
    symbols: List[str] = scenario["symbols"]
    raw_scores: List[float] = scenario["scores"]
    held_symbols: List[str] = scenario["held_symbols"]

    scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)

    # Patch MT5PriceFeed.get_tick so _scan_for_entries doesn't touch real MT5
    mock_tick = {"bid": 1.0, "ask": 1.0, "mid": 1.0}
    with patch("strategy_manager.runner.MT5PriceFeed") as mock_feed_cls:
        mock_feed_cls.get_tick.return_value = mock_tick

        runner = _make_runner(symbols, held_symbols)

        # Ensure MAX_OPEN_POSITIONS is large enough that it never blocks buying
        original_max = Config.MAX_OPEN_POSITIONS
        Config.MAX_OPEN_POSITIONS = len(symbols) + 10

        try:
            runner._scan_for_entries(scores_tensor)
        finally:
            Config.MAX_OPEN_POSITIONS = original_max

    # ── Collect actually-called buy symbols ──────────────────────────────────
    buy_calls = runner.trader.buy.call_args_list
    called_symbols = {call.args[0] for call in buy_calls}

    # ── Assert for each symbol ───────────────────────────────────────────────
    for idx, sym in enumerate(symbols):
        score = raw_scores[idx]
        is_held = sym in held_symbols
        should_buy = (score > _THRESHOLD) and (not is_held)

        if should_buy:
            assert sym in called_symbols, (
                f"buy() was NOT called for '{sym}' "
                f"(score={score:.6f} > threshold={_THRESHOLD}, not held), "
                f"but it SHOULD have been. Called symbols: {called_symbols}"
            )
        else:
            assert sym not in called_symbols, (
                f"buy() WAS called for '{sym}' unexpectedly "
                f"(score={score:.6f}, threshold={_THRESHOLD}, held={is_held}). "
                f"Called symbols: {called_symbols}"
            )
