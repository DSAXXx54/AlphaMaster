# Feature: mt5-alphagpt-refactor, Property 8: 订单方向构造正确性
# Feature: mt5-alphagpt-refactor, Property 9: 所有订单附带正确 Magic Number
"""
Property-based tests for execution.trader (MT5Trader).

Property 8 Validates: Requirements 6.2, 6.3
Property 9 Validates: Requirements 6.6
"""

import types
import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, strategies as st

from execution.trader import MT5Trader
from config import Config


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_ok_result():
    """Build a fake mt5.order_send() result with retcode == 10009 (success)."""
    result = MagicMock()
    result.retcode = 10009
    result.order = 12345
    return result


def _make_account_info():
    """Build a fake mt5.account_info() result."""
    info = MagicMock()
    info.equity = 10000.0
    info.margin_free = 5000.0
    return info


# ── Shared Hypothesis strategies ─────────────────────────────────────────────

# Valid MT5 symbol strings: alphanumeric, 2-12 characters
symbol_strategy = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
    min_size=2,
    max_size=12,
)

# Valid lot sizes: positive floats in realistic MT5 range
lot_strategy = st.floats(
    min_value=0.01,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)


# ── Property 8: 订单方向构造正确性 ────────────────────────────────────────────
# Validates: Requirements 6.2, 6.3


@settings(max_examples=100, deadline=None)
@given(
    symbol=symbol_strategy,
    lot=lot_strategy,
)
def test_property8_order_type_correctness(symbol: str, lot: float):
    """
    For any symbol/lot combination:
    - MT5Trader.buy() must pass a request with type == ORDER_TYPE_BUY (0)
    - MT5Trader.sell() must pass a request with type == ORDER_TYPE_SELL (1)

    Validates: Requirements 6.2, 6.3
    """
    # MT5 constants (stub values match the real MT5 API integers)
    ORDER_TYPE_BUY  = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1

    captured_buy_request  = {}
    captured_sell_request = {}

    def fake_order_send_buy(request):
        captured_buy_request.update(request)
        return _make_ok_result()

    def fake_order_send_sell(request):
        captured_sell_request.update(request)
        return _make_ok_result()

    trader = MT5Trader()

    # ── Test buy() ────────────────────────────────────────────────────────────
    with patch("execution.trader.mt5") as mock_mt5:
        mock_mt5.ORDER_TYPE_BUY   = ORDER_TYPE_BUY
        mock_mt5.ORDER_TYPE_SELL  = ORDER_TYPE_SELL
        mock_mt5.TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
        mock_mt5.account_info.return_value = _make_account_info()
        mock_mt5.order_send.side_effect = fake_order_send_buy

        result = trader.buy(symbol, lot)

    assert result is True, f"buy() should return True on success (symbol={symbol}, lot={lot})"
    assert "type" in captured_buy_request, "buy() request must contain 'type' key"
    assert captured_buy_request["type"] == ORDER_TYPE_BUY, (
        f"buy() must use ORDER_TYPE_BUY={ORDER_TYPE_BUY}, "
        f"got {captured_buy_request['type']}"
    )

    # ── Test sell() ───────────────────────────────────────────────────────────
    with patch("execution.trader.mt5") as mock_mt5:
        mock_mt5.ORDER_TYPE_BUY   = ORDER_TYPE_BUY
        mock_mt5.ORDER_TYPE_SELL  = ORDER_TYPE_SELL
        mock_mt5.TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
        mock_mt5.account_info.return_value = _make_account_info()
        mock_mt5.order_send.side_effect = fake_order_send_sell

        result = trader.sell(symbol, lot)

    assert result is True, f"sell() should return True on success (symbol={symbol}, lot={lot})"
    assert "type" in captured_sell_request, "sell() request must contain 'type' key"
    assert captured_sell_request["type"] == ORDER_TYPE_SELL, (
        f"sell() must use ORDER_TYPE_SELL={ORDER_TYPE_SELL}, "
        f"got {captured_sell_request['type']}"
    )


# ── Property 9: 所有订单附带正确 Magic Number ─────────────────────────────────
# Validates: Requirements 6.6


@settings(max_examples=100, deadline=None)
@given(
    symbol=symbol_strategy,
    lot=lot_strategy,
)
def test_property9_magic_number_on_all_orders(symbol: str, lot: float):
    """
    For any order sent through MT5Trader (buy or sell, any symbol/lot),
    the MqlTradeRequest's `magic` field must equal Config.MAGIC_NUMBER.

    This ensures every order is tagged for strategy identification regardless
    of the trading direction or instrument.

    Validates: Requirements 6.6
    """
    ORDER_TYPE_BUY   = 0
    ORDER_TYPE_SELL  = 1
    TRADE_ACTION_DEAL = 1

    captured_buy_request  = {}
    captured_sell_request = {}

    def fake_order_send_buy(request):
        captured_buy_request.update(request)
        return _make_ok_result()

    def fake_order_send_sell(request):
        captured_sell_request.update(request)
        return _make_ok_result()

    trader = MT5Trader()

    # ── Assert magic on buy() ─────────────────────────────────────────────────
    with patch("execution.trader.mt5") as mock_mt5:
        mock_mt5.ORDER_TYPE_BUY   = ORDER_TYPE_BUY
        mock_mt5.ORDER_TYPE_SELL  = ORDER_TYPE_SELL
        mock_mt5.TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
        mock_mt5.account_info.return_value = _make_account_info()
        mock_mt5.order_send.side_effect = fake_order_send_buy

        result = trader.buy(symbol, lot)

    assert result is True, f"buy() should return True (symbol={symbol}, lot={lot})"
    assert "magic" in captured_buy_request, "buy() request must contain 'magic' key"
    assert captured_buy_request["magic"] == Config.MAGIC_NUMBER, (
        f"buy() magic field must equal Config.MAGIC_NUMBER={Config.MAGIC_NUMBER}, "
        f"got {captured_buy_request['magic']} (symbol={symbol}, lot={lot})"
    )

    # ── Assert magic on sell() ────────────────────────────────────────────────
    with patch("execution.trader.mt5") as mock_mt5:
        mock_mt5.ORDER_TYPE_BUY   = ORDER_TYPE_BUY
        mock_mt5.ORDER_TYPE_SELL  = ORDER_TYPE_SELL
        mock_mt5.TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
        mock_mt5.account_info.return_value = _make_account_info()
        mock_mt5.order_send.side_effect = fake_order_send_sell

        result = trader.sell(symbol, lot)

    assert result is True, f"sell() should return True (symbol={symbol}, lot={lot})"
    assert "magic" in captured_sell_request, "sell() request must contain 'magic' key"
    assert captured_sell_request["magic"] == Config.MAGIC_NUMBER, (
        f"sell() magic field must equal Config.MAGIC_NUMBER={Config.MAGIC_NUMBER}, "
        f"got {captured_sell_request['magic']} (symbol={symbol}, lot={lot})"
    )
