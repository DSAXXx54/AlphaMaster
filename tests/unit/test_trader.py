"""
tests/unit/test_trader.py — MT5Trader 单元测试

涵盖：
  - 订单被拒绝（retcode != 10009）→ buy()/sell() 返回 False（Req 6.5）
  - 账户信息查询失败 → buy()/sell() 中止并返回 False，且 order_send() 不被调用（Req 6.7）
  - 订单成功（retcode == 10009）→ buy()/sell() 返回 True（Req 6.5）
"""
from unittest.mock import MagicMock, patch

import execution.trader as trader_module
from execution.trader import MT5Trader


# ─────────────────────────────────────────────────────────────────
# 辅助：构建一个符合 MT5 接口规范的 mock mt5 对象
# ─────────────────────────────────────────────────────────────────

def _make_mock_mt5(
    account_equity: float = 10000.0,
    account_margin_free: float = 5000.0,
    retcode: int = 10009,
    order_ticket: int = 123456,
    account_returns_none: bool = False,
) -> MagicMock:
    """构建模拟的 MetaTrader5 模块对象。

    Args:
        account_equity:       account_info().equity 的值
        account_margin_free:  account_info().margin_free 的值
        retcode:              order_send().retcode 的值
        order_ticket:         order_send().order 的值
        account_returns_none: 若为 True，则 account_info() 返回 None
    """
    mock_mt5 = MagicMock()

    # MT5 常量（trader.py 直接引用，必须作为属性存在）
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.TRADE_ACTION_DEAL = 1

    # account_info()
    if account_returns_none:
        mock_mt5.account_info.return_value = None
    else:
        acct = MagicMock()
        acct.equity = account_equity
        acct.margin_free = account_margin_free
        mock_mt5.account_info.return_value = acct

    # order_send()
    result = MagicMock()
    result.retcode = retcode
    result.order = order_ticket
    result.comment = ""
    mock_mt5.order_send.return_value = result

    return mock_mt5


# ─────────────────────────────────────────────────────────────────
# 测试 buy() — 订单被拒绝（retcode != 10009）→ 返回 False（Req 6.5）
# ─────────────────────────────────────────────────────────────────

def test_buy_order_rejected_returns_false():
    """retcode=10006（TRADE_RETCODE_REJECT）时 buy() 必须返回 False。"""
    mock_mt5 = _make_mock_mt5(retcode=10006)

    with patch.object(trader_module, "mt5", mock_mt5):
        trader = MT5Trader()
        result = trader.buy("XAUUSD", 0.01)

    assert result is False, "buy() 在 retcode!=10009 时应返回 False"


# ─────────────────────────────────────────────────────────────────
# 测试 buy() — 账户信息查询失败 → 中止下单，order_send() 不被调用（Req 6.7）
# ─────────────────────────────────────────────────────────────────

def test_buy_account_info_fails_aborts_order():
    """account_info() 返回 None 时，buy() 应中止并返回 False，且不调用 order_send()。"""
    mock_mt5 = _make_mock_mt5(account_returns_none=True)

    with patch.object(trader_module, "mt5", mock_mt5):
        trader = MT5Trader()
        result = trader.buy("XAUUSD", 0.01)

    assert result is False, "account_info()=None 时 buy() 应返回 False"
    mock_mt5.order_send.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# 测试 sell() — 订单被拒绝（retcode != 10009）→ 返回 False（Req 6.5）
# ─────────────────────────────────────────────────────────────────

def test_sell_order_rejected_returns_false():
    """retcode=10006 时 sell() 必须返回 False。"""
    mock_mt5 = _make_mock_mt5(retcode=10006)

    with patch.object(trader_module, "mt5", mock_mt5):
        trader = MT5Trader()
        result = trader.sell("XAUUSD", 0.01)

    assert result is False, "sell() 在 retcode!=10009 时应返回 False"


# ─────────────────────────────────────────────────────────────────
# 测试 sell() — 账户信息查询失败 → 中止下单，order_send() 不被调用（Req 6.7）
# ─────────────────────────────────────────────────────────────────

def test_sell_account_info_fails_aborts_order():
    """account_info() 返回 None 时，sell() 应中止并返回 False，且不调用 order_send()。"""
    mock_mt5 = _make_mock_mt5(account_returns_none=True)

    with patch.object(trader_module, "mt5", mock_mt5):
        trader = MT5Trader()
        result = trader.sell("XAUUSD", 0.01)

    assert result is False, "account_info()=None 时 sell() 应返回 False"
    mock_mt5.order_send.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# 测试 buy() — 订单成功（retcode == 10009）→ 返回 True（Req 6.5）
# ─────────────────────────────────────────────────────────────────

def test_buy_order_success_returns_true():
    """retcode=10009（TRADE_RETCODE_DONE）时 buy() 必须返回 True。"""
    mock_mt5 = _make_mock_mt5(retcode=10009, order_ticket=123456)

    with patch.object(trader_module, "mt5", mock_mt5):
        trader = MT5Trader()
        result = trader.buy("XAUUSD", 0.01)

    assert result is True, "buy() 在 retcode==10009 时应返回 True"


# ─────────────────────────────────────────────────────────────────
# 测试 sell() — 订单成功（retcode == 10009）→ 返回 True（Req 6.5）
# ─────────────────────────────────────────────────────────────────

def test_sell_order_success_returns_true():
    """retcode=10009 时 sell() 必须返回 True。"""
    mock_mt5 = _make_mock_mt5(retcode=10009, order_ticket=789012)

    with patch.object(trader_module, "mt5", mock_mt5):
        trader = MT5Trader()
        result = trader.sell("XAUUSD", 0.01)

    assert result is True, "sell() 在 retcode==10009 时应返回 True"
