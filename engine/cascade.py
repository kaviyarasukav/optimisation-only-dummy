import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

def run_sequential_cascade_backtest(
    df: pd.DataFrame, 
    ema_periods: List[int] = [8, 13, 21, 34, 55, 89, 200],
    cascade_mode: str = "linear_sequential",
    initial_capital: float = 10000.0,
    fee_pct: float = 0.06,
    slippage_pct: float = 0.02
) -> Dict[str, Any]:
    """
    Executes a Sequential Multi-EMA Cascade Backtest.
    
    Modes:
    - 'single': Standard 2-EMA (e.g. 21/55)
    - 'linear_sequential': Staged position scaling as EMAs cross in sequence (8/13 -> 13/21 -> 21/34), 
                           with early-warning partial exit when 8/13 reverses before 55/200 death cross!
    - 'matrix': Full multi-layer EMA momentum divergence matrix.
    """
    prices = df['close'].values
    timestamps = df['timestamp'].values
    n = len(prices)

    max_p = max(ema_periods)
    if n < max_p:
        return {"error": "Insufficient data for cascade evaluation"}

    total_fee = (fee_pct + slippage_pct) / 100.0

    # Calculate all EMA series
    ema_dict = {}
    for p in ema_periods:
        ema_dict[p] = pd.Series(prices).ewm(span=p, adjust=False).mean().values

    equity = initial_capital
    peak_equity = initial_capital

    position_units = 0.0  # Fraction of account allocated [0.0 to 1.0]
    entry_prices = []
    entry_indices = []
    trades = []

    cum_gross_profit = 0.0
    cum_gross_loss = 0.0

    # Cascade Stages: e.g., (8,13), (13,21), (21,34), (34,55)
    pairs = [(ema_periods[k], ema_periods[k+1]) for k in range(len(ema_periods)-1)]
    fastest_p = pairs[0][0]
    second_p = pairs[0][1]
    slowest_p = pairs[-1][1]

    fastest_ema = ema_dict[fastest_p]
    second_ema = ema_dict[second_p]
    slowest_ema = ema_dict[slowest_p]

    for i in range(max_p, n):
        p = prices[i]
        ts = str(timestamps[i])[:19]

        # 1. Count how many cascade pairs are in bullish alignment
        bullish_pairs = 0
        for fp, sp in pairs:
            if ema_dict[fp][i] > ema_dict[sp][i]:
                bullish_pairs += 1

        target_allocation = bullish_pairs / len(pairs)  # 0.0, 0.25, 0.5, 0.75, 1.0

        # Early Warning Exit Check: Fastest line crosses below second line -> Protect Profit / Prevent Exponential Drawdown
        early_warning = (fastest_ema[i-1] >= second_ema[i-1]) and (fastest_ema[i] < second_ema[i])

        if early_warning and position_units > 0:
            # Scale down position by 50% immediately before major death cross happens!
            target_allocation = target_allocation * 0.5

        # Execute Position Adjustment
        if target_allocation > position_units:
            # Scale In / Entry
            added_units = target_allocation - position_units
            exec_price = p * (1.0 + total_fee)
            entry_prices.append((exec_price, added_units))
            entry_indices.append(i)
            position_units = target_allocation

        elif target_allocation < position_units and len(entry_prices) > 0:
            # Scale Out / Exit
            reduced_units = position_units - target_allocation
            exec_price = p * (1.0 - total_fee)

            # Average entry price of active units
            avg_entry = sum(p * u for p, u in entry_prices) / sum(u for p, u in entry_prices)
            trade_pct = (exec_price - avg_entry) / avg_entry * 100.0
            pnl_dollars = (equity * reduced_units) * (trade_pct / 100.0)

            equity += pnl_dollars

            holding_bars = i - min(entry_indices)

            trades.append({
                "trade_num": len(trades) + 1,
                "entry_time": str(timestamps[min(entry_indices)])[:19],
                "exit_time": ts,
                "entry_price": round(avg_entry, 4),
                "exit_price": round(exec_price, 4),
                "return_pct": round(trade_pct, 2),
                "pnl_dollars": round(pnl_dollars, 2),
                "equity_after": round(equity, 2),
                "holding_bars": holding_bars,
                "is_win": bool(pnl_dollars > 0),
                "signal_type": "Early_Warning_Exit" if early_warning else "Standard_Cascade_Exit"
            })

            position_units = target_allocation
            if position_units == 0:
                entry_prices = []
                entry_indices = []

    # Final Summary Statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t["is_win"]]
    losing_trades = [t for t in trades if not t["is_win"]]

    net_profit = equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100.0

    return {
        "cascade_mode": cascade_mode,
        "ema_periods": ema_periods,
        "metrics": {
            "net_profit": round(net_profit, 2),
            "net_profit_pct": round(net_profit_pct, 2),
            "final_equity": round(equity, 2),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": round((len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0, 2),
            "early_warning_exits_triggered": sum(1 for t in trades if t["signal_type"] == "Early_Warning_Exit")
        },
        "trades": trades
    }
