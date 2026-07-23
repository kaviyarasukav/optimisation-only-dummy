import numpy as np
import pandas as pd
from typing import Dict, List, Any

def run_grid_exponentiation_test(
    df: pd.DataFrame,
    fast_p: int = 26,
    slow_p: int = 37,
    grid_step_pct: float = 1.0,
    max_grid_levels: int = 3,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.06
) -> Dict[str, Any]:
    """
    Evaluates Grid Exponentiation position scaling.
    Tests whether grid compounding/scaling elevates low, average, or high 
    profitability parameters into superior performance.
    """
    prices = df['close'].values
    timestamps = df['timestamp'].values
    n = len(prices)

    if n < slow_p:
        return {"error": "Insufficient data"}

    fast_ema = pd.Series(prices).ewm(span=fast_p, adjust=False).mean().values
    slow_ema = pd.Series(prices).ewm(span=slow_p, adjust=False).mean().values

    fee = fee_pct / 100.0
    equity = initial_capital
    grid_trades = []

    # Grid transition statistics
    grid_hits = {i: 0 for i in range(-max_grid_levels, max_grid_levels + 1)}

    position = 0
    base_entry_price = 0.0
    active_grid_levels = []

    for i in range(slow_p, n):
        p = prices[i]
        ts = str(timestamps[i])[:19]
        prev_f = fast_ema[i-1]
        prev_s = slow_ema[i-1]
        curr_f = fast_ema[i]
        curr_s = slow_ema[i]

        # Initial Entry Signal
        if position == 0 and prev_f <= prev_s and curr_f > curr_s:
            position = 1
            base_entry_price = p * (1.0 + fee)
            active_grid_levels = [{"level": 0, "price": base_entry_price, "size": 1.0}]
            grid_hits[0] += 1

        elif position == 1:
            # Check Grid Level Scaling (Pullback Additions or Take Profit Grid Exponent)
            change_pct = (p - base_entry_price) / base_entry_price * 100.0
            cur_level = int(change_pct // grid_step_pct)

            if cur_level in grid_hits:
                grid_hits[cur_level] += 1

            # Grid Scale-In (Add level if price drops by grid_step_pct up to max_grid_levels)
            if cur_level < 0 and abs(cur_level) <= max_grid_levels:
                existing_levels = [g["level"] for g in active_grid_levels]
                if cur_level not in existing_levels:
                    grid_price = p * (1.0 + fee)
                    active_grid_levels.append({"level": cur_level, "price": grid_price, "size": 0.5})

            # Signal Exit (EMA Bearish Crossover) -> Close all grid positions
            if prev_f >= prev_s and curr_f < curr_s:
                exit_price = p * (1.0 - fee)
                total_pnl = 0.0
                total_cost = sum(g["price"] * g["size"] for g in active_grid_levels)
                total_units = sum(g["size"] for g in active_grid_levels)
                avg_grid_entry = total_cost / total_units

                trade_pct = (exit_price - avg_grid_entry) / avg_grid_entry * 100.0
                pnl_dollars = (equity * 0.1 * total_units) * (trade_pct / 100.0)
                equity += pnl_dollars

                grid_trades.append({
                    "exit_time": ts,
                    "grid_entry_count": len(active_grid_levels),
                    "avg_entry_price": round(avg_grid_entry, 4),
                    "exit_price": round(exit_price, 4),
                    "return_pct": round(trade_pct, 2),
                    "pnl_dollars": round(pnl_dollars, 2),
                    "is_win": bool(pnl_dollars > 0)
                })

                position = 0
                active_grid_levels = []

    total_t = len(grid_trades)
    wins = [t for t in grid_trades if t["is_win"]]
    net_pnl = equity - initial_capital
    net_pnl_pct = (net_pnl / initial_capital) * 100.0

    total_hits = sum(grid_hits.values())
    grid_probabilities = {lvl: round((count / total_hits * 100.0), 2) if total_hits > 0 else 0.0 for lvl, count in grid_hits.items()}

    return {
        "parameters": {"fast_ema": fast_p, "slow_ema": slow_p},
        "grid_settings": {"grid_step_pct": grid_step_pct, "max_grid_levels": max_grid_levels},
        "metrics": {
            "net_profit": round(net_pnl, 2),
            "net_profit_pct": round(net_pnl_pct, 2),
            "total_grid_trades": total_t,
            "win_rate_pct": round((len(wins) / total_t * 100.0) if total_t > 0 else 0.0, 2),
            "profit_factor": round((sum(t["pnl_dollars"] for t in wins) / sum(abs(t["pnl_dollars"]) for t in grid_trades if not t["is_win"])) if sum(abs(t["pnl_dollars"]) for t in grid_trades if not t["is_win"]) > 0 else 999.0, 2)
        },
        "grid_level_transition_probabilities_pct": grid_probabilities,
        "grid_trades": grid_trades
    }
