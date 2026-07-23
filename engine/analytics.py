import numpy as np
import pandas as pd
from typing import Dict, List, Any

def calculate_comprehensive_metrics(trades: List[Dict[str, Any]], initial_capital: float, df: pd.DataFrame, fee_pct: float = 0.06, slippage_pct: float = 0.02) -> Dict[str, Any]:
    """
    Calculates 30+ quantitative factors, institutional risk ratios, 
    date metadata, and execution impact matrices.
    """
    total_trades = len(trades)
    if total_trades == 0:
        return {"total_trades": 0, "net_profit": 0.0, "net_profit_pct": 0.0}

    winning_trades = [t for t in trades if t["is_win"]]
    losing_trades = [t for t in trades if not t["is_win"]]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    # Returns & PnLs
    trade_rets = [t["return_pct"] for t in trades]
    trade_pnls = [t["pnl_dollars"] for t in trades]
    win_pnls = [t["pnl_dollars"] for t in winning_trades]
    loss_pnls = [abs(t["pnl_dollars"]) for t in losing_trades]

    gross_profit = sum(win_pnls)
    gross_loss = sum(loss_pnls)
    net_profit = gross_profit - gross_loss
    final_equity = initial_capital + net_profit
    net_profit_pct = (net_profit / initial_capital) * 100.0 if initial_capital > 0 else 0.0

    # Profit Factors (Raw, Net of Fee, Realized Fee+Slippage)
    raw_profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    
    fee_drag_pct = fee_pct * 2.0
    net_gross_profit = gross_profit * (1.0 - fee_drag_pct / 100.0)
    net_gross_loss = gross_loss * (1.0 + fee_drag_pct / 100.0)
    net_profit_factor = (net_gross_profit / net_gross_loss) if net_gross_loss > 0 else (999.0 if net_gross_profit > 0 else 0.0)

    total_drag_pct = (fee_pct + slippage_pct) * 2.0
    realized_gross_profit = gross_profit * (1.0 - total_drag_pct / 100.0)
    realized_gross_loss = gross_loss * (1.0 + total_drag_pct / 100.0)
    realized_profit_factor = (realized_gross_profit / realized_gross_loss) if realized_gross_loss > 0 else (999.0 if realized_gross_profit > 0 else 0.0)

    # Average & Max Wins / Losses
    avg_win_dollar = np.mean(win_pnls) if win_pnls else 0.0
    avg_loss_dollar = np.mean(loss_pnls) if loss_pnls else 0.0
    avg_win_pct = np.mean([t["return_pct"] for t in winning_trades]) if winning_trades else 0.0
    avg_loss_pct = np.mean([abs(t["return_pct"]) for t in losing_trades]) if losing_trades else 0.0

    max_win_dollar = max(win_pnls) if win_pnls else 0.0
    max_loss_dollar = max(loss_pnls) if loss_pnls else 0.0
    max_win_pct = max([t["return_pct"] for t in winning_trades]) if winning_trades else 0.0
    max_loss_pct = max([abs(t["return_pct"]) for t in losing_trades]) if losing_trades else 0.0

    # Win/Loss PnL Payoff Ratio
    payoff_ratio = (avg_win_dollar / avg_loss_dollar) if avg_loss_dollar > 0 else (999.0 if avg_win_dollar > 0 else 0.0)

    # Expectancy ($ per trade & % per trade)
    win_prob = win_count / total_trades
    loss_prob = loss_count / total_trades
    expectancy_dollar = (win_prob * avg_win_dollar) - (loss_prob * avg_loss_dollar)
    expectancy_pct = (win_prob * avg_win_pct) - (loss_prob * avg_loss_pct)

    # Consecutive Wins & Losses
    max_consec_wins = 0
    max_consec_losses = 0
    cur_wins = 0
    cur_losses = 0

    for t in trades:
        if t["is_win"]:
            cur_wins += 1
            cur_losses = 0
            if cur_wins > max_consec_wins:
                max_consec_wins = cur_wins
        else:
            cur_losses += 1
            cur_wins = 0
            if cur_losses > max_consec_losses:
                max_consec_losses = cur_losses

    # Holding Durations (Bar count & hours)
    holding_bars = [t["holding_bars"] for t in trades]
    avg_holding = np.mean(holding_bars) if holding_bars else 0.0
    median_holding = np.median(holding_bars) if holding_bars else 0.0
    min_holding = min(holding_bars) if holding_bars else 0
    max_holding = max(holding_bars) if holding_bars else 0
    total_active_bars = sum(holding_bars)
    
    total_candles = len(df)
    market_exposure_pct = (total_active_bars / total_candles * 100.0) if total_candles > 0 else 0.0

    # Dataset Date Information & CAGR
    start_date = str(df['timestamp'].iloc[0])[:19]
    end_date = str(df['timestamp'].iloc[-1])[:19]
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    total_years = max(0.01, total_days / 365.0)

    cagr_pct = (((final_equity / initial_capital) ** (1.0 / total_years)) - 1.0) * 100.0 if initial_capital > 0 and final_equity > 0 else 0.0

    # Max Drawdown $, %, Duration & Ulcer Index
    peak = initial_capital
    max_dd_dollar = 0.0
    max_dd_pct = 0.0
    cur_equity = initial_capital
    cur_dd_duration = 0
    max_dd_duration_bars = 0
    drawdowns_squared = []

    for t in trades:
        cur_equity += t["pnl_dollars"]
        if cur_equity > peak:
            peak = cur_equity
            cur_dd_duration = 0
            drawdowns_squared.append(0.0)
        else:
            dd_d = peak - cur_equity
            dd_p = (dd_d / peak) * 100.0
            drawdowns_squared.append(dd_p ** 2)
            if dd_d > max_dd_dollar:
                max_dd_dollar = dd_d
            if dd_p > max_dd_pct:
                max_dd_pct = dd_p
            cur_dd_duration += t["holding_bars"]
            if cur_dd_duration > max_dd_duration_bars:
                max_dd_duration_bars = cur_dd_duration

    ulcer_index = np.sqrt(np.mean(drawdowns_squared)) if drawdowns_squared else 0.0
    martin_ratio = (cagr_pct / ulcer_index) if ulcer_index > 0 else 0.0

    # Risk Ratios (Sharpe, Sortino, Calmar, Recovery Factor, SQN, Tail Ratio, Gain-to-Pain)
    rets_arr = np.array(trade_rets) / 100.0
    std_ret = np.std(rets_arr) if len(rets_arr) > 1 else 0.0
    mean_ret = np.mean(rets_arr) if len(rets_arr) > 0 else 0.0

    sharpe_ratio = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0

    downside_rets = rets_arr[rets_arr < 0]
    downside_std = np.std(downside_rets) if len(downside_rets) > 0 else 0.0
    sortino_ratio = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 1e-8 else 0.0

    sqn = (mean_ret / std_ret * np.sqrt(total_trades)) if std_ret > 1e-8 else 0.0
    calmar_ratio = (cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0
    recovery_factor = (net_profit / max_dd_dollar) if max_dd_dollar > 0 else 0.0

    pos_sum = sum(rets_arr[rets_arr > 0])
    neg_sum = abs(sum(rets_arr[rets_arr < 0]))
    gain_to_pain_ratio = (pos_sum / neg_sum) if neg_sum > 0 else 999.0

    pct_95_win = np.percentile(rets_arr, 95) if len(rets_arr) > 0 else 0.0
    pct_5_loss = abs(np.percentile(rets_arr, 5)) if len(rets_arr) > 0 else 1.0
    tail_ratio = (pct_95_win / pct_5_loss) if pct_5_loss > 0 else 0.0

    return {
        # Dataset Metadata
        "dataset_start_date": start_date,
        "dataset_end_date": end_date,
        "total_dataset_days": total_days,
        "total_candles": total_candles,
        
        # Capital & Returns
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "net_profit": round(net_profit, 2),
        "net_profit_pct": round(net_profit_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),

        # Trade Counts & Win Rate
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate_pct": round(win_rate, 2),
        "raw_profit_factor": round(raw_profit_factor, 2),
        "net_profit_factor": round(net_profit_factor, 2),
        "realized_profit_factor": round(realized_profit_factor, 2),
        "profit_factor": round(raw_profit_factor, 2),
        "payoff_ratio": round(payoff_ratio, 2),

        # Expectancy & Averages
        "expectancy_dollars": round(expectancy_dollar, 2),
        "expectancy_pct": round(expectancy_pct, 2),
        "avg_win_dollar": round(avg_win_dollar, 2),
        "avg_loss_dollar": round(avg_loss_dollar, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "max_win_dollar": round(max_win_dollar, 2),
        "max_loss_dollar": round(max_loss_dollar, 2),
        "max_win_pct": round(max_win_pct, 2),
        "max_loss_pct": round(max_loss_pct, 2),

        # Streak & Duration
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "avg_holding_bars": round(avg_holding, 1),
        "median_holding_bars": float(median_holding),
        "min_holding_bars": int(min_holding),
        "max_holding_bars": int(max_holding),
        "market_exposure_pct": round(market_exposure_pct, 2),

        # Institutional Risk Factors & Advanced Ratios
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_dollars": round(max_dd_dollar, 2),
        "max_drawdown_duration_bars": int(max_dd_duration_bars),
        "ulcer_index": round(ulcer_index, 2),
        "martin_ratio": round(martin_ratio, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "calmar_ratio": round(calmar_ratio, 2),
        "gain_to_pain_ratio": round(gain_to_pain_ratio, 2),
        "tail_ratio": round(tail_ratio, 2),
        "recovery_factor": round(recovery_factor, 2),
        "system_quality_number_sqn": round(sqn, 2)
    }

def run_detailed_single_backtest(df: pd.DataFrame, fast_p: int, slow_p: int, initial_capital: float = 10000.0, fee: float = 0.0006) -> Dict[str, Any]:
    """
    Executes a detailed single backtest and calculates 30+ quantitative factors.
    """
    prices = df['close'].values
    timestamps = df['timestamp'].values
    n = len(prices)

    if n < slow_p:
        return {"error": "Insufficient data"}

    fast_ema = pd.Series(prices).ewm(span=fast_p, adjust=False).mean().values
    slow_ema = pd.Series(prices).ewm(span=slow_p, adjust=False).mean().values

    equity = initial_capital
    position = 0
    entry_price = 0.0
    entry_idx = 0
    entry_time = None

    trades = []
    equity_curve = []
    cumulative_profit_pct = []
    cumulative_loss_pct = []
    drawdown_curve = []

    peak_equity = initial_capital
    cum_gross_profit = 0.0
    cum_gross_loss = 0.0

    for i in range(slow_p, n):
        p = prices[i]
        ts = str(timestamps[i])[:19]
        prev_f = fast_ema[i-1]
        prev_s = slow_ema[i-1]
        curr_f = fast_ema[i]
        curr_s = slow_ema[i]

        if position == 0 and prev_f <= prev_s and curr_f > curr_s:
            position = 1
            entry_price = p * (1.0 + fee)
            entry_idx = i
            entry_time = ts

        elif position == 1 and prev_f >= prev_s and curr_f < curr_s:
            exit_price = p * (1.0 - fee)
            trade_pct = (exit_price - entry_price) / entry_price * 100.0
            holding_bars = i - entry_idx
            
            pnl_dollars = equity * (trade_pct / 100.0)
            equity += pnl_dollars

            if pnl_dollars > 0:
                cum_gross_profit += (trade_pct / 100.0)
            else:
                cum_gross_loss += abs(trade_pct / 100.0)

            trades.append({
                "trade_num": len(trades) + 1,
                "entry_time": entry_time,
                "exit_time": ts,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "return_pct": round(trade_pct, 2),
                "pnl_dollars": round(pnl_dollars, 2),
                "equity_after": round(equity, 2),
                "holding_bars": holding_bars,
                "is_win": bool(pnl_dollars > 0)
            })

            position = 0

        if equity > peak_equity:
            peak_equity = equity
        dd_pct = (peak_equity - equity) / peak_equity * 100.0 if peak_equity > 0 else 0.0

        equity_curve.append({"timestamp": ts, "equity": round(equity, 2)})
        drawdown_curve.append({"timestamp": ts, "drawdown_pct": round(dd_pct, 2)})
        cumulative_profit_pct.append({"timestamp": ts, "cum_profit_pct": round(cum_gross_profit * 100.0, 2)})
        cumulative_loss_pct.append({"timestamp": ts, "cum_loss_pct": round(cum_gross_loss * 100.0, 2)})

    metrics = calculate_comprehensive_metrics(trades, initial_capital, df)

    return {
        "parameters": {"fast_ema": fast_p, "slow_ema": slow_p},
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "cumulative_profit_pct": cumulative_profit_pct,
        "cumulative_loss_pct": cumulative_loss_pct
    }
