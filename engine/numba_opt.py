import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@njit(fastmath=True)
def calculate_ema_array(prices, period):
    """Calculates Exponential Moving Average for a 1D price array."""
    n = len(prices)
    ema = np.empty(n, dtype=np.float64)
    if n == 0 or period <= 0:
        return ema
    
    alpha = 2.0 / (period + 1.0)
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = prices[i] * alpha + ema[i-1] * (1.0 - alpha)
    return ema


@njit(fastmath=True)
def evaluate_ema_pair(prices, fast_ema, slow_ema, fast_p, slow_p, initial_capital=10000.0, fee=0.0006):
    """
    Evaluates trading performance for pre-computed Fast & Slow EMA arrays.
    Robust against division by zero, empty arrays, or extreme drawdowns.
    """
    n = len(prices)
    if n < slow_p:
        return np.zeros(18, dtype=np.float64)

    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    entry_idx = 0

    total_trades = 0
    winning_trades = 0
    losing_trades = 0

    gross_profit = 0.0
    gross_loss = 0.0

    max_win_pct = 0.0
    max_loss_pct = 0.0

    total_holding_bars = 0
    min_holding_bars = 999999
    max_holding_bars = 0

    equity = initial_capital
    peak_equity = initial_capital
    max_drawdown_dollars = 0.0
    max_drawdown_pct = 0.0
    
    cur_dd_duration = 0
    max_dd_duration = 0

    returns_list = np.zeros(n, dtype=np.float64)
    ret_count = 0

    # Start evaluation loop after slow EMA warm-up
    for i in range(slow_p, n):
        prev_f = fast_ema[i-1]
        prev_s = slow_ema[i-1]
        curr_f = fast_ema[i]
        curr_s = slow_ema[i]
        price = prices[i]

        # Drawdown duration tracking
        if equity < peak_equity:
            cur_dd_duration += 1
            if cur_dd_duration > max_dd_duration:
                max_dd_duration = cur_dd_duration
        else:
            cur_dd_duration = 0

        # Bullish Crossover (BUY signal)
        if position == 0 and prev_f <= prev_s and curr_f > curr_s:
            position = 1
            entry_price = price * (1.0 + fee)
            entry_idx = i

        # Bearish Crossover (SELL signal)
        elif position == 1 and prev_f >= prev_s and curr_f < curr_s:
            exit_price = price * (1.0 - fee)
            trade_pct = (exit_price - entry_price) / entry_price
            holding_bars = i - entry_idx

            total_trades += 1
            total_holding_bars += holding_bars
            if holding_bars < min_holding_bars:
                min_holding_bars = holding_bars
            if holding_bars > max_holding_bars:
                max_holding_bars = holding_bars

            trade_pnl = equity * trade_pct
            equity += trade_pnl

            if ret_count < n:
                returns_list[ret_count] = trade_pct
                ret_count += 1

            if trade_pnl > 0:
                winning_trades += 1
                gross_profit += trade_pnl
                if trade_pct > max_win_pct:
                    max_win_pct = trade_pct
            else:
                losing_trades += 1
                gross_loss += abs(trade_pnl)
                if trade_pct < max_loss_pct:
                    max_loss_pct = trade_pct

            # Safe Peak Equity & Drawdown Update
            if equity > peak_equity:
                peak_equity = equity
            elif peak_equity > 0:
                dd_dollars = peak_equity - equity
                dd_pct = dd_dollars / peak_equity
                if dd_dollars > max_drawdown_dollars:
                    max_drawdown_dollars = dd_dollars
                if dd_pct > max_drawdown_pct:
                    max_drawdown_pct = dd_pct

            position = 0

    # Close open position at final candle if still active
    if position == 1 and n > entry_idx:
        exit_price = prices[n-1] * (1.0 - fee)
        trade_pct = (exit_price - entry_price) / entry_price
        holding_bars = (n - 1) - entry_idx

        total_trades += 1
        total_holding_bars += holding_bars
        if holding_bars < min_holding_bars:
            min_holding_bars = holding_bars
        if holding_bars > max_holding_bars:
            max_holding_bars = holding_bars

        trade_pnl = equity * trade_pct
        equity += trade_pnl
        if trade_pnl > 0:
            winning_trades += 1
            gross_profit += trade_pnl
        else:
            losing_trades += 1
            gross_loss += abs(trade_pnl)

    net_profit = equity - initial_capital
    net_profit_pct = (equity - initial_capital) / initial_capital * 100.0 if initial_capital > 0 else 0.0
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_holding_bars = (total_holding_bars / total_trades) if total_trades > 0 else 0.0
    if min_holding_bars == 999999:
        min_holding_bars = 0

    # Sharpe ratio calculation
    sharpe = 0.0
    if ret_count > 1:
        valid_rets = returns_list[:ret_count]
        std = np.std(valid_rets)
        if std > 1e-8:
            sharpe = (np.mean(valid_rets) / std) * np.sqrt(252.0)

    # Sanitize NaNs and Infs
    if np.isnan(sharpe) or np.isinf(sharpe):
        sharpe = 0.0

    res = np.array([
        fast_p,                  # 0
        slow_p,                  # 1
        net_profit,              # 2
        net_profit_pct,          # 3
        total_trades,            # 4
        win_rate,                # 5
        profit_factor,           # 6
        max_drawdown_pct * 100,  # 7
        max_drawdown_dollars,    # 8
        sharpe,                  # 9
        avg_holding_bars,        # 10
        min_holding_bars,        # 11
        max_holding_bars,        # 12
        winning_trades,          # 13
        losing_trades,           # 14
        gross_profit,            # 15
        gross_loss,              # 16
        max_dd_duration          # 17
    ], dtype=np.float64)

    return res


def run_parallel_grid_search(prices: np.ndarray, fast_range: np.ndarray, slow_range: np.ndarray, initial_capital: float = 10000.0, fee: float = 0.0006) -> np.ndarray:
    """
    High-speed grid search using EMA pre-computation.
    Pre-computes EMA series for all unique period values first, 
    then evaluates all valid (fast < slow) pairs.
    """
    all_periods = np.unique(np.concatenate((fast_range, slow_range)))
    if len(all_periods) == 0:
        return np.zeros((0, 18), dtype=np.float64)

    max_period = int(np.max(all_periods))
    n = len(prices)

    # Pre-calculate EMA matrix: shape (max_period + 1, n)
    ema_matrix = np.zeros((max_period + 1, n), dtype=np.float64)
    for p in all_periods:
        ema_matrix[p] = calculate_ema_array(prices, p)

    # Build valid parameter tuples
    valid_pairs = []
    for fp in fast_range:
        for sp in slow_range:
            if fp < sp:
                valid_pairs.append((fp, sp))

    num_combos = len(valid_pairs)
    if num_combos == 0:
        return np.zeros((0, 18), dtype=np.float64)

    results = np.zeros((num_combos, 18), dtype=np.float64)

    # Inner execution function
    def process_pair(idx):
        fp, sp = valid_pairs[idx]
        fast_ema = ema_matrix[fp]
        slow_ema = ema_matrix[sp]
        return evaluate_ema_pair(prices, fast_ema, slow_ema, fp, sp, initial_capital, fee)

    for idx in range(num_combos):
        results[idx] = process_pair(idx)

    # Replace any potential NaNs or Infs in results matrix
    np.nan_to_num(results, copy=False, nan=0.0, posinf=999.0, neginf=-999.0)
    return results
