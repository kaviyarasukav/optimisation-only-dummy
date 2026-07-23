import pandas as pd
from typing import Tuple, List, Dict, Any

def split_in_out_of_sample(df: pd.DataFrame, in_sample_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame chronologically into In-Sample (training optimization) 
    and Out-of-Sample (unseen validation) datasets.
    """
    n = len(df)
    split_idx = int(n * in_sample_ratio)
    
    in_sample_df = df.iloc[:split_idx].reset_index(drop=True)
    out_sample_df = df.iloc[split_idx:].reset_index(drop=True)

    return in_sample_df, out_sample_df

def format_grid_results(results_matrix, top_n: int = 50) -> List[Dict[str, Any]]:
    """
    Converts raw grid results matrix into ranked list of dicts with 18+ quantitative factors.
    """
    results_list = []
    for row in results_matrix:
        results_list.append({
            "fast_ema": int(row[0]),
            "slow_ema": int(row[1]),
            "net_profit": float(round(row[2], 2)),
            "net_profit_pct": float(round(row[3], 2)),
            "total_trades": int(row[4]),
            "win_rate": float(round(row[5], 2)),
            "profit_factor": float(round(row[6], 2)),
            "max_drawdown_pct": float(round(row[7], 2)),
            "max_drawdown_dollars": float(round(row[8], 2)),
            "sharpe_ratio": float(round(row[9], 2)),
            "avg_holding_bars": float(round(row[10], 1)),
            "min_holding_bars": int(row[11]),
            "max_holding_bars": int(row[12]),
            "winning_trades": int(row[13]),
            "losing_trades": int(row[14]),
            "gross_profit": float(round(row[15], 2)),
            "gross_loss": float(round(row[16], 2)),
            "max_drawdown_duration_bars": int(row[17])
        })

    # Sort by Net Profit % descending
    results_list.sort(key=lambda x: x["net_profit_pct"], reverse=True)
    return results_list
