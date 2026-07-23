import os
import json
import pandas as pd
from typing import Dict, Any

BALANCE_SHEET_CSV = "reports/portfolio_balance_sheet.csv"

def sync_balance_sheet_and_metrics(metrics_dict: Dict[str, Any], output_path: str = BALANCE_SHEET_CSV) -> str:
    """
    Syncs strategy metrics, account equity, drawdowns, and balance sheets 
    into a structured CSV spreadsheet ready for Google Sheets import.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    record = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "initial_capital": metrics_dict.get("initial_capital", 10000.0),
        "final_equity": metrics_dict.get("final_equity", 10000.0),
        "net_profit": metrics_dict.get("net_profit", 0.0),
        "net_profit_pct": metrics_dict.get("net_profit_pct", 0.0),
        "win_rate_pct": metrics_dict.get("win_rate_pct", 0.0),
        "profit_factor": metrics_dict.get("profit_factor", 0.0),
        "max_drawdown_pct": metrics_dict.get("max_drawdown_pct", 0.0),
        "total_trades": metrics_dict.get("total_trades", 0),
        "sharpe_ratio": metrics_dict.get("sharpe_ratio", 0.0),
        "sortino_ratio": metrics_dict.get("sortino_ratio", 0.0),
        "sqn_score": metrics_dict.get("system_quality_number_sqn", 0.0)
    }

    df_new = pd.DataFrame([record])

    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(output_path, index=False)
    print(f"[Sheets Sync] Balance sheet updated at {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)
