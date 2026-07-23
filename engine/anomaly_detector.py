import numpy as np
import pandas as pd
from typing import Dict, List, Any

def run_anomaly_and_pattern_detection(trades: List[Dict[str, Any]], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs automated Anomaly Detection, Pattern Recognition, Session Analysis, 
    and Statistical Risk Probability Modeling (VaR & CVaR).
    """
    if not trades or len(trades) == 0:
        return {"status": "No trades to analyze"}

    returns = np.array([t["return_pct"] for t in trades])
    pnls = np.array([t["pnl_dollars"] for t in trades])

    # 1. Anomaly Detection: Z-Score Outliers (|Z| > 2.5)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1.0
    z_scores = (returns - mean_ret) / (std_ret if std_ret > 0 else 1.0)

    anomalous_trades = []
    for idx, z in enumerate(z_scores):
        if abs(z) > 2.5:
            anomalous_trades.append({
                "trade_num": trades[idx]["trade_num"],
                "return_pct": trades[idx]["return_pct"],
                "pnl_dollars": trades[idx]["pnl_dollars"],
                "z_score": round(float(z), 2),
                "anomaly_type": "Outlier Win" if z > 0 else "Outlier Loss"
            })

    # 2. Pattern Recognition: Trading Session Performance Breakdown
    # Sessions (UTC): Asian (00:00-08:00), London (08:00-16:00), New York (13:00-21:00)
    session_stats = {"Asian": [], "London": [], "New_York": [], "Off_Hours": []}

    for t in trades:
        try:
            entry_dt = pd.to_datetime(t["entry_time"])
            hour = entry_dt.hour
            if 0 <= hour < 8:
                session_stats["Asian"].append(t)
            elif 8 <= hour < 13:
                session_stats["London"].append(t)
            elif 13 <= hour < 21:
                session_stats["New_York"].append(t)
            else:
                session_stats["Off_Hours"].append(t)
        except Exception:
            pass

    session_summary = {}
    for sess, sess_trades in session_stats.items():
        if sess_trades:
            s_rets = [tr["return_pct"] for tr in sess_trades]
            s_wins = [tr for tr in sess_trades if tr["is_win"]]
            session_summary[sess] = {
                "trade_count": len(sess_trades),
                "win_rate_pct": round(len(s_wins) / len(sess_trades) * 100.0, 2),
                "avg_return_pct": round(float(np.mean(s_rets)), 2),
                "net_pnl": round(float(sum(tr["pnl_dollars"] for tr in sess_trades)), 2)
            }

    # 3. Statistical Probability Modeling: Value at Risk (VaR 95% & 99%) & Expected Shortfall (CVaR)
    sorted_rets = np.sort(returns)
    var_95 = float(np.percentile(sorted_rets, 5))
    var_99 = float(np.percentile(sorted_rets, 1))

    cvar_95_losses = sorted_rets[sorted_rets <= var_95]
    cvar_95 = float(np.mean(cvar_95_losses)) if len(cvar_95_losses) > 0 else var_95

    # Volatility Regime Shift Detection
    df_vol = df['close'].pct_change().rolling(20).std()
    recent_vol = float(df_vol.iloc[-1]) if not df_vol.empty else 0.0
    hist_vol_95 = float(df_vol.quantile(0.95)) if not df_vol.empty else 0.0
    volatility_regime = "High Volatility Spike" if recent_vol > hist_vol_95 else "Normal Volatility"

    return {
        "anomaly_detection": {
            "total_anomalies_found": len(anomalous_trades),
            "anomalous_trades": anomalous_trades
        },
        "pattern_recognition": {
            "session_performance": session_summary,
            "volatility_regime": volatility_regime,
            "recent_20bar_volatility": round(recent_vol * 100.0, 4)
        },
        "statistical_probability_models": {
            "value_at_risk_95_pct": round(var_95, 2),
            "value_at_risk_99_pct": round(var_99, 2),
            "expected_shortfall_cvar_95_pct": round(cvar_95, 2),
            "return_distribution_skewness": round(float(pd.Series(returns).skew()), 2),
            "return_distribution_kurtosis": round(float(pd.Series(returns).kurtosis()), 2)
        }
    }
