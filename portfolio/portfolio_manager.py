import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def run_portfolio_risk_allocation(asset_returns_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
    """
    Computes Portfolio Risk Parity Leverage Allocation, Balance Sheet Metrics, 
    and Correlation Heatmap Matrix across assets.
    """
    df_rets = pd.DataFrame(asset_returns_dict).dropna()
    if df_rets.empty:
        return {"error": "Insufficient asset data for portfolio allocation"}

    # Correlation Matrix
    corr_matrix = df_rets.corr().round(3).to_dict()

    # Volatilities
    vols = df_rets.std()
    inv_vols = 1.0 / (vols + 1e-8)
    risk_parity_weights = (inv_vols / inv_vols.sum()).round(4).to_dict()

    # Portfolio Diversification Ratio
    weighted_vol = sum(risk_parity_weights[col] * vols[col] for col in df_rets.columns)
    port_vol = float(np.sqrt(np.dot(list(risk_parity_weights.values()), np.dot(df_rets.cov(), list(risk_parity_weights.values())))))
    diversification_ratio = round(weighted_vol / (port_vol + 1e-8), 2)

    return {
        "portfolio_diversification_ratio": diversification_ratio,
        "risk_parity_leverage_weights": risk_parity_weights,
        "asset_correlation_matrix": corr_matrix,
        "asset_volatilities": vols.round(4).to_dict()
    }
