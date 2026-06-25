"""
Baseline Strategies for Ablation Comparison

Implements:
1. Buy & Hold (equal-weight, rebalance once)
2. Rolling Markowitz Mean-Variance Optimization (monthly rebalance, trailing 252-day covariance)
"""

import os
import logging
import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.optimize import minimize
from pypfopt import risk_models

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


def get_returns_matrix(df):
    """Reshape processed flat dataframe into a pivot table of log returns."""
    returns = df.pivot(index='Date', columns='Ticker', values='Log_Return')
    return returns.dropna()


def backtest_buy_and_hold(prices_df):
    """
    Equally weighted buy-and-hold strategy baseline using vectorbt.
    """
    logger.info("Running Buy and Hold Baseline...")
    prices = prices_df.pivot(index='Date', columns='Ticker', values='Close')
    prices.index = pd.to_datetime(prices.index)
    
    num_assets = len(prices.columns)
    weights = np.array([1.0/num_assets] * num_assets)
    
    portfolio = vbt.Portfolio.from_orders(
        prices,
        size=weights,
        size_type='targetpercent',
        group_by=True,
        cash_sharing=True,
        init_cash=1000000,
        fees=0.001,
        freq='D'
    )
    
    return portfolio


def markowitz_max_sharpe(returns_df, risk_free_rate=0.02):
    """
    Solve for max Sharpe Ratio Markowitz portfolio (long-only, fully invested).
    Uses the provided risk_free_rate (should be period-appropriate, not hardcoded).
    Returns optimal weight array, or equal-weight if optimization fails.
    """
    num_assets = len(returns_df.columns)
    equal_weight = np.ones(num_assets) / num_assets
    
    if len(returns_df) < 60:
        return equal_weight
    
    mean_returns = returns_df.mean() * 252
    # Use Ledoit-Wolf shrinkage to prevent singular matrices
    cov_matrix = risk_models.CovarianceShrinkage(returns_df, returns_data=True).ledoit_wolf()
    
    # CovarianceShrinkage returns a DataFrame, so we use .values
    if np.any(np.isnan(cov_matrix.values)) or np.any(np.isinf(cov_matrix.values)):
        return equal_weight
    
    def neg_sharpe(weights):
        port_ret = np.sum(mean_returns.values * weights)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        if port_std < 1e-10:
            return 0
        return -(port_ret - risk_free_rate) / port_std
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.40) for _ in range(num_assets))
    
    try:
        result = minimize(
            neg_sharpe,
            equal_weight,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        if result.success:
            return result.x
        else:
            return equal_weight
    except Exception:
        return equal_weight


def backtest_rolling_markowitz(prices_df, rebalance_freq='MS', lookback=252):
    """
    Rolling Markowitz MVO: rebalance monthly using trailing `lookback`-day covariance.
    
    This is a proper rolling baseline, not a static single optimization.
    At each rebalance date, we compute Markowitz weights using only data
    strictly before that date (no look-ahead bias).
    
    Args:
        prices_df: DataFrame with columns [Date, Ticker, Close, Log_Return, ...]
        rebalance_freq: Pandas frequency string for rebalance dates ('MS' = month start)
        lookback: Number of trading days for trailing covariance window
    """
    logger.info("Running Rolling Markowitz MVO Baseline...")
    
    prices = prices_df.pivot(index='Date', columns='Ticker', values='Close')
    prices.index = pd.to_datetime(prices.index)
    
    returns = prices_df.pivot(index='Date', columns='Ticker', values='Log_Return')
    returns.index = pd.to_datetime(returns.index)
    returns = returns.dropna(how='all')
    
    tickers = prices.columns.tolist()
    num_assets = len(tickers)
    
    # Generate rebalance dates (month starts within our date range)
    rebalance_dates = pd.date_range(
        start=prices.index[lookback],  # Can't optimize until we have enough history
        end=prices.index[-1],
        freq=rebalance_freq
    )
    
    # For each trading day, determine which weights to use
    all_dates = prices.index
    weight_schedule = pd.DataFrame(
        index=all_dates,
        columns=tickers,
        data=1.0/num_assets  # Default to equal weight
    )
    
    current_weights = np.ones(num_assets) / num_assets
    
    for reb_date in rebalance_dates:
        valid_dates = returns.index[returns.index <= reb_date]
        if len(valid_dates) < lookback:
            continue
        
        trailing_returns = returns.loc[valid_dates[-lookback:]]
        trailing_returns = trailing_returns.dropna(axis=1, how='all').dropna()
        
        if len(trailing_returns) < 60:
            continue
        
        # Get period-appropriate risk-free rate
        reb_rf = 0.02  # default
        if 'Risk_Free_Rate' in prices_df.columns:
            rf_rows = prices_df[(pd.to_datetime(prices_df['Date']) <= reb_date)]
            if len(rf_rows) > 0:
                last_rf = rf_rows['Risk_Free_Rate'].dropna()
                if len(last_rf) > 0:
                    reb_rf = float(last_rf.iloc[-1])
        
        new_weights = markowitz_max_sharpe(trailing_returns, risk_free_rate=reb_rf)
        
        # Map back to full ticker list
        current_weights = np.zeros(num_assets)
        for i, t in enumerate(tickers):
            if t in trailing_returns.columns:
                col_idx = trailing_returns.columns.tolist().index(t)
                if col_idx < len(new_weights):
                    current_weights[i] = new_weights[col_idx]
        
        total = np.sum(current_weights)
        if total > 0:
            current_weights = current_weights / total
        else:
            current_weights = np.ones(num_assets) / num_assets
        
        # Apply these weights from rebalance date until next rebalance
        weight_schedule.loc[reb_date:] = current_weights
    
    # Build portfolio with the weight schedule
    # For vectorbt, we use from_orders with targetpercent at each rebalance
    portfolio = vbt.Portfolio.from_orders(
        prices,
        size=weight_schedule.values,
        size_type='targetpercent',
        group_by=True,
        cash_sharing=True,
        init_cash=1000000,
        fees=0.001,
        freq='D'
    )
    
    return portfolio


def compute_metrics(portfolio=None, portfolio_history=None):
    """
    Compute a standardized set of performance metrics from either a vectorbt Portfolio
    or a raw portfolio value Series.
    
    Returns dict with: total_return, sharpe, sortino, calmar, max_drawdown, annualized_vol
    """
    if portfolio is not None:
        total_return = portfolio.total_return()
        sharpe = portfolio.sharpe_ratio()
        max_dd = portfolio.max_drawdown()
        
        # Compute Sortino and Calmar from the equity curve
        equity = portfolio.value()
        daily_returns = equity.pct_change().dropna()
    elif portfolio_history is not None:
        series = pd.Series(portfolio_history)
        daily_returns = series.pct_change().dropna()
        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        
        cum = (1 + daily_returns).cumprod()
        rolling_max = cum.cummax()
        drawdowns = cum / rolling_max - 1
        max_dd = drawdowns.min()
        
        sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
    else:
        return {k: np.nan for k in ['total_return', 'sharpe', 'sortino', 'calmar', 'max_drawdown', 'annualized_vol']}
    
    # Sortino: penalize only downside deviation
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
    sortino = np.sqrt(252) * daily_returns.mean() / (downside_std + 1e-8)
    
    # Calmar: annualized return / max drawdown
    ann_return = (1 + total_return) ** (252 / max(len(daily_returns), 1)) - 1
    calmar = ann_return / (abs(max_dd) + 1e-8)
    
    # Annualized Volatility
    ann_vol = daily_returns.std() * np.sqrt(252)
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_dd,
        'annualized_vol': ann_vol
    }


if __name__ == "__main__":
    processed_data_path = os.path.join(DATA_DIR, "processed_features.csv")
    if not os.path.exists(processed_data_path):
        logger.error("Processed data not found at %s. Run data_fetcher.py first.", processed_data_path)
    else:
        df = pd.read_csv(processed_data_path)
        
        # Buy & Hold
        bnh_portfolio = backtest_buy_and_hold(df)
        bnh_metrics = compute_metrics(portfolio=bnh_portfolio)
        logger.info("Buy & Hold Metrics: %s", bnh_metrics)
        
        # Rolling Markowitz
        mvo_portfolio = backtest_rolling_markowitz(df)
        mvo_metrics = compute_metrics(portfolio=mvo_portfolio)
        logger.info("Rolling Markowitz Metrics: %s", mvo_metrics)
