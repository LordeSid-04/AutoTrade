"""
Walk-Forward Regime Ablation Study with Bootstrap Confidence Intervals and Multi-Seed Reporting

Tests strategies across 4 structurally distinct market regimes (including a validation tuning set).
Strategies:
1. Buy & Hold (equal-weight)
2. Rolling Markowitz MVO (monthly rebalance)
3. Equal-Weight Baseline (1/N allocation, no learning)
4. Offline CQL Alone (multi-seed)
5. Offline TD3+BC Alone (multi-seed)
6. Full Hierarchical System: LLM Strategist + CQL Executor (multi-seed)
7. Full Hierarchical System: LLM Strategist + TD3+BC Executor (multi-seed)
"""

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import d3rlpy

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rl.custom_encoders import register_custom_encoders
register_custom_encoders()

from src.eval.baselines import (
    backtest_buy_and_hold,
    backtest_rolling_markowitz,
    compute_metrics
)
from src.env.trading_env import MultiAssetTradingEnv
from src.strategist.llm_agent import LLMStrategist

from src.regimes import REGIMES

TICKERS = [
    "SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"
]

SEEDS = [42, 43, 44, 45, 46]

# ─── Bootstrap Confidence Interval Engine ───

def _compute_metrics_from_returns(daily_returns):
    """Compute the standard metric dict from a daily returns array/Series."""
    if len(daily_returns) < 2:
        return {k: np.nan for k in ['total_return', 'sharpe', 'sortino', 'calmar', 'max_drawdown', 'annualized_vol']}

    dr = np.asarray(daily_returns, dtype=np.float64)
    total_return = np.prod(1 + dr) - 1
    ann_vol = np.std(dr, ddof=1) * np.sqrt(252)
    sharpe = np.sqrt(252) * np.mean(dr) / (np.std(dr, ddof=1) + 1e-8)

    down = dr[dr < 0]
    down_std = np.std(down, ddof=1) if len(down) > 1 else 1e-8
    sortino = np.sqrt(252) * np.mean(dr) / (down_std + 1e-8)

    cum = np.cumprod(1 + dr)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    max_dd = np.min(drawdowns)

    ann_return = (1 + total_return) ** (252 / max(len(dr), 1)) - 1
    calmar = ann_return / (abs(max_dd) + 1e-8)

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_dd,
        'annualized_vol': ann_vol
    }


def bootstrap_ci(daily_returns, n_bootstrap=1000, ci=0.95, block_size=21):
    dr = np.asarray(daily_returns, dtype=np.float64)
    n = len(dr)

    if n < block_size * 2:
        point = _compute_metrics_from_returns(dr)
        return {k: {'point': v, 'ci_lo': np.nan, 'ci_hi': np.nan} for k, v in point.items()}

    rng = np.random.default_rng(seed=42)
    alpha = 1 - ci
    metric_names = ['total_return', 'sharpe', 'sortino', 'calmar', 'max_drawdown', 'annualized_vol']
    boot_samples = {m: [] for m in metric_names}

    n_blocks = int(np.ceil(n / block_size))

    for _ in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        resampled = dr[indices]
        metrics = _compute_metrics_from_returns(resampled)
        for m in metric_names:
            boot_samples[m].append(metrics[m])

    point_estimate = _compute_metrics_from_returns(dr)

    result = {}
    for m in metric_names:
        arr = np.array(boot_samples[m])
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            lo = np.percentile(arr, 100 * alpha / 2)
            hi = np.percentile(arr, 100 * (1 - alpha / 2))
        else:
            lo, hi = np.nan, np.nan
        result[m] = {'point': point_estimate[m], 'ci_lo': lo, 'ci_hi': hi}

    return result


# ─── Strategy Runners ───

def filter_by_date(df, start, end):
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    return df.loc[mask].copy()

import json

def get_cached_constraints(date_str, cache):
    """Finds the most recent cached constraint for the given date."""
    available_dates = sorted([d for d in cache.keys() if d <= date_str])
    if available_dates:
        return cache[available_dates[-1]]
    return None

def run_dumb_llm_baseline(df, cache_path):
    """
    Dumb LLM Baseline: Uses the LLM's regime call to scale exposure via a trivial rule.
    If risk-off: 50% Safe Haven, 50% Equal-Weight Equities.
    Else: 100% Equal-Weight Equities.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
            
    env = MultiAssetTradingEnv(df, TICKERS)
    obs, _ = env.reset()
    done = False
    
    current_month = None
    action = np.ones(len(TICKERS)) / len(TICKERS)
    
    while not done:
        current_date_str = env.dates[env.current_step]
        month_str = current_date_str[:7] # YYYY-MM
        
        # Re-evaluate rules monthly
        if month_str != current_month:
            current_month = month_str
            c = get_cached_constraints(current_date_str, cache)
            if c and c['regime'] == 'risk-off':
                safe_haven = c['safe_haven']
                if safe_haven in TICKERS:
                    sh_idx = TICKERS.index(safe_haven)
                    action = np.ones(len(TICKERS)) * (0.5 / (len(TICKERS) - 1))
                    action[sh_idx] = 0.5
            else:
                action = np.ones(len(TICKERS)) / len(TICKERS)
                
            action = action / np.sum(action)
            
        obs, reward, done, _, _ = env.step(action)

    portfolio_values = np.array(env.portfolio_history)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = compute_metrics(portfolio_history=env.portfolio_history)
    return metrics, daily_returns


def run_rl_in_env_multi_seed(df, algo_name, use_dynamic_cache=False, cache_path=None):
    """Runs an RL strategy across all configured seeds and aggregates the results.
       If use_dynamic_cache is True, updates sector_caps dynamically per month."""
    all_metrics = []
    all_daily_returns = []
    
    cache = {}
    if use_dynamic_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
    
    for seed in SEEDS:
        model_path = os.path.join(DATA_DIR, f"{algo_name.lower()}_full_seed_{seed}.d3")
        if not os.path.exists(model_path):
            logger.warning(f"Model for {algo_name} seed {seed} not found at {model_path}.")
            continue
            
        try:
            model = d3rlpy.load_learnable(model_path)
            env = MultiAssetTradingEnv(df, TICKERS)
            obs, _ = env.reset()
            done = False
            current_month = None
            
            while not done:
                current_date_str = env.dates[env.current_step]
                month_str = current_date_str[:7]
                
                # Dynamically update constraints at the start of each month
                if use_dynamic_cache and month_str != current_month:
                    current_month = month_str
                    c = get_cached_constraints(current_date_str, cache)
                    if c:
                        env.sector_caps = c['caps']
                        env.recommended_safe_haven = c['safe_haven']
                
                action = model.predict(np.array([obs]))[0]
                obs, reward, done, _, _ = env.step(action)

            portfolio_values = np.array(env.portfolio_history)
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

            metrics = compute_metrics(portfolio_history=env.portfolio_history)
            all_metrics.append(metrics)
            all_daily_returns.append(daily_returns)
        except Exception as e:
            logger.error(f"RL execution failed for {algo_name} seed {seed}: {e}")
            traceback.print_exc()
            
    if not all_metrics:
        return None, None

    # Aggregate metric means and stds
    aggregated_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if values:
            aggregated_metrics[key] = np.mean(values)
            aggregated_metrics[f"{key}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
        else:
            aggregated_metrics[key] = np.nan
            aggregated_metrics[f"{key}_std"] = np.nan

    # For daily returns, just return the mean across seeds
    mean_daily_returns = np.mean(all_daily_returns, axis=0)
    return aggregated_metrics, mean_daily_returns


def run_equal_weight_in_env(env):
    obs, _ = env.reset()
    done = False
    while not done:
        action = np.ones(len(TICKERS)) / len(TICKERS)
        obs, reward, done, _, _ = env.step(action)

    portfolio_values = np.array(env.portfolio_history)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = compute_metrics(portfolio_history=env.portfolio_history)
    return metrics, daily_returns


def _extract_portfolio_returns(portfolio):
    equity = portfolio.value()
    daily_returns = equity.pct_change().dropna().values
    return np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)


NAN_METRICS = {k: np.nan for k in ['total_return', 'sharpe', 'sortino', 'calmar', 'max_drawdown', 'annualized_vol']}

def run_ablation_table(df, enable_bootstrap=True, n_bootstrap=1000):
    print("\n" + "="*70)
    print("RUNNING MULTI-SEED WALK-FORWARD ABLATION TABLE")
    if enable_bootstrap:
        print(f"  with {n_bootstrap}-resample block-bootstrap 95% confidence intervals")
    print("="*70)

    results = []
    ci_results = []

    for regime_name, (start_date, end_date) in REGIMES.items():
        print(f"\n{'-'*60}")
        print(f"REGIME: {regime_name} ({start_date} to {end_date})")
        print(f"{'-'*60}")

        regime_df = filter_by_date(df, start_date, end_date)
        if len(regime_df) == 0:
            logger.warning("No data for regime. Skipping.")
            continue

        strategies = {}

        # --- Strategy 1: Buy & Hold ---
        print("  Running Buy and Hold Baseline...")
        try:
            bnh_portfolio = backtest_buy_and_hold(regime_df)
            bnh_metrics = compute_metrics(portfolio=bnh_portfolio)
            bnh_returns = _extract_portfolio_returns(bnh_portfolio)
            strategies["Buy & Hold"] = (bnh_metrics, bnh_returns)
        except Exception as e:
            strategies["Buy & Hold"] = (dict(NAN_METRICS), np.array([]))

        # --- Strategy 2: Rolling Markowitz MVO ---
        print("  Running Rolling Markowitz MVO Baseline...")
        prior_start = str(int(start_date[:4]) - 2) + start_date[4:]
        extended_df = filter_by_date(df, prior_start, end_date)
        try:
            mvo_portfolio = backtest_rolling_markowitz(extended_df)
            mvo_metrics = compute_metrics(portfolio=mvo_portfolio)
            mvo_returns = _extract_portfolio_returns(mvo_portfolio)
            strategies["Markowitz MVO (Rolling)"] = (mvo_metrics, mvo_returns)
        except Exception as e:
            strategies["Markowitz MVO (Rolling)"] = (dict(NAN_METRICS), np.array([]))

        # --- Strategy 3: Equal-Weight Baseline ---
        print("  Running Equal-Weight Baseline...")
        try:
            vanilla_env = MultiAssetTradingEnv(regime_df, TICKERS)
            vanilla_metrics, vanilla_returns = run_equal_weight_in_env(vanilla_env)
            strategies["Equal-Weight Baseline"] = (vanilla_metrics, vanilla_returns)
        except Exception as e:
            strategies["Equal-Weight Baseline"] = (dict(NAN_METRICS), np.array([]))

        # --- Strategy 4 & 5: Offline RL Alone (Multi-Seed) ---
        for algo in ["CQL", "TD3BC"]:
            print(f"  Running Offline {algo} Alone (multi-seed)...")
            algo_metrics, algo_returns = run_rl_in_env_multi_seed(regime_df, algo)
            if algo_metrics is None:
                strategies[f"Offline {algo} Only"] = (dict(NAN_METRICS), np.array([]))
            else:
                strategies[f"Offline {algo} Only"] = (algo_metrics, algo_returns)

        # --- Strategy 6: Dumb LLM + Static Rule Baseline ---
        print("  Running Dumb LLM + Static Rule Baseline...")
        llm_cache_path = os.path.join(DATA_DIR, "llm_monthly_cache.json")
        try:
            dumb_metrics, dumb_returns = run_dumb_llm_baseline(regime_df, llm_cache_path)
            strategies["Dumb LLM Baseline"] = (dumb_metrics, dumb_returns)
        except Exception as e:
            logger.error(f"Dumb LLM Baseline failed: {e}")
            strategies["Dumb LLM Baseline"] = (dict(NAN_METRICS), np.array([]))

        # --- Strategy 7 & 8: Full Hierarchical (LLM + RL) with Dynamic Rolling Constraints ---
        for algo in ["CQL", "TD3BC"]:
            print(f"  Running Full Hierarchical (LLM + {algo}) with Dynamic Constraints...")
            try:
                hier_metrics, hier_returns = run_rl_in_env_multi_seed(
                    regime_df, 
                    algo,
                    use_dynamic_cache=True,
                    cache_path=llm_cache_path
                )
                if hier_metrics is None:
                    strategies[f"Hierarchical LLM+{algo}"] = (dict(NAN_METRICS), np.array([]))
                else:
                    strategies[f"Hierarchical LLM+{algo}"] = (hier_metrics, hier_returns)
            except Exception as e:
                logger.error(f"Hierarchical system failed: {e}")
                strategies[f"Hierarchical LLM+{algo}"] = (dict(NAN_METRICS), np.array([]))

        # --- Collect results & bootstrap ---
        for strat_name, (metrics, daily_rets) in strategies.items():
            # Format multi-seed output "Mean ± Std" for RL strategies
            formatted_metrics = {}
            for k in ['total_return', 'sharpe', 'sortino', 'calmar', 'max_drawdown', 'annualized_vol']:
                if f"{k}_std" in metrics:
                    formatted_metrics[k] = f"{metrics[k]:.4f} ± {metrics[f'{k}_std']:.4f}"
                else:
                    formatted_metrics[k] = f"{metrics[k]:.4f}" if k in metrics and not np.isnan(metrics[k]) else "N/A"
                    
            results.append({"Regime": regime_name, "Strategy": strat_name, **formatted_metrics})

            if enable_bootstrap and len(daily_rets) > 0:
                ci = bootstrap_ci(daily_rets, n_bootstrap=n_bootstrap)
                for metric_name, ci_data in ci.items():
                    ci_results.append({
                        "Regime": regime_name,
                        "Strategy": strat_name,
                        "Metric": metric_name,
                        "Point": ci_data['point'],
                        "CI_Lo_2.5%": ci_data['ci_lo'],
                        "CI_Hi_97.5%": ci_data['ci_hi']
                    })

    # ─── Format and Display Point Estimate Table ───
    results_df = pd.DataFrame(results)
    display_cols = {
        'total_return': 'Total Return',
        'sharpe': 'Sharpe',
        'sortino': 'Sortino',
        'calmar': 'Calmar',
        'max_drawdown': 'Max Drawdown',
        'annualized_vol': 'Ann. Vol'
    }
    results_df = results_df.rename(columns=display_cols)

    print("\n" + "="*70)
    print("FINAL ABLATION TABLE (RL Models = Mean ± Std)")
    print("="*70)
    print(results_df.to_string(index=False))

    results_path = os.path.join(DATA_DIR, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # ─── Format and Display Bootstrap CI Table ───
    if enable_bootstrap and ci_results:
        ci_df = pd.DataFrame(ci_results)
        ci_path = os.path.join(DATA_DIR, "ablation_bootstrap_ci.csv")
        ci_df.to_csv(ci_path, index=False)

        print("\n" + "="*70)
        print("BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 block-bootstrap resamples)")
        print("="*70)

        for metric in ['sharpe', 'sortino', 'max_drawdown']:
            metric_ci = ci_df[ci_df['Metric'] == metric]
            if len(metric_ci) == 0:
                continue
            print(f"\n--- {metric.upper()} ---")
            for _, row in metric_ci.iterrows():
                lo = f"{row['CI_Lo_2.5%']:.4f}" if not np.isnan(row['CI_Lo_2.5%']) else "N/A"
                hi = f"{row['CI_Hi_97.5%']:.4f}" if not np.isnan(row['CI_Hi_97.5%']) else "N/A"
                pt = f"{row['Point']:.4f}" if not np.isnan(row['Point']) else "N/A"
                print(f"  {row['Regime']:>25s} | {row['Strategy']:<27s} | {pt}  [{lo}, {hi}]")

        print(f"\nFull CI table saved to {ci_path}")

    return results_df

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    processed_data_path = os.path.join(DATA_DIR, "processed_features.csv")
    if not os.path.exists(processed_data_path):
        logger.error(f"Processed data not found at {processed_data_path}. Run data_fetcher.py first.")
        sys.exit(1)

    df = pd.read_csv(processed_data_path)
    run_ablation_table(df, enable_bootstrap=True, n_bootstrap=1000)
