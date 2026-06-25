import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import d3rlpy

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.eval.baselines import (
    backtest_buy_and_hold,
    backtest_rolling_markowitz
)
from src.rl.custom_encoders import register_custom_encoders
register_custom_encoders()

from src.env.trading_env import MultiAssetTradingEnv
from src.strategist.llm_agent import LLMStrategist

from src.regimes import REGIMES

TICKERS = [
    "SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"
]

SEEDS = [42, 43, 44, 45, 46]

_MODEL_CACHE = {}

def get_cached_model(model_path):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = d3rlpy.load_learnable(model_path)
    return _MODEL_CACHE[model_path]

def run_rl_in_env_multi_seed(df, data_dir, algo_name, use_constraints=None, safe_haven=None):
    all_trajectories = []
    env_dates = None
    
    for seed in SEEDS:
        model_path = os.path.join(data_dir, f"{algo_name.lower()}_full_seed_{seed}.d3")
        if not os.path.exists(model_path):
            continue
            
        try:
            model = get_cached_model(model_path)
            
            env = MultiAssetTradingEnv(df, TICKERS)
            if use_constraints is not None:
                env.sector_caps = use_constraints
            if safe_haven is not None:
                env.recommended_safe_haven = safe_haven

            obs, _ = env.reset()
            done = False
            while not done:
                action = model.predict(np.array([obs]))[0]
                obs, reward, done, _, _ = env.step(action)
                
            all_trajectories.append(env.portfolio_history)
            env_dates = env.dates
        except Exception as e:
            logger.error(f"RL execution failed for {algo_name} seed {seed}: {e}")
            
    if not all_trajectories:
        return None, None
        
    # Average the portfolio trajectories across seeds
    min_len = min([len(t) for t in all_trajectories])
    truncated_trajectories = [t[:min_len] for t in all_trajectories]
    mean_trajectory = np.mean(truncated_trajectories, axis=0)
    
    return mean_trajectory, env_dates[:min_len]

def run_equal_weight_in_env(env):
    obs, _ = env.reset()
    done = False
    while not done:
        action = np.ones(len(TICKERS)) / len(TICKERS)
        obs, reward, done, _, _ = env.step(action)
    return env.portfolio_history

def generate_curves():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    processed_data_path = os.path.join(data_dir, "processed_features.csv")
    df = pd.read_csv(processed_data_path)
    
    try:
        strategist = LLMStrategist()
    except Exception as e:
        logger.error(f"Failed to initialize LLMStrategist: {e}")
        strategist = None
        
    all_curves = []

    for regime_name, (start_date, end_date) in REGIMES.items():
        print(f"Generating equity curves for {regime_name}...")
        regime_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        
        if len(regime_df) == 0:
            continue
            
        # 1. Buy & Hold
        try:
            bnh_portfolio = backtest_buy_and_hold(regime_df)
            bnh_val = bnh_portfolio.value()
            for date, val in bnh_val.items():
                all_curves.append({
                    "Regime": regime_name,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Strategy": "Buy & Hold",
                    "Value": val
                })
        except Exception as e:
            logger.error(f"Buy & Hold curve failed: {e}")

        # 2. Rolling Markowitz MVO
        try:
            prior_start = str(int(start_date[:4]) - 2) + start_date[4:]
            extended_df = df[(df['Date'] >= prior_start) & (df['Date'] <= end_date)].copy()
            mvo_portfolio = backtest_rolling_markowitz(extended_df)
            mvo_val = mvo_portfolio.value()
            mvo_val = mvo_val.loc[start_date:end_date]
            for date, val in mvo_val.items():
                all_curves.append({
                    "Regime": regime_name,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Strategy": "Markowitz MVO (Rolling)",
                    "Value": val
                })
        except Exception as e:
            logger.error(f"Markowitz MVO curve failed: {e}")

        # 3. Equal-Weight Baseline
        try:
            vanilla_env = MultiAssetTradingEnv(regime_df, TICKERS)
            vanilla_hist = run_equal_weight_in_env(vanilla_env)
            if vanilla_hist:
                dates = vanilla_env.dates
                for i, val in enumerate(vanilla_hist):
                    idx = min(i, len(dates)-1)
                    all_curves.append({
                        "Regime": regime_name,
                        "Date": dates[idx],
                        "Strategy": "Equal-Weight Baseline",
                        "Value": val
                    })
        except Exception as e:
            logger.error(f"Equal-Weight Baseline curve failed: {e}")

        # 4 & 5. Offline RL Only (Multi-Seed)
        for algo in ["CQL", "TD3BC"]:
            try:
                hist, dates = run_rl_in_env_multi_seed(regime_df, data_dir, algo, use_constraints=None)
                if hist is not None and dates is not None:
                    for i, val in enumerate(hist):
                        idx = min(i, len(dates)-1)
                        all_curves.append({
                            "Regime": regime_name,
                            "Date": dates[idx],
                            "Strategy": f"Offline {algo} Only",
                            "Value": val
                        })
            except Exception as e:
                logger.error(f"Offline {algo} curve failed: {e}")

        # 6 & 7. Full Hierarchical LLM+RL
        constraints = None
        if strategist is not None:
            try:
                constraints, _ = strategist.get_weekly_constraints(start_date)
            except Exception as e:
                logger.error(f"LLM Strategist constraints query failed for {regime_name}: {e}")

        for algo in ["CQL", "TD3BC"]:
            try:
                use_caps = constraints.sector_exposure_caps if constraints is not None else None
                use_sh = constraints.recommended_safe_haven if constraints is not None else None
                
                hist, dates = run_rl_in_env_multi_seed(
                    regime_df, 
                    data_dir,
                    algo,
                    use_constraints=use_caps,
                    safe_haven=use_sh
                )
                if hist is not None and dates is not None:
                    for i, val in enumerate(hist):
                        idx = min(i, len(dates)-1)
                        all_curves.append({
                            "Regime": regime_name,
                            "Date": dates[idx],
                            "Strategy": f"Hierarchical LLM+{algo}",
                            "Value": val
                        })
            except Exception as e:
                logger.error(f"Hierarchical {algo} curve failed: {e}")

    curves_df = pd.DataFrame(all_curves)
    curves_path = os.path.join(data_dir, "ablation_equity_curves.csv")
    curves_df.to_csv(curves_path, index=False)
    print(f"Equity curves saved to {curves_path}")

if __name__ == "__main__":
    generate_curves()
