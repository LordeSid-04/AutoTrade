"""
Expert Data Generation via Multi-Policy Behavior Cloning

Generates the offline RL training dataset by running MULTIPLE distinct "behavior
experts" through the trading environment. CQL learns a conservative policy relative
to the behavior distribution it observes — if that distribution is narrow (one policy,
repeated), CQL is structurally bounded by that single expert. Diverse policies give
CQL room to learn which behaviors work best across different market conditions.

Expert Policies:
  1. Max-Sharpe Markowitz (low noise)   — optimal risk-adjusted expert
  2. Min-Variance Markowitz (low noise) — risk-averse alternative
  3. Inverse-Volatility Weighting       — momentum-adjacent heuristic
  4. Max-Sharpe Markowitz (high noise)  — high-exploration variant
  5. Equal-Weight Rebalancing           — naive sub-optimal baseline
"""

import os
import sys
import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.trading_env import MultiAssetTradingEnv
from src.regimes import REGIMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

TICKERS = [
    "SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"
]

LOOKBACK_DAYS = 252


def _get_risk_free_rate(df_full, current_date):
    """Get the period-appropriate risk-free rate from the dataset."""
    row = df_full[(df_full['Date'] < current_date) & (df_full['Ticker'] == TICKERS[0])]
    if len(row) > 0 and 'Risk_Free_Rate' in row.columns:
        rf = row['Risk_Free_Rate'].iloc[-1]
        if pd.notna(rf) and rf > 0:
            return float(rf)
    return 0.02  # Conservative fallback


def _get_trailing_prices(df_full, current_date, tickers):
    """Get trailing LOOKBACK_DAYS of wide-format close prices strictly before current_date."""
    prices_long = df_full[df_full['Date'] < current_date].copy()
    if len(prices_long) == 0:
        return None

    prices_wide = prices_long.pivot(index='Date', columns='Ticker', values='Close')
    prices_wide = prices_wide.reindex(columns=tickers)

    if len(prices_wide) < LOOKBACK_DAYS:
        return None

    prices_window = prices_wide.iloc[-LOOKBACK_DAYS:]
    valid_cols = prices_window.dropna(axis=1, how='all').columns
    if len(valid_cols) < 3:
        return None

    prices_clean = prices_window[valid_cols].ffill().dropna()
    if len(prices_clean) < 60:
        return None

    return prices_clean


def compute_max_sharpe_weights(df_full, current_date, tickers):
    """Max-Sharpe Markowitz weights using period-appropriate risk-free rate."""
    num_assets = len(tickers)
    equal_weight = np.ones(num_assets) / num_assets

    prices_clean = _get_trailing_prices(df_full, current_date, tickers)
    if prices_clean is None:
        return equal_weight

    rf = _get_risk_free_rate(df_full, current_date)

    try:
        mu = expected_returns.mean_historical_return(prices_clean)
        S = risk_models.CovarianceShrinkage(prices_clean).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 0.40))
        ef.max_sharpe(risk_free_rate=rf)
        cleaned = ef.clean_weights(cutoff=0.01)

        weights = np.zeros(num_assets)
        for i, ticker in enumerate(tickers):
            if ticker in cleaned:
                weights[i] = cleaned[ticker]
        total = np.sum(weights)
        return weights / total if total > 0 else equal_weight
    except Exception:
        return equal_weight


def compute_min_variance_weights(df_full, current_date, tickers):
    """Minimum-variance Markowitz weights — risk-averse expert."""
    num_assets = len(tickers)
    equal_weight = np.ones(num_assets) / num_assets

    prices_clean = _get_trailing_prices(df_full, current_date, tickers)
    if prices_clean is None:
        return equal_weight

    try:
        mu = expected_returns.mean_historical_return(prices_clean)
        S = risk_models.CovarianceShrinkage(prices_clean).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 0.40))
        ef.min_volatility()
        cleaned = ef.clean_weights(cutoff=0.01)

        weights = np.zeros(num_assets)
        for i, ticker in enumerate(tickers):
            if ticker in cleaned:
                weights[i] = cleaned[ticker]
        total = np.sum(weights)
        return weights / total if total > 0 else equal_weight
    except Exception:
        return equal_weight


def compute_inverse_vol_weights(df_full, current_date, tickers):
    """Inverse-volatility weighting — allocates more to lower-vol assets."""
    num_assets = len(tickers)
    equal_weight = np.ones(num_assets) / num_assets

    prices_clean = _get_trailing_prices(df_full, current_date, tickers)
    if prices_clean is None:
        return equal_weight

    try:
        returns = prices_clean.pct_change().dropna()
        vols = returns.std() * np.sqrt(252)
        inv_vols = 1.0 / (vols + 1e-8)

        weights = np.zeros(num_assets)
        for i, ticker in enumerate(tickers):
            if ticker in inv_vols.index:
                weights[i] = inv_vols[ticker]
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
            weights = np.clip(weights, 0, 0.40)
            weights = weights / np.sum(weights)
            return weights
        return equal_weight
    except Exception:
        return equal_weight


def compute_random_dirichlet_weights(df_full, current_date, tickers):
    """Pure random valid portfolio for high exploration of state-action space."""
    return np.random.dirichlet(np.ones(len(tickers)))

# --- Episode configurations ---
EPISODES = [
    {"name": "Max-Sharpe (low noise)",  "policy_fn": compute_max_sharpe_weights,  "noise_std": 0.01},
    {"name": "Min-Variance (low noise)", "policy_fn": compute_min_variance_weights, "noise_std": 0.01},
    {"name": "Inverse-Volatility",       "policy_fn": compute_inverse_vol_weights,  "noise_std": 0.02},
    {"name": "Max-Sharpe (high noise)",  "policy_fn": compute_max_sharpe_weights,  "noise_std": 0.08},
    {"name": "Dirichlet Random 1",       "policy_fn": compute_random_dirichlet_weights, "noise_std": 0.00},
    {"name": "Dirichlet Random 2",       "policy_fn": compute_random_dirichlet_weights, "noise_std": 0.00},
    {"name": "Equal-Weight (baseline)",  "policy_fn": None,                         "noise_std": 0.00},
]


def generate_expert_dataset():
    print("Loading historical data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "processed_features.csv"))
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Handle missing data explicitly upstream (forward-fill up to 5 days, then drop)
    df = df.sort_values(['Ticker', 'Date'])
    df = df.groupby('Ticker', group_keys=False).apply(lambda x: x.ffill(limit=5))
    df = df.dropna()
    df = df.sort_values(by=['Date', 'Ticker'])

    env = MultiAssetTradingEnv(df, TICKERS)

    observations = []
    actions = []
    rewards = []
    terminals = []

    total_expert = 0
    total_fallback = 0

    print(f"Generating Multi-Policy Expert Dataset ({len(EPISODES)} episodes)...")
    print(f"Using {LOOKBACK_DAYS}-day rolling windows with dynamic risk-free rate.\n")

    for ep_idx, ep_config in enumerate(EPISODES):
        ep_name = ep_config["name"]
        policy_fn = ep_config["policy_fn"]
        noise_std = ep_config["noise_std"]

        print(f"Episode {ep_idx+1}/{len(EPISODES)}: {ep_name} (noise std={noise_std})")

        obs, _ = env.reset()
        done = False
        step = 0
        ep_expert = 0
        ep_fallback = 0
        was_test = False

        while not done:
            current_date = env.dates[env.current_step]

            # Inject random synthetic constraints every 21 steps to train constraint-awareness
            if step % 21 == 0:
                if np.random.rand() < 0.5:
                    env.sector_caps = None
                else:
                    caps = {}
                    num_constrained = np.random.randint(1, 4)
                    assets_to_constrain = np.random.choice([t for t in TICKERS if t not in ['TLT', 'GLD', 'XLU', 'XLP']], num_constrained, replace=False)
                    for t in assets_to_constrain:
                        caps[t] = np.random.uniform(0.05, 0.20)
                    env.sector_caps = caps

            if policy_fn is None:
                # Equal-weight baseline: deterministic, no optimization
                expert_weights = np.ones(len(TICKERS)) / len(TICKERS)
                ep_fallback += 1
            else:
                expert_weights = policy_fn(df, current_date, TICKERS)
                is_equal = np.allclose(expert_weights, np.ones(len(TICKERS)) / len(TICKERS), atol=0.01)
                if is_equal:
                    ep_fallback += 1
                else:
                    ep_expert += 1

            # Add exploration noise
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=len(TICKERS))
                noisy_action = expert_weights + noise
                noisy_action = np.clip(noisy_action, 0, 1)
                noisy_action = noisy_action / (np.sum(noisy_action) + 1e-8)
            else:
                noisy_action = expert_weights

            next_obs, reward, done, _, _ = env.step(noisy_action)

            # Record the ACTUAL executed (clipped) weights, not the intended noisy action.
            # This teaches the RL model how to naturally satisfy the constraints in 'obs'.
            actual_action = env.weights[:-1].copy()

            # Check if current_date is in an Evaluation Regime (Val or Test)
            is_test = False
            for start, end in REGIMES.values():
                if start <= current_date <= end:
                    is_test = True
                    break

            if is_test:
                if not was_test and len(terminals) > 0:
                    terminals[-1] = 1
            else:
                observations.append(obs)
                actions.append(actual_action)
                rewards.append(reward)
                terminals.append(1 if done else 0)

            was_test = is_test
            obs = next_obs
            step += 1

            if step % 500 == 0:
                print(f"  Step {step} (Expert: {ep_expert}, Fallback: {ep_fallback})")

        total_expert += ep_expert
        total_fallback += ep_fallback
        print(f"  Episode complete: {step} steps (Expert: {ep_expert}, Fallback: {ep_fallback})\n")

    total_steps = total_expert + total_fallback
    print(f"Dataset Summary:")
    print(f"  Total transitions: {total_steps}")
    print(f"  Expert (optimized) actions: {total_expert} ({100*total_expert/max(total_steps,1):.1f}%)")
    print(f"  Fallback (equal-weight) actions: {total_fallback} ({100*total_fallback/max(total_steps,1):.1f}%)")
    print(f"  Expert/Fallback ratio: {total_expert/(total_fallback+1):.2f}")

    # Clean up
    # Ensure there are no NaNs or Infs left
    observations = np.nan_to_num(np.array(observations, dtype=np.float32), nan=np.nan, posinf=1e6, neginf=-1e6)
    actions = np.array(actions, dtype=np.float32)
    rewards_arr = np.array(rewards, dtype=np.float32)
    terminals_arr = np.array(terminals, dtype=np.float32)
    
    if np.isnan(observations).any():
        raise ValueError("NaNs found in observations after upstream cleaning! Fix data pipeline.")

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards_arr,
        terminals=terminals_arr
    )

    out_path = os.path.join(DATA_DIR, "expert_dataset.h5")
    dataset.dump(out_path)
    print(f"Expert Dataset saved to {out_path}")

    return dataset

if __name__ == "__main__":
    generate_expert_dataset()
