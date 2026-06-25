"""
Daily Live AI Portfolio Rebalance Executor

Orchestrates the full hierarchical pipeline:
1. LLM Strategist generates macro regime constraints via RAG
2. CQL offline RL agent generates target portfolio weights
3. Hierarchical constraint clipping enforces LLM caps
4. Alpaca brokerage executes the rebalanced portfolio

Usage:
    python src/live/execute_daily.py           # Dry run (no real trades)
    python src/live/execute_daily.py --live     # Submit real orders to Alpaca
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.live.alpaca_trader import AlpacaManager
from src.strategist.llm_agent import LLMStrategist
import d3rlpy
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ==========================================
# TICKERS - must match training environment exactly
# ==========================================
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"]
NUM_ASSETS = len(TICKERS)


def build_live_observation(trader, constraints=None):
    """
    Build a real observation vector from live market data.
    Must match the feature engineering exactly:
    [Close, Log_Return, SMA_20, SMA_50, Parkinson_Vol, RSI_14, Div_Yield, Overnight_Ret, BA_Spread, Dollar_Volume] per asset + current portfolio weights.
    """
    logger.info("Fetching live market data for observation vector...")

    # Fetch 60 trading days of history (enough for SMA_50 + RSI_14 warmup)
    df = yf.download(TICKERS, period="4mo", progress=False)

    if df.empty:
        logger.warning("Could not fetch live market data. Falling back to equal-weight.")
        return None

    features = []
    for ticker in TICKERS:
        try:
            close = df['Close'][ticker].dropna()
            if len(close) < 50:
                logger.warning("Insufficient data for %s (%d days). Using zeros.", ticker, len(close))
                features.extend([0.0] * 10)
                continue

            latest_close = close.iloc[-1]
            log_return = np.log(close.iloc[-1] / close.iloc[-2])
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            
            # Parkinson Volatility
            high = df['High'][ticker].dropna()
            low = df['Low'][ticker].dropna()
            high_low_ratio = np.where((high > 0) & (low > 0) & (high >= low), high / low, 1.0)
            park_var = (1.0 / (4.0 * np.log(2.0))) * (np.log(high_low_ratio) ** 2)
            park_vol = np.sqrt(pd.Series(park_var).rolling(20).mean().iloc[-1]) * np.sqrt(252)
            
            # RSI 14
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rs = gain / (loss + 1e-10)
            rsi_14 = 100 - (100 / (1 + rs))

            # New features
            div_yield = 0.0 # Proxy for live inference without paid data
            
            open_price = df['Open'][ticker].dropna()
            overnight_ret = (open_price.iloc[-1] / close.iloc[-2]) - 1
            
            ba_spread = 0.001 # Proxy typical ETF spread
            
            volume = df['Volume'][ticker].dropna()
            dollar_vol = (latest_close * volume.iloc[-1]) / 1e6

            features.extend([
                latest_close, log_return, sma_20, sma_50, 
                park_vol, rsi_14, div_yield, overnight_ret, 
                ba_spread, dollar_vol
            ])
        except Exception as e:
            logger.warning("Feature computation failed for %s: %s. Using zeros.", ticker, e)
            features.extend([0.0] * 10)

    # Append current portfolio weights (num_assets + 1 for cash)
    positions = trader.get_positions()
    status = trader.get_account_status()
    total_value = status.get('portfolio_value', 1.0)

    weights = []
    for ticker in TICKERS:
        if isinstance(positions, dict) and 'error' not in positions and ticker in positions:
            weights.append(positions[ticker]['market_value'] / total_value)
        else:
            weights.append(0.0)

    cash_weight = status.get('cash', total_value) / total_value
    weights.append(cash_weight)

    # Append sector caps from LLM Strategist
    caps = [1.0] * NUM_ASSETS
    if constraints and hasattr(constraints, 'sector_exposure_caps'):
        for i, ticker in enumerate(TICKERS):
            if ticker in constraints.sector_exposure_caps:
                caps[i] = constraints.sector_exposure_caps[ticker]

    obs = np.array(features + weights + caps, dtype=np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    expected_dim = NUM_ASSETS * 10 + (NUM_ASSETS + 1) + NUM_ASSETS  # 150 + 16 + 15 = 181
    if len(obs) != expected_dim:
        logger.warning("Observation dimension mismatch. Got %d, expected %d. Padding/truncating.", len(obs), expected_dim)
        if len(obs) < expected_dim:
            obs = np.pad(obs, (0, expected_dim - len(obs)))
        else:
            obs = obs[:expected_dim]

    logger.info("Live observation vector built successfully (%d features).", len(obs))
    return obs


def run_daily_execution(dry_run=True):
    print("=" * 50)
    print("STARTING LIVE AI PORTFOLIO REBALANCE")
    print("=" * 50)

    load_dotenv()
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")

    if not alpaca_key or not alpaca_secret:
        logger.critical("Alpaca API keys not found in .env file.")
        sys.exit(1)

    trader = AlpacaManager(alpaca_key, alpaca_secret, paper=True)
    status = trader.get_account_status()

    if "error" in status:
        logger.critical("Could not connect to Alpaca: %s", status['error'])
        sys.exit(1)

    logger.info("Alpaca Connected. Portfolio Value: $%s", f"{status['portfolio_value']:,.2f}")

    if dry_run:
        logger.info("DRY_RUN mode. No market orders will be submitted.")

    # ------------------------------------------
    # 1. Macro Analysis (LLM Strategist)
    # ------------------------------------------
    logger.info("Querying LLM Strategist for today's Macro Regime...")
    strategist = LLMStrategist()
    today_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    constraints, evidence = strategist.get_weekly_constraints(today_date)
    logger.info("LLM Regime Classification: %s", constraints.regime_classification.upper())
    logger.info("Volatility Target: %.2f", constraints.max_portfolio_volatility_target)
    logger.info("Recommended Safe Haven: %s", getattr(constraints, 'recommended_safe_haven', 'CASH'))

    # ------------------------------------------
    # 2. Alpha Generation (RL Agent)
    # ------------------------------------------
    logger.info("Booting Conservative Q-Learning (CQL) Engine...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cql_full.d3")

    live_obs = build_live_observation(trader, constraints)

    rl_succeeded = False
    if os.path.exists(model_path) and live_obs is not None:
        try:
            cql = d3rlpy.load_learnable(model_path)
            raw_action = cql.predict(np.array([live_obs]))[0]
            exp_action = np.exp(raw_action - np.max(raw_action))
            target_weights = exp_action / (np.sum(exp_action) + 1e-8)
            rl_succeeded = True
            logger.info("CQL inference successful.")
        except Exception as e:
            logger.error("RL Inference Failed: %s", e)
    else:
        if not os.path.exists(model_path):
            logger.warning("RL Model not found at %s.", model_path)
        if live_obs is None:
            logger.warning("Could not build live observation vector.")

    if not rl_succeeded:
        logger.info("Falling back to equal-weight allocation.")
        target_weights = np.ones(NUM_ASSETS) / NUM_ASSETS

    # ------------------------------------------
    # 3. Apply Hierarchical Constraints
    # ------------------------------------------
    logger.info("Applying LLM Sector Caps to RL Output...")
    excess_weight = 0.0
    for i, ticker in enumerate(TICKERS):
        if ticker in constraints.sector_exposure_caps:
            cap = constraints.sector_exposure_caps[ticker]
            if target_weights[i] > cap:
                excess_weight += target_weights[i] - cap
                target_weights[i] = cap

    safe_haven = getattr(constraints, 'recommended_safe_haven', 'CASH')
    if excess_weight > 0.0 and safe_haven in TICKERS:
        safe_idx = TICKERS.index(safe_haven)
        target_weights[safe_idx] += excess_weight
        logger.info("Reallocated %.1f%% excess to safe haven: %s", excess_weight * 100, safe_haven)

    total = np.sum(target_weights)
    if total > 1.0:
        # Only re-normalize if weights exceed 100%
        target_weights = target_weights / total

    target_dict = {TICKERS[i]: target_weights[i] for i in range(len(TICKERS)) if target_weights[i] > 0.01}
    logger.info("Final Target Portfolio Allocations:")
    for ticker, weight in target_dict.items():
        logger.info("  %s: %.1f%%", ticker, weight * 100)

    # ------------------------------------------
    # 4. Live Brokerage Execution
    # ------------------------------------------
    if dry_run:
        print("\nDRY RUN COMPLETE. System is primed and ready. Use --live flag to execute real trades.")
    else:
        logger.info("Executing Market Orders on Alpaca...")
        result = trader.execute_target_weights(target_dict)
        logger.info("Rebalance Complete! Submitted %d sells and %d buys.", result['sells'], result['buys'])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Execute daily AI portfolio rebalance")
    parser.add_argument("--live", action="store_true", help="Submit real orders to Alpaca (default: dry run)")
    args = parser.parse_args()

    run_daily_execution(dry_run=not args.live)
