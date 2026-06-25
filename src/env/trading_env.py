import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MultiAssetTradingEnv(gym.Env):
    """
    A custom multi-asset trading environment for RL.
    Follows Gymnasium API.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, tickers, initial_capital=1e6, transaction_cost_pct=0.001, cvar_lambda=2.0, render_mode=None):
        super(MultiAssetTradingEnv, self).__init__()
        self.df = df
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct  # base commission
        self.cvar_lambda = cvar_lambda
        self.render_mode = render_mode

        # Ensure df is sorted by Date and Ticker
        self.df = self.df.sort_values(by=['Date', 'Ticker'])
        self.dates = self.df['Date'].unique()
        
        self.num_assets = len(self.tickers)
        
        # State space: For each asset: Close, Log_Return, SMA_20, SMA_50, Parkinson_Vol, RSI_14, Div_Yield, Overnight_Ret, BA_Spread, Dollar_Volume
        # Plus current portfolio weights (num_assets + 1 for cash)
        # Plus current sector caps (num_assets)
        num_features_per_asset = 10
        self.obs_dim = self.num_assets * num_features_per_asset + (self.num_assets + 1) + self.num_assets
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Action space: Target weights for the num_assets (cash is 1 - sum(weights))
        # We allow negative weights? No, let's assume long-only for now, bounded [0, 1].
        # Softmax will be applied to output to ensure they sum to <= 1.
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float32
        )

        self.sector_caps = None
        self.recommended_safe_haven = 'CASH'
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initial state
        self.portfolio_value = self.initial_capital
        # Start 100% in cash
        self.weights = np.zeros(self.num_assets + 1)
        self.weights[-1] = 1.0 # Last element is cash
        
        # Memory to keep track of returns and actions
        self.asset_memory = [self.initial_capital]
        self.rewards_memory = []
        self.raw_returns_memory = []  # Raw portfolio returns BEFORE CVaR penalty (prevents recursive contamination)
        self.actions_memory = []
        self.portfolio_history = [self.initial_capital]
        self.date_memory = [self.dates[self.current_step]]
        self.last_features = np.zeros(self.num_assets * 10, dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        current_date = self.dates[self.current_step]
        day_data = self.df[self.df['Date'] == current_date]
        
        # Make sure data is in same order as tickers
        day_data = day_data.set_index('Ticker').reindex(self.tickers)
        
        # Extract all 10 institutional features
        features = day_data[['Close', 'Log_Return', 'SMA_20', 'SMA_50', 'Parkinson_Vol', 'RSI_14', 'Div_Yield', 'Overnight_Ret', 'BA_Spread', 'Dollar_Volume']].values.flatten()
        
        # Robust Imputation
        mask = np.isnan(features) | np.isinf(features)
        features[mask] = self.last_features[mask]
        self.last_features = features.copy()
        
        # Extract current caps
        caps = np.ones(self.num_assets, dtype=np.float32)
        if self.sector_caps is not None:
            for i, ticker in enumerate(self.tickers):
                if ticker in self.sector_caps:
                    caps[i] = self.sector_caps[ticker]
        
        # Append current weights and caps
        obs = np.concatenate([features, self.weights, caps])
        return obs.astype(np.float32)

    def step(self, actions):
        # Actions are target weights for the assets.
        # Apply softmax to ensure valid weights (or simple normalization)
        actions = np.clip(actions, 0, 1)
        target_weights = actions / (np.sum(actions) + 1e-8)
        
        # Apply LLM Strategist Constraints (Sector Caps)
        constraint_violation_penalty = 0.0
        if self.sector_caps is not None:
            excess_weight = 0.0
            for i, ticker in enumerate(self.tickers):
                if ticker in self.sector_caps:
                    if target_weights[i] > self.sector_caps[ticker]:
                        violation = target_weights[i] - self.sector_caps[ticker]
                        excess_weight += violation
                        constraint_violation_penalty += violation
                        target_weights[i] = self.sector_caps[ticker]
            
            # Re-normalize excess to the recommended safe haven instead of pure cash
            safe_ticker = getattr(self, 'recommended_safe_haven', 'CASH')
            if excess_weight > 0.0 and safe_ticker in self.tickers:
                idx = self.tickers.index(safe_ticker)
                cap = self.sector_caps.get(safe_ticker, 1.0)
                room = cap - target_weights[idx]
                if room > 0:
                    allocation = min(room, excess_weight)
                    target_weights[idx] += allocation
                    excess_weight -= allocation
        # Allow cash
        total_invested = np.sum(target_weights)
        if total_invested > 1.0:
            target_weights = target_weights / total_invested
            cash_weight = 0.0
        else:
            cash_weight = 1.0 - total_invested
            
        new_weights = np.append(target_weights, cash_weight)
        
        # Calculate turnover penalty (transaction costs)
        # Extract BA Spread for each asset (default to 10bps if missing)
        current_date = self.dates[self.current_step]
        day_data = self.df[self.df['Date'] == current_date].set_index('Ticker').reindex(self.tickers)
        ba_spread = day_data['BA_Spread'].fillna(0.001).values
        
        weight_changes = np.abs(new_weights[:-1] - self.weights[:-1])
        
        # Rigorous Transaction Cost Model:
        # 1. Base commission: self.transaction_cost_pct
        # 2. Half-spread: ba_spread / 2.0
        # 3. Market Impact (Quadratic slippage): weight_changes^2 * slippage_factor
        # Using slippage_factor = 0.01 means trading 100% of the portfolio into 1 asset incurs 100bps of purely slippage-based penalty.
        slippage_factor = 0.01 
        
        cost_per_asset = weight_changes * (ba_spread / 2.0 + self.transaction_cost_pct) + (weight_changes ** 2) * slippage_factor
        transaction_costs = np.sum(cost_per_asset) * self.portfolio_value
        
        self.portfolio_value -= transaction_costs
        
        self.weights = new_weights
        
        # Step forward in time
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        if not done:
            # Calculate new portfolio value based on next day returns
            # (We already have day_data from earlier in step)
            # Fill missing returns with 0.0 (neutral daily return)
            returns = day_data['Log_Return'].fillna(0.0).values
            
            # Simple return calculation: portfolio_value * sum(weight * exp(log_return))
            simple_returns = np.exp(returns) - 1
            portfolio_return = np.sum(self.weights[:-1] * simple_returns) # Cash has 0 return
            
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            
            # Store raw return BEFORE any penalty (critical: prevents recursive CVaR contamination)
            self.raw_returns_memory.append(portfolio_return)
            
            # --- CVaR Downside Penalty Math ---
            # IMPORTANT: Compute CVaR from raw_returns_memory, NOT rewards_memory.
            # rewards_memory contains past CVaR penalties, which would create a feedback loop.
            cvar_penalty = 0.0
            cvar_window = 60  # ~3-month rolling window (need sufficient samples for meaningful tail estimate)
            if len(self.raw_returns_memory) >= cvar_window:
                recent_returns = np.array(self.raw_returns_memory[-cvar_window:])
                # Calculate 95% VaR (5th percentile of worst returns)
                var_95 = np.percentile(recent_returns, 5)
                # CVaR (Expected Shortfall) = E[R | R <= VaR_95]
                tail_returns = recent_returns[recent_returns <= var_95]
                
                cvar_95 = np.mean(tail_returns) if len(tail_returns) > 0 else var_95
                
                # Apply penalty only if the tail is negative (losing money)
                if cvar_95 < 0:
                    cvar_penalty = cvar_95 * self.cvar_lambda
            
            # Asymmetric soft penalty for attempting to violate LLM sector caps.
            # A 0.01 multiplier means a 10% weight violation incurs a 0.001 (10 bps) daily return drag.
            soft_constraint_penalty = constraint_violation_penalty * 0.01

            reward = portfolio_return + cvar_penalty - soft_constraint_penalty
            self.portfolio_value = new_portfolio_value
            self.portfolio_history.append(self.portfolio_value)
        else:
            reward = 0
            
        self.asset_memory.append(self.portfolio_value)
        self.rewards_memory.append(reward)
        self.actions_memory.append(self.weights)
        
        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        turnover_scalar = np.sum(np.abs(new_weights[:-1] - self.weights[:-1]))
        info = {'portfolio_value': self.portfolio_value, 'turnover': turnover_scalar}
        
        return obs, reward, done, False, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Date: {self.dates[self.current_step]}, Value: {self.portfolio_value:.2f}")

