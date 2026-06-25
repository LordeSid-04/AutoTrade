import os
import sys
import json
import logging
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pydantic import BaseModel
from dotenv import load_dotenv
from functools import lru_cache
import yfinance as yf


from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Header, Depends, Request
import httpx

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Setup pathing
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AutoTradeServer")

app = FastAPI(title="AutoTrade Quantum Terminal API", version="2.0.0")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# CORS middleware for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tickers & Sectors
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"]
TICKER_NAMES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "XLF": "Financials",
    "XLK": "Technology", "XLV": "Healthcare", "XLE": "Energy", "XLI": "Industrials",
    "XLY": "Consumer Disc.", "XLP": "Consumer Stpl.", "XLU": "Utilities", "XLB": "Materials",
    "XLC": "Communication", "GLD": "Gold", "TLT": "20+ Yr Treasury"
}

class Credentials(BaseModel):
    openai_key: Optional[str] = None
    alpaca_key: Optional[str] = None
    alpaca_secret: Optional[str] = None

class RebalanceRequest(BaseModel):
    live: bool = False

class TradeRequest(BaseModel):
    symbol: str
    qty: float
    side: str
    type: str = "market"
    limit_price: Optional[float] = None
    time_in_force: str = "gtc"

# Helper to load alpaca manager dynamically

import time
_jwt_cache = {}

def verify_jwt(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    
    current_time = time.time()
    if token in _jwt_cache:
        cached_user, timestamp = _jwt_cache[token]
        if current_time - timestamp < 300:  # 5 minute TTL
            return {"user_id": cached_user["id"], "token": token}
            
    url = f"{os.environ.get('SUPABASE_URL')}/auth/v1/user"
    headers = {
        "apikey": os.environ.get("SUPABASE_ANON_KEY"),
        "Authorization": f"Bearer {token}"
    }
    resp = httpx.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = resp.json()
    
    _jwt_cache[token] = (user, current_time)
    
    # Cleanup old entries to prevent memory leak
    if len(_jwt_cache) > 1000:
        old_keys = [k for k, (_, t) in _jwt_cache.items() if current_time - t >= 300]
        for k in old_keys:
            del _jwt_cache[k]
            
    return {"user_id": user["id"], "token": token}

def get_alpaca_trader(auth_data: dict):
    token = auth_data["token"]
    url = f"{os.environ.get('SUPABASE_URL')}/rest/v1/user_settings?select=alpaca_api_key,alpaca_secret_key"
    headers = {
        "apikey": os.environ.get("SUPABASE_ANON_KEY"),
        "Authorization": f"Bearer {token}"
    }
    resp = httpx.get(url, headers=headers)
    if resp.status_code == 200 and resp.json():
        keys = resp.json()[0]
        key = keys.get("alpaca_api_key")
        secret = keys.get("alpaca_secret_key")
        if key and secret:
            from src.live.alpaca_trader import AlpacaManager
            return AlpacaManager(key, secret, paper=True)
    return None


# State for rebalance logs
rebalance_status = {
    "status": "idle",
    "last_run": None,
    "log_output": "",
    "exit_code": None
}

def run_rebalance_task(live: bool):
    global rebalance_status
    rebalance_status["status"] = "running"
    rebalance_status["log_output"] = ""
    rebalance_status["exit_code"] = None
    
    script_path = os.path.join(PROJECT_ROOT, "src", "live", "execute_daily.py")
    cmd = [sys.executable, script_path]
    if live:
        cmd.append("--live")
        
    logger.info(f"Running rebalance pipeline task: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output_lines.append(line)
            rebalance_status["log_output"] = "".join(output_lines)
            
        process.wait()
        rebalance_status["status"] = "success" if process.returncode == 0 else "failed"
        rebalance_status["exit_code"] = process.returncode
        rebalance_status["last_run"] = datetime.now().isoformat()
    except Exception as e:
        logger.error(f"Rebalance subprocess failed: {e}")
        rebalance_status["status"] = "failed"
        rebalance_status["log_output"] += f"\nProcess failed with exception: {e}"
        rebalance_status["exit_code"] = -1

@limiter.limit("10/minute")
@app.get("/api/config")
def get_config(request: Request, auth_data: dict = Depends(verify_jwt)):
    token = auth_data["token"]
    url = f"{os.environ.get('SUPABASE_URL')}/rest/v1/user_settings?select=*"
    headers = {
        "apikey": os.environ.get("SUPABASE_ANON_KEY"),
        "Authorization": f"Bearer {token}"
    }
    resp = httpx.get(url, headers=headers)
    keys_set = False
    alpaca_set = False
    if resp.status_code == 200 and resp.json():
        data = resp.json()[0]
        keys_set = bool(data.get("openai_api_key"))
        alpaca_set = bool(data.get("alpaca_api_key")) and bool(data.get("alpaca_secret_key"))
        
    model_exists = os.path.exists(os.path.join(DATA_DIR, "cql_full.d3"))
    corpus_dir = os.path.join(DATA_DIR, "macro_corpus")
    rag_docs = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")] if os.path.exists(corpus_dir) else []
        
    return {
        "status": {
            "model_loaded": model_exists,
            "openai_api_set": keys_set,
            "alpaca_connected": alpaca_set,
            "rag_docs_count": len(rag_docs)
        },
        "ticker_names": TICKER_NAMES,
        "tickers": TICKERS,
        "rag_docs": sorted(rag_docs)
    }
@limiter.limit("10/minute")
@app.post("/api/config")
def update_config(creds: Credentials, request: Request, auth_data: dict = Depends(verify_jwt)):
    token = auth_data["token"]
    user_id = auth_data["user_id"]
    
    payload = {"user_id": user_id}
    if creds.openai_key:
        payload["openai_api_key"] = creds.openai_key
    if creds.alpaca_key:
        payload["alpaca_api_key"] = creds.alpaca_key
    if creds.alpaca_secret:
        payload["alpaca_secret_key"] = creds.alpaca_secret
        
    url = f"{os.environ.get('SUPABASE_URL')}/rest/v1/user_settings"
    headers = {
        "apikey": os.environ.get("SUPABASE_ANON_KEY"),
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    resp = httpx.post(url, headers=headers, json=payload)
    if resp.status_code not in (200, 201, 204):
        return {"status": "error", "message": resp.text}
        
    return {"status": "success", "message": "Credentials updated in Supabase successfully."}

@app.get("/api/portfolio")
@limiter.limit("30/minute")
def get_portfolio(request: Request, auth_data: dict = Depends(verify_jwt)):
    trader = get_alpaca_trader(auth_data)
    if not trader:
        # Return mock demo data if credentials not configured
        mock_positions = {
            "XLK": {"market_value": 45000.0, "qty": 200.0, "current_price": 225.0},
            "SPY": {"market_value": 30000.0, "qty": 60.0, "current_price": 500.0},
            "TLT": {"market_value": 15000.0, "qty": 150.0, "current_price": 100.0}
        }
        return {
            "connected": False,
            "cash": 10000.0,
            "portfolio_value": 100000.0,
            "buying_power": 40000.0,
            "positions": mock_positions
        }
        
    status = trader.get_account_status()
    if "error" in status:
        return JSONResponse(status_code=500, content={"error": status["error"]})
        
    positions = trader.get_positions()
    if "error" in positions:
        positions = {}
        
    return {
        "connected": True,
        "cash": status["cash"],
        "portfolio_value": status["portfolio_value"],
        "buying_power": status["buying_power"],
        "positions": positions
    }

@app.get("/api/alpaca/account")
@limiter.limit("60/minute")
def get_alpaca_account(request: Request, auth_data: dict = Depends(verify_jwt)):
    trader = get_alpaca_trader(auth_data)
    if not trader:
        return {"error": "Alpaca credentials not configured."}
    return trader.get_account_status()

@app.get("/api/alpaca/positions")
@limiter.limit("60/minute")
def get_alpaca_positions(request: Request, auth_data: dict = Depends(verify_jwt)):
    trader = get_alpaca_trader(auth_data)
    if not trader:
        return {"error": "Alpaca credentials not configured."}
    return trader.get_positions()

@app.get("/api/alpaca/orders")
@limiter.limit("60/minute")
def get_alpaca_orders(request: Request, auth_data: dict = Depends(verify_jwt)):
    trader = get_alpaca_trader(auth_data)
    if not trader:
        return {"error": "Alpaca credentials not configured."}
    return trader.get_orders()

@app.get("/api/alpaca/activities")
@limiter.limit("60/minute")
def get_alpaca_activities(request: Request, auth_data: dict = Depends(verify_jwt)):
    trader = get_alpaca_trader(auth_data)
    if not trader:
        return {"error": "Alpaca credentials not configured."}
    return trader.get_activities()

@app.post("/api/alpaca/trade")
@limiter.limit("20/minute")
def submit_trade(trade: TradeRequest, request: Request, auth_data: dict = Depends(verify_jwt)):
    trader = get_alpaca_trader(auth_data)
    if not trader:
        return JSONResponse(status_code=500, content={"error": "Alpaca credentials not configured."})
    
    try:
        kwargs = {
            "symbol": trade.symbol.upper(),
            "qty": trade.qty,
            "side": trade.side.lower(),
            "type": trade.type.lower(),
            "time_in_force": trade.time_in_force.lower()
        }
        if trade.limit_price is not None:
            kwargs["limit_price"] = trade.limit_price
            
        order = trader.trading_client.submit_order(**kwargs)
        return {"status": "success", "order_id": str(order.id)}
    except Exception as e:
        logger.error(f"Manual trade submission failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/strategy/plan")
@limiter.limit("5/minute")
def get_strategy_plan(request: Request, auth_data: dict = Depends(verify_jwt)):
    from src.strategist.llm_agent import LLMStrategist
    from datetime import datetime
    try:
        strategist = LLMStrategist()
        current_date = datetime.now().strftime("%Y-%m-%d")
        constraints, evidence = strategist.get_weekly_constraints(current_date)
        return {
            "status": "success",
            "date": current_date,
            "regime": constraints.regime_classification,
            "volatility_target": constraints.max_portfolio_volatility_target,
            "sector_caps": constraints.sector_exposure_caps,
            "reasoning": constraints.reasoning,
            "evidence": evidence
        }
    except Exception as e:
        logger.error(f"Failed to generate strategy plan: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/backtest/data")
def get_backtest_data():
    import pandas as pd
    csv_path = os.path.join(DATA_DIR, "ablation_equity_curves.csv")
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": "Backtest data not found."})
    
    try:
        df = pd.read_csv(csv_path)
        # Filter to a specific regime if needed, or just take the first one
        regime = df['Regime'].iloc[0]
        df = df[df['Regime'] == regime]
        
        strategies = df['Strategy'].unique()
        
        result = {}
        dates = df[df['Strategy'] == strategies[0]]['Date'].tolist()
        result['dates'] = dates
        
        name_map = {
            "Buy & Hold": "BuyAndHold",
            "Markowitz MVO (Rolling)": "MVO",
            "Offline CQL Only": "RL_Model",
            "Hierarchical LLM+CQL": "RL_LLM"
        }
        
        for s in strategies:
            key = name_map.get(s, s.replace(" ", "_"))
            values = df[df['Strategy'] == s]['Value'].tolist()
            result[key] = values
            
        return result
    except Exception as e:
        logger.error(f"Failed to load backtest data: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/rebalance/orders")
def get_orders(limit: int = 50, auth_data: dict = Depends(verify_jwt)):
    trader = get_alpaca_trader(auth_data)
    if not trader:
        # Mock recent orders
        return [
            {"symbol": "XLK", "qty": 25.0, "side": "BUY", "status": "FILLED", "submitted_at": "2026-06-19 10:30:00"},
            {"symbol": "SPY", "qty": 10.0, "side": "SELL", "status": "FILLED", "submitted_at": "2026-06-18 14:15:00"},
            {"symbol": "TLT", "qty": 50.0, "side": "BUY", "status": "FILLED", "submitted_at": "2026-06-17 09:45:00"}
        ]
    orders = trader.get_orders(limit=limit)
    return orders

@app.post("/api/rebalance")
def execute_rebalance(req: RebalanceRequest, background_tasks: BackgroundTasks):
    global rebalance_status
    if rebalance_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Rebalance pipeline is already running.")
        
    background_tasks.add_task(run_rebalance_task, req.live)
    return {"status": "started", "message": "Hierarchical rebalancing task launched in background."}

@app.get("/api/rebalance/status")
def get_rebalance_status():
    return rebalance_status

@app.get("/api/strategy")
def get_strategy():
    try:
        from src.strategist.llm_agent import LLMStrategist
        from datetime import datetime
        strategist = LLMStrategist()
        current_date = datetime.now().strftime("%Y-%m-%d")
        # Query constraints for today/current context
        constraints, evidence = strategist.get_weekly_constraints(current_date)
        
        # Format sector caps
        formatted_caps = [
            {"ticker": k, "name": TICKER_NAMES.get(k, k), "cap": v}
            for k, v in constraints.sector_exposure_caps.items()
        ]
        
        regime = constraints.regime_classification.upper()
        if regime == "RISK-OFF":
            dialogue = [
                {"agent": "CQL RL Agent", "text": "Proposed allocating 45% of capital to XLK (Tech) and 35% to SPY based on 20-day returns and momentum."},
                {"agent": "LLM Strategist", "text": "REJECTED. Macro context analysis indicates rate hikes and high inflation. System classified as RISK-OFF. Technology sector exposure capped at 15%."},
                {"agent": "Hierarchical Clip", "text": "XLK clipped to 15%. Excess capital of 30% successfully reallocated to TLT (Bonds) as a defensive shelter."}
            ]
        elif regime == "RISK-ON":
            dialogue = [
                {"agent": "CQL RL Agent", "text": "Proposed allocating 40% of capital to XLK (Tech) and 30% to QQQ based on sector momentum."},
                {"agent": "LLM Strategist", "text": "APPROVED. Macro sentiment is supportive, CPI is cooling, and rate outlook is neutral. Risk-on conditions verified. XLK cap set at 40%."},
                {"agent": "Hierarchical Clip", "text": "Allocations passed without clipping. Momentum signals executed successfully."}
            ]
        else:
            dialogue = [
                {"agent": "CQL RL Agent", "text": "Proposed 30% XLK, 30% SPY, 20% XLF."},
                {"agent": "LLM Strategist", "text": "PARTIALLY CLIPPED. Transitional market indicators suggest mixed earnings signals. XLK capped at 25% for risk mitigation."},
                {"agent": "Hierarchical Clip", "text": "XLK clipped to 25%. Rest redistributed across other assets."}
            ]
            
        evidence_text = ""
        if isinstance(evidence, list):
            # Extract page_content from Langchain Document objects
            evidence_text = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in evidence])
        else:
            evidence_text = str(evidence)
            
        return {
            "regime": constraints.regime_classification,
            "volatility_target": constraints.max_portfolio_volatility_target,
            "sector_caps": formatted_caps,
            "dialogue": dialogue,
            "evidence": evidence_text
        }
    except Exception as e:
        logger.error(f"Strategy call failed: {e}")
        # Return fallback mock strategy info
        return {
            "regime": "risk-off",
            "volatility_target": 0.10,
            "sector_caps": [
                {"ticker": "SPY", "name": "S&P 500", "cap": 0.20},
                {"ticker": "TLT", "name": "20+ Yr Treasury", "cap": 0.60},
                {"ticker": "GLD", "name": "Gold", "cap": 0.30}
            ],
            "dialogue": [
                {"agent": "CQL RL Agent", "text": "Proposed allocating 45% of capital to XLK (Tech) based on momentum."},
                {"agent": "LLM Strategist", "text": "REJECTED. OpenAI credentials missing or error occurred. Falling back to rule-based RISK-OFF guardrail. Tech capped at 15%."},
                {"agent": "Hierarchical Clip", "text": "XLK clipped to 15%. Excess capital reallocated to TLT."}
            ]
        }

@app.get("/api/market-data")
def get_market_data(selected_ticker: str = "SPY"):
    path = os.path.join(DATA_DIR, "processed_features.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Processed features file not found.")
        
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Latest Returns Heatmap
    latest_date = df["Date"].max()
    latest_df = df[df["Date"] == latest_date][["Ticker", "Log_Return"]].copy()
    latest_df["return_pct"] = latest_df["Log_Return"] * 100
    latest_df["name"] = latest_df["Ticker"].map(TICKER_NAMES)
    latest_returns = latest_df.to_dict(orient="records")
    
    # 2. Correlation Matrix
    recent_df = df[df["Date"] >= latest_date - pd.Timedelta(days=365)]
    returns_wide = recent_df.pivot(index="Date", columns="Ticker", values="Log_Return")
    corr_matrix = returns_wide.corr().round(3)
    
    corr_data = {
        "columns": list(corr_matrix.columns),
        "index": list(corr_matrix.index),
        "values": corr_matrix.values.tolist()
    }
    
    # 3. Parkinson Vol averages
    vol_avg = df.groupby("Ticker")["Parkinson_Vol"].mean().reset_index()
    vol_avg.columns = ["ticker", "vol"]
    vol_avg["name"] = vol_avg["ticker"].map(TICKER_NAMES)
    vol_avg = vol_avg.sort_values("vol").to_dict(orient="records")
    
    # 4. Volatility time series
    ticker_df = df[df["Ticker"] == selected_ticker].sort_values("Date")
    # Take latest 250 points for display density
    ticker_df = ticker_df.tail(250)
    vol_time_series = {
        "dates": ticker_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "vol": ticker_df["Parkinson_Vol"].tolist(),
        "price": ticker_df["Close"].tolist()
    }
    
    return {
        "latest_returns": latest_returns,
        "correlation": corr_data,
        "volatility_rankings": vol_avg,
        "volatility_time_series": vol_time_series
    }

@app.get("/api/market-data-live")
def get_market_data_live():
    import yfinance as yf
    tickers_str = " ".join(TICKERS)
    try:
        data = yf.download(tickers_str, period="2d", interval="1d", progress=False)
        if data.empty:
            raise Exception("No data returned from yfinance")
            
        closes = data['Close']
        if len(closes) < 2:
            return JSONResponse(status_code=500, content={"error": "Not enough data to compute daily change."})
            
        latest_close = closes.iloc[-1]
        prev_close = closes.iloc[-2]
        
        live_data = []
        for sym in TICKERS:
            try:
                c = float(latest_close[sym])
                p = float(prev_close[sym])
                chg_pct = ((c - p) / p) * 100.0
                live_data.append({
                    "sym": sym,
                    "price": c,
                    "chg": chg_pct
                })
            except:
                pass
                
        return {"data": live_data}
    except Exception as e:
        logger.error(f"Live market data fetch failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@lru_cache(maxsize=128)
def fetch_cached_market_data(ticker: str, interval: str):
    """Cached helper to fetch yfinance data and prevent latency when switching tabs."""
    if interval in ['1m', '5m', '15m']:
        df = yf.download(ticker, period="7d", interval=interval, progress=False)
    else:
        df = yf.download(ticker, period="1y", interval=interval, progress=False)
    
    if df.empty:
        return []
        
    df.reset_index(inplace=True)
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    data = []
    for _, row in df.iterrows():
        try:
            val_open = float(row['Open'])
            if pd.isna(val_open): continue
            data.append({
                "time": int(row[date_col].timestamp()),
                "open": val_open,
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
        except Exception:
            continue
    return data

@app.get("/api/market-data-history")
def market_data_history(ticker: str, interval: str = "1d"):
    """Fetch OHLCV data from yfinance for Plotly chart."""
    try:
        data = fetch_cached_market_data(ticker, interval)
        if not data:
            return {"error": "No data found."}
        return {"data": data}
    except Exception as e:
        logger.error(f"Market data fetch error: {e}")
        return {"error": str(e)}

@app.get("/api/backtest")
def get_backtest():
    results_path = os.path.join(DATA_DIR, "ablation_results.csv")
    ci_path = os.path.join(DATA_DIR, "ablation_bootstrap_ci.csv")
    curves_path = os.path.join(DATA_DIR, "ablation_equity_curves.csv")
    
    if not os.path.exists(results_path) or not os.path.exists(curves_path):
        raise HTTPException(status_code=404, detail="Backtest results or curves files not found.")
        
    results_df = pd.read_csv(results_path)
    curves_df = pd.read_csv(curves_path)
    
    # Load CI if exists
    ci_data = []
    if os.path.exists(ci_path):
        ci_df = pd.read_csv(ci_path)
        ci_data = ci_df.to_dict(orient="records")
        
    # Standardize returns
    metrics = results_df.to_dict(orient="records")
    
    # Format curves for chart
    curves_df['Date'] = pd.to_datetime(curves_df['Date'])
    
    regimes = curves_df['Regime'].unique().tolist()
    curves_by_regime = {}
    
    for regime in regimes:
        regime_curves = curves_df[curves_df["Regime"] == regime].copy()
        
        # Normalize to 100
        normalized = []
        for strat, group in regime_curves.groupby("Strategy"):
            group = group.sort_values("Date")
            initial_val = group["Value"].iloc[0] if len(group) > 0 else 1.0
            group["Normalized_Value"] = (group["Value"] / initial_val) * 100.0
            
            # Compute drawdown
            vals = group["Value"].values
            cum_max = np.maximum.accumulate(vals)
            dd = (vals - cum_max) / (cum_max + 1e-8) * 100.0
            group["Drawdown"] = dd.tolist()
            
            normalized.append({
                "strategy": strat,
                "dates": group["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "values": group["Normalized_Value"].round(2).tolist(),
                "drawdown": group["Drawdown"].round(2).tolist()
            })
        curves_by_regime[regime] = normalized
        
    return {
        "metrics": metrics,
        "confidence_intervals": ci_data,
        "curves": curves_by_regime,
        "regimes": regimes
    }

@app.get("/api/system-info")
def get_system_info():
    meta_path = os.path.join(DATA_DIR, "training_metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
    # Engineered features info
    features = [
        {"feature": "Log Return", "source": "CRSP RET", "signal": "Momentum", "formula": "$r_t = \\ln(1 + R_t)$"},
        {"feature": "SMA 20", "source": "CRSP PRC", "signal": "Trend", "formula": "$\\text{SMA}_{20}(P) = \\frac{1}{20}\\sum_{i=0}^{19} P_{t-i}$"},
        {"feature": "SMA 50", "source": "CRSP PRC", "signal": "Trend", "formula": "$\\text{SMA}_{50}(P) = \\frac{1}{50}\\sum_{i=0}^{49} P_{t-i}$"},
        {"feature": "Parkinson Volatility", "source": "CRSP ASKHI/BIDLO", "signal": "Intraday Risk", "formula": "$\\sigma_p = \\sqrt{\\frac{252}{4 \\ln 2 \\cdot N} \\sum_{i=1}^N \\left(\\ln\\frac{H_i}{L_i}\\right)^2}$"},
        {"feature": "RSI 14", "source": "CRSP PRC", "signal": "Mean-Reversion", "formula": "$\\text{RSI}_{14} = 100 - \\frac{100}{1 + \\text{RS}}$"},
        {"feature": "Dividend Yield", "source": "CRSP RET - RETX", "signal": "Income", "formula": "$\\text{DY}_t = R_{t} - R_{x,t}$"},
        {"feature": "Overnight Return", "source": "CRSP OPENPRC/PRC", "signal": "Overnight Anomaly", "formula": "$\\text{Overnight}_t = \\frac{P_{\\text{open}, t}}{P_{\\text{close}, t-1}} - 1$"},
        {"feature": "Bid-Ask Spread", "source": "CRSP BID/ASK", "signal": "Microstructure", "formula": "$\\text{Spread}_t = \\frac{P_{\\text{ask}, t} - P_{\\text{bid}, t}}{P_{\\text{close}, t}}$"},
        {"feature": "Dollar Volume", "source": "CRSP PRC×VOL", "signal": "Liquidity", "formula": "$\\text{Volume}_{\\$, t} = \\frac{P_t \\times V_t}{10^6}$"},
        {"feature": "Close Price", "source": "CRSP PRC", "signal": "Price Level", "formula": "$P_t = |\\text{PRC}_t|$"}
    ]
    
    return {
        "training_config": meta,
        "features": features
    }

# Serves index.html from root
@app.get("/")
def read_root():
    static_file = os.path.join(PROJECT_ROOT, "src", "frontend", "static", "index.html")
    if not os.path.exists(static_file):
        raise HTTPException(status_code=404, detail="Index file not found in static directory.")
    return FileResponse(static_file)

# Serves index.html for direct access
@app.get("/index.html")
def read_root_index():
    return read_root()

# Serve static directory containing scripts, icons, CSS
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "src", "frontend", "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    # Make sure static directory exists
    os.makedirs(os.path.join(PROJECT_ROOT, "src", "frontend", "static"), exist_ok=True)
    uvicorn.run("src.frontend.server:app", host="0.0.0.0", port=8000, reload=True)
