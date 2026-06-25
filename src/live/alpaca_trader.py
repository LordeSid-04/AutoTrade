"""
Alpaca Brokerage Manager

Handles all live brokerage operations:
- Account status queries
- Position management
- Order execution (market orders)
- Target-weight rebalancing

All output uses the logging module to prevent accidental secret leakage
through bare print() statements.
"""

import time
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logger = logging.getLogger(__name__)


class AlpacaManager:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)
        logger.info("Alpaca client initialized (paper=%s)", paper)

    def get_account_status(self):
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
            }
        except Exception as e:
            logger.error("Failed to fetch account status: %s", e)
            return {"error": str(e)}

    def get_positions(self):
        try:
            positions = self.client.get_all_positions()
            formatted = {}
            for pos in positions:
                formatted[pos.symbol] = {
                    "market_value": float(pos.market_value),
                    "qty": float(pos.qty),
                    "current_price": float(pos.current_price)
                }
            return formatted
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return {"error": str(e)}

    def get_orders(self, limit=20):
        try:
            from alpaca.trading.requests import GetOrdersRequest
            req = GetOrdersRequest(status='all', limit=limit)
            orders = self.client.get_orders(req)
            formatted = []
            for o in orders:
                formatted.append({
                    "symbol": o.symbol,
                    "qty": o.qty,
                    "side": o.side.name if hasattr(o.side, 'name') else str(o.side),
                    "status": o.status.name if hasattr(o.status, 'name') else str(o.status),
                    "submitted_at": str(o.submitted_at)
                })
            return formatted
        except Exception as e:
            logger.error("Failed to fetch orders: %s", e)
            return []

    def get_activities(self):
        try:
            from alpaca.trading.requests import GetAccountActivitiesRequest
            from alpaca.trading.enums import ActivityType
            req = GetAccountActivitiesRequest(
                activity_types=[ActivityType.FILL, ActivityType.DIV, ActivityType.INT, ActivityType.TRANS]
            )
            activities = self.client.get_account_activities(req)
            formatted = []
            for a in activities:
                formatted.append({
                    "activity_type": str(a.activity_type),
                    "symbol": a.symbol,
                    "qty": a.qty,
                    "price": a.price,
                    "side": a.side,
                    "date": str(a.transaction_time)
                })
            return formatted
        except Exception as e:
            logger.error("Failed to fetch activities: %s", e)
            return []

    def execute_target_weights(self, target_weights_dict: dict):
        """
        Executes trades to align the portfolio with the target weights.
        Sells excess positions first, then buys deficit positions.
        """
        account = self.client.get_account()
        total_equity = float(account.portfolio_value)

        current_positions = self.get_positions()

        sells = []
        buys = []

        for symbol, target_weight in target_weights_dict.items():
            target_value = total_equity * target_weight

            current_value = 0.0
            if symbol in current_positions and isinstance(current_positions, dict) and 'error' not in current_positions:
                current_value = current_positions[symbol]["market_value"]

            delta = target_value - current_value

            if delta < -10.0:  # Sell threshold
                sells.append((symbol, round(abs(delta), 2)))
            elif delta > 10.0:  # Buy threshold
                buys.append((symbol, round(delta, 2)))

        # Execute Sells First
        for symbol, notional in sells:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            try:
                self.client.submit_order(req)
                logger.info("SELL %s $%.2f submitted", symbol, notional)
            except Exception as e:
                logger.error("Failed to sell %s: %s", symbol, e)

        time.sleep(2)  # Wait for sells to clear buying power

        # Execute Buys
        for symbol, notional in buys:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            try:
                self.client.submit_order(req)
                logger.info("BUY %s $%.2f submitted", symbol, notional)
            except Exception as e:
                logger.error("Failed to buy %s: %s", symbol, e)

        logger.info("Rebalance complete: %d sells, %d buys", len(sells), len(buys))
        return {"status": "success", "sells": len(sells), "buys": len(buys)}
