import json
import time
import datetime
import requests
import websocket
import threading

from typing import Optional
from pathlib import Path
from logging import Logger

from src.calculations import kelly_bet
from src.manifold.constants import WS_URL, API_BASE
from src.manifold.types import FullMarket, OutcomeType
from src.manifold.utils import (
    place_limit_order,
    place_comment,
    get_my_account,
)
from src.agent import init_pipeline


class Bot:
    def __init__(
        self,
        logger: Logger,
        manifold_api_key: str,
        predict_market: callable,
        trade_loop_wait: int,
        get_newest_limit: int,
        market_filters: dict,
        max_trade_amount: Optional[int],
        comment_with_reasoning: bool,
        kelly_alpha: float,
        expires_millis_after: Optional[int],
        dry_run: bool,
        max_trade_time: Optional[int] = None,
        auto_sell_threshold: Optional[float] = None,
    ):
        self.logger = logger
        self.predict_market = predict_market
        self.manifold_api_key = manifold_api_key
        self.trade_loop_wait = trade_loop_wait
        self.get_newest_limit = get_newest_limit
        self.market_filters = market_filters
        self.max_trade_amount = max_trade_amount
        self.comment_with_reasoning = comment_with_reasoning
        self.kelly_alpha = kelly_alpha
        self.last_search_timestamp = None
        self.expires_millis_after = expires_millis_after
        self.dry_run = dry_run
        self.max_trade_time = max_trade_time
        self.ws = None
        self.txid = 0
        self.ws_thread = None
        self.is_running = False
        self.auto_sell_threshold = auto_sell_threshold
        self.active_positions = {}
        self.last_ack_time = time.time()

    def subscribe_to_position_updates(self, market_id: str, user_id: str):
        """Subscribe to updates for a specific market position"""
        topic = f"contract/{market_id}/user-metrics/{user_id}"
        self.subscribe_to_topics([topic])

    def get_my_positions(self):
        """Get all current positions and subscribe to their updates"""
        try:
            # Get user ID from account info
            account = get_my_account(self.manifold_api_key)
            user_id = account.id

            # Get all bets
            response = requests.get(
                f"{API_BASE}bets",
                params={"userId": user_id, "limit": 5},
                headers={"Authorization": f"Key {self.manifold_api_key}"},
            )
            if response.status_code != 200:
                self.logger.error(f"Failed to get bets: {response.status_code}")
                return

            # Get unique market IDs from bets
            market_ids = set(bet["contractId"] for bet in response.json())

            # Get positions for each market
            for market_id in market_ids:
                response = requests.get(
                    f"{API_BASE}market/{market_id}/positions",
                    params={"userId": user_id},
                    headers={"Authorization": f"Key {self.manifold_api_key}"},
                )
                if response.status_code == 200:
                    positions = response.json()
                    if positions:  # If we have a position
                        self.active_positions[market_id] = positions[0]
                        self.subscribe_to_position_updates(market_id, user_id)

        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")

    def handle_position_update(self, market_id: str, position_data: dict):
        """Handle updates to a position and sell if threshold reached"""
        try:
            if self.auto_sell_threshold is None:
                return

            self.active_positions[market_id] = position_data

            # Calculate current percentage of max payout
            payout = position_data.get("payout", 0)
            invested = position_data.get("invested", 0)
            if invested > 0:
                payout_percentage = (payout / invested - 1) * 100

                if payout_percentage >= self.auto_sell_threshold:
                    # Determine which outcome to sell
                    outcome = position_data.get("maxSharesOutcome")
                    if outcome:
                        self.logger.info(
                            f"Selling position in market {market_id} at {payout_percentage}% profit"
                        )
                        response = requests.post(
                            f"{API_BASE}market/{market_id}/sell",
                            headers={"Authorization": f"Key {self.manifold_api_key}"},
                            json={"outcome": outcome},
                        )
                        if response.status_code == 200:
                            self.logger.info(
                                f"Successfully sold position in market {market_id}"
                            )
                        else:
                            self.logger.error(
                                f"Failed to sell position: {response.status_code}"
                            )

        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")

    def get_probability_estimate(self, market: FullMarket):
        prediction = self.predict_market(
            question=market.question,
            description=market.textDescription,
            current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            creatorUsername=market.creatorUsername,
            comments=market.comments,
        )
        return prediction.answer, prediction.reasoning

    def can_trade(self, market: FullMarket, bankroll: float):
        for exclude_group in self.market_filters.get("exclude_groups", []):
            if exclude_group in market.groupSlugs:
                return False
        if bankroll < self.max_trade_amount:
            return False
        return True

    def trade_on_market(self, market):
        """Trade on a single market"""
        bankroll = get_my_account(self.manifold_api_key)
        self.logger.debug(f"Evaluating market {market.id}: type={market.outcomeType}")

        if market.outcomeType != OutcomeType.BINARY:
            self.logger.debug(f"Skipping non-binary market {market.id}")
            return

        if not self.can_trade(market, bankroll):
            self.logger.debug(
                f"Market {market.id} failed trade criteria: bankroll={bankroll}, filters={self.market_filters}"
            )
            return

        self.logger.info(f"Trading on market: {market}")
        try:
            probability_estimate, reasoning = self.get_probability_estimate(market)

            self.logger.info(
                f"Probability estimate for market {market.id}: {probability_estimate}"
            )

            bet_amount, bet_outcome = kelly_bet(
                probability_estimate,
                market.probability,
                self.kelly_alpha,
                bankroll,
                self.max_trade_amount,
            )

            if bet_amount > 0:
                bet = place_limit_order(
                    market.id,
                    probability_estimate,
                    bet_amount,
                    bet_outcome,
                    self.manifold_api_key,
                    expires_millis_after=self.expires_millis_after,
                    dry_run=self.dry_run,
                )
                self.logger.info(f"Placed trade: {bet}")

                if self.comment_with_reasoning:
                    place_comment(market.id, reasoning, self.manifold_api_key)
                    self.logger.info(f"Commented on market: {market.id}")
        except Exception as e:
            self.logger.error(f"Error trading on market {market.id}: {e}")

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            msg = json.loads(message)
            self.logger.debug(f"Received WebSocket message: {msg}")  # Debug raw message

            if msg.get("type") == "ack":
                # Update last_ack_time when we receive a ack
                self.logger.info(f"Received ack at {time.time()}")
                self.last_ack_time = time.time()

            if msg.get("type") == "broadcast":
                if msg.get("topic") == "global/new-contract":
                    market_data = msg.get("data", {})
                    market_contract = market_data.get("contract")
                    market_id = market_contract.get("id") if market_contract else None
                    if not market_id:
                        self.logger.warning(
                            f"Received market data without ID: {market_data}"
                        )
                        return

                    self.logger.info(f"Received new market: {market_id}")

                    # Fetch full market data using the API
                    try:
                        response = requests.get(
                            f"{API_BASE}market/{market_id}",
                            headers={"Authorization": f"Key {self.manifold_api_key}"},
                        )
                        if response.status_code == 200:
                            full_market_data = response.json()
                            market = FullMarket(
                                **full_market_data
                            )  # Convert to FullMarket object
                            self.trade_on_market(market)
                        else:
                            self.logger.error(
                                f"Failed to fetch full market data: {response.status_code}"
                            )
                    except Exception as e:
                        self.logger.error(f"Error fetching full market data: {e}")

                elif msg.get("topic", "").startswith(
                    "contract/"
                ) and "user-metrics" in msg.get("topic", ""):
                    market_id = msg.get("topic").split("/")[1]
                    self.handle_position_update(market_id, msg.get("data", {}))
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}", exc_info=True)

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        self.logger.info(
            f"WebSocket connection closed: {close_status_code} - {close_msg}"
        )
        if self.is_running:
            self.logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self.connect_websocket()

    def on_open(self, ws):
        """Handle WebSocket connection open"""
        self.logger.info("WebSocket connection established")
        # Subscribe to new markets
        self.subscribe_to_topics(["global/new-contract"])
        # Start ping thread to keep connection alive
        threading.Thread(target=self.ping_thread, daemon=True).start()

    def subscribe_to_topics(self, topics):
        """Subscribe to WebSocket topics"""
        if self.ws and self.ws.sock and self.ws.sock.connected:
            message = {"type": "subscribe", "txid": self.txid, "topics": topics}
            self.ws.send(json.dumps(message))
            self.txid += 1
            self.logger.info(f"Subscribed to topics: {topics}")

    def ping_thread(self):
        """Send periodic pings to keep the WebSocket connection alive"""
        self.last_ack_time = time.time()  # Initialize when thread starts

        while self.is_running and self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                current_time = time.time()
                if current_time - self.last_ack_time > 120:
                    self.logger.warning("No ack received in 2 minutes, reconnecting...")
                    self.ws.close()
                    break

                message = {"type": "ping", "txid": self.txid}
                self.ws.send(json.dumps(message))
                self.txid += 1
                self.logger.info(f"Ping sent at {current_time}")
                time.sleep(30)
            except Exception as e:
                self.logger.error(f"Error in ping thread: {e}")
                break

    def connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            self.ws = websocket.WebSocketApp(
                WS_URL,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()
        except Exception as e:
            self.logger.error(f"Error connecting to WebSocket: {e}")

    def run(self):
        """Run the bot with WebSocket connection"""
        self.is_running = True
        self.connect_websocket()

        self.get_my_positions()

        # Keep the main thread alive while the WebSocket runs in background
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Bot stopping due to keyboard interrupt")
            self.is_running = False
            if self.ws:
                self.ws.close()


def init_from_config(
    config_path: Path, log_level: str, max_trade_time: Optional[int] = None
) -> Bot:
    predict_market, logger, _, _, _ = init_pipeline(config_path, log_level, "deploy")
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(config["secrets_path"], "r") as f:
        secrets = json.load(f)
    return Bot(
        logger=logger,
        manifold_api_key=secrets["manifold_api_key"],
        predict_market=predict_market,
        trade_loop_wait=config["trade_loop_wait"],
        get_newest_limit=config["get_newest_limit"],
        market_filters=config["market_filters"],
        max_trade_amount=config["bet"]["max_trade_amount"],
        comment_with_reasoning=config["comment_with_reasoning"],
        kelly_alpha=config["bet"]["kelly_alpha"],
        expires_millis_after=config["bet"]["expires_millis_after"],
        dry_run=config["bet"]["dry_run"],
        max_trade_time=max_trade_time,
        auto_sell_threshold=config["auto_sell_threshold"],
    )
