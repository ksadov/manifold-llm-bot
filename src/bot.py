import datetime
import json
import random
import threading
import time
from logging import Logger
from pathlib import Path
from typing import Optional

import requests
import websocket

from src.agent import init_pipeline
from src.calculations import kelly_bet
from src.manifold.constants import API_BASE, WS_URL
from src.manifold.types import FullMarket, OutcomeType
from src.manifold.utils import get_my_account, place_limit_order
from src.trade_database import MarketPositionDB


class Bot:
    def __init__(
        self,
        logger: Logger,
        manifold_api_key: str,
        predict_market: callable,
        market_filters: dict,
        max_trade_amount: Optional[int],
        kelly_alpha: float,
        expires_millis_after: Optional[int],
        dry_run: bool,
        db_path: Optional[str] = None,
        auto_sell_threshold: Optional[float] = None,
    ):
        self.logger = logger
        self.db = MarketPositionDB(db_path) if db_path else None
        self.db.init_db()
        self.predict_market = predict_market
        self.manifold_api_key = manifold_api_key
        self.market_filters = market_filters
        self.max_trade_amount = max_trade_amount
        self.kelly_alpha = kelly_alpha
        self.last_search_timestamp = None
        self.expires_millis_after = expires_millis_after
        self.dry_run = dry_run
        self.ws = None
        self.txid = 0
        self.ws_thread = None
        self.is_running = False
        self.auto_sell_threshold = auto_sell_threshold
        self.last_ack_time = time.time()
        # Add reconnection state management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_reconnect_delay = 10  # Start with 10 seconds instead of 5
        self.is_reconnecting = (
            False  # Flag to prevent multiple concurrent reconnections
        )

        account = get_my_account(self.manifold_api_key)
        self.user_id = account.id
        self.subscribed_topics = set()

    def get_my_positions(self):
        """Get positions from database and subscribe to market updates"""
        if self.db is None:
            self.logger.error("No database path provided, skipping position retrieval")
            return
        try:
            # Load positions from database instead of API
            saved_positions = self.db.get_all_positions()

            for i, position in enumerate(saved_positions):
                market_id = position.market_id
                self.subscribe_to_topics([f"contract/{market_id}/new-bet"])
                # we can only make 500 requests per minute
                if i > 0 and i % 500 == 0:
                    time.sleep(60)
            self.logger.info(f"Subscribed to {len(saved_positions)} positions")

        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")

    def handle_new_bet(self, market_id: str):
        """Handle new bet for a market we may have a stake in and sell if threshold reached"""
        self.logger.debug(f"Received position update for market {market_id}")
        position = self.db.get_position(market_id)
        if position is None:
            self.logger.debug(f"No active position for market {market_id}, skipping")
            return
        try:
            if self.auto_sell_threshold is None:
                self.logger.debug(f"No auto sell threshold set, skipping")
                return

            probability_response = requests.get(
                f"{API_BASE}market/{market_id}/prob",
                headers={"Authorization": f"Key {self.manifold_api_key}"},
            )
            probability = probability_response.json().get("prob", 0)

            if position.shares > 0:
                payout_percentage = (
                    probability if position.outcome == "YES" else 1 - probability
                )
                self.logger.debug(f"Payout percentage: {payout_percentage}")

                if payout_percentage >= self.auto_sell_threshold:
                    # Determine which outcome to sell
                    self.logger.info(
                        f"Selling position in market {market_id} at {payout_percentage}% profit"
                    )
                    response = requests.post(
                        f"{API_BASE}market/{market_id}/sell",
                        headers={"Authorization": f"Key {self.manifold_api_key}"},
                        json={"outcome": position.outcome},
                    )
                    if response.status_code == 200:
                        self.logger.info(
                            f"Successfully sold position in market {market_id}"
                        )
                        self.db.remove_position(market_id)
                        self.subscribe_to_topics(
                            [f"contract/{market_id}/new-bet"], unsubscribe=True
                        )
                    else:
                        self.logger.info(f"Response: {response.json()}")
                        self.logger.error(
                            f"Failed to sell position: {response.status_code}"
                        )
                else:
                    self.logger.debug(
                        f"Not selling position in market {market_id} at {payout_percentage}"
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
        bankroll = get_my_account(self.manifold_api_key).balance
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

            if probability_estimate == 0:
                # this breaks the API
                probability_estimate = 0.01

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
                self.subscribe_to_topics([f"contract/{market.id}/new-bet"])
                if self.db is not None:
                    self.db.add_position_limited(
                        market_id=market.id,
                        max_shares_outcome=bet_outcome,
                        total_shares=bet.shares,
                        last_bet_time=bet.createdTime,
                    )

        except Exception as e:
            self.logger.error(f"Error trading on market {market.id}: {e}")

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            msg = json.loads(message)
            self.logger.debug(f"Received WebSocket message: {msg}")  # Debug raw message

            if msg.get("type") == "ack":
                # Update last_ack_time when we receive a ack
                self.logger.debug(f"Received ack at {time.time()}")
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

                elif "new-bet" in msg.get("topic", ""):
                    self.logger.info(
                        f"Received position update for market {msg.get('topic')}"
                    )
                    market_id = msg.get("topic").split("/")[1]
                    self.handle_new_bet(market_id)
                else:
                    self.logger.info(f"Received message with topic: {msg.get('topic')}")
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}", exc_info=True)

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close with exponential backoff"""
        self.logger.info(
            f"WebSocket connection closed: {close_status_code} - {close_msg}"
        )

        # Clear connection state
        self.subscribed_topics.clear()
        self.ws = None
        self.txid = 0

        # Only attempt reconnection if bot is still running and we're not already reconnecting
        if self.is_running and not self.is_reconnecting:
            self.attempt_reconnection()

    def attempt_reconnection(self):
        """Handle reconnection with exponential backoff and maximum attempts"""
        if self.is_reconnecting:
            self.logger.debug("Reconnection already in progress, skipping")
            return

        self.is_reconnecting = True

        try:
            while (
                self.is_running
                and self.reconnect_attempts < self.max_reconnect_attempts
            ):
                self.reconnect_attempts += 1

                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.base_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
                    300,  # Cap at 5 minutes
                )
                # Add jitter to avoid thundering herd
                jitter = random.uniform(0.1, 0.3) * delay
                total_delay = delay + jitter

                self.logger.info(
                    f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                    f"in {total_delay:.1f} seconds..."
                )

                time.sleep(total_delay)

                if not self.is_running:
                    break

                try:
                    self.connect_websocket()
                    # Give connection time to establish
                    time.sleep(2)

                    # Check if connection was successful
                    if self.ws and self.ws.sock and self.ws.sock.connected:
                        self.logger.info("Reconnection successful!")
                        self.reconnect_attempts = 0  # Reset on successful connection
                        self.is_reconnecting = False
                        return
                    else:
                        self.logger.warning(
                            "Reconnection failed - connection not established"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Reconnection attempt {self.reconnect_attempts} failed: {e}"
                    )

            if self.reconnect_attempts >= self.max_reconnect_attempts:
                self.logger.error(
                    f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. "
                    f"Stopping reconnection attempts."
                )
                self.is_running = False

        finally:
            self.is_reconnecting = False

    def on_open(self, ws):
        """Handle WebSocket connection open"""
        self.logger.info("WebSocket connection established")
        # Reset reconnection state on successful connection
        self.reconnect_attempts = 0
        self.is_reconnecting = False

        # Subscribe to new markets
        self.subscribe_to_topics(["global/new-contract"])
        self.get_my_positions()
        # Start ping thread to keep connection alive
        threading.Thread(target=self.ping_thread, daemon=True).start()

    def subscribe_to_topics(self, topics, unsubscribe=False):
        """Subscribe to WebSocket topics"""
        if self.ws and self.ws.sock and self.ws.sock.connected:
            sub_type = "unsubscribe" if unsubscribe else "subscribe"
            # Filter topics based on current subscriptions
            topics_to_process = []
            for topic in topics:
                if unsubscribe:
                    if topic in self.subscribed_topics:
                        topics_to_process.append(topic)
                        self.subscribed_topics.remove(topic)
                else:
                    if topic not in self.subscribed_topics:
                        topics_to_process.append(topic)
                        self.subscribed_topics.add(topic)

            if topics_to_process:
                message = {
                    "type": sub_type,
                    "txid": self.txid,
                    "topics": topics_to_process,
                }
                self.ws.send(json.dumps(message))
                self.txid += 1
                self.logger.info(f"{sub_type}d to {topics_to_process}")

    def ping_thread(self):
        """Send periodic pings to keep the WebSocket connection alive"""
        self.last_ack_time = time.time()

        while self.is_running and self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                current_time = time.time()
                if current_time - self.last_ack_time > 120:
                    self.logger.warning(
                        "No ack received in 2 minutes, closing connection for reconnection..."
                    )
                    # Close the connection cleanly - this will trigger on_close which handles reconnection
                    self.ws.close()
                    break

                message = {"type": "ping", "txid": self.txid}
                self.ws.send(json.dumps(message))
                self.txid += 1
                self.logger.debug(f"Ping sent at {current_time}")
                time.sleep(30)
            except Exception as e:
                self.logger.error(f"Error in ping thread: {e}")
                # Close connection on ping errors too
                if self.ws:
                    self.ws.close()
                break

    def connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            # Make sure any existing websocket is properly closed
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass  # Ignore errors when closing
                self.ws = None

            self.logger.info("Establishing WebSocket connection...")
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
            raise

    def run(self):
        """Run the bot with WebSocket connection"""
        self.is_running = True
        self.connect_websocket()

        # Keep the main thread alive while the WebSocket runs in background
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Bot stopping due to keyboard interrupt")
            self.is_running = False
            if self.ws:
                self.ws.close()


def init_from_config(config_path: Path, log_level: str) -> Bot:
    predict_market, logger, _, _, _ = init_pipeline(config_path, log_level, "deploy")
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(config["secrets_path"], "r") as f:
        secrets = json.load(f)
    return Bot(
        logger=logger,
        manifold_api_key=secrets["manifold_api_key"],
        predict_market=predict_market,
        market_filters=config["market_filters"],
        max_trade_amount=config["bet"]["max_trade_amount"],
        kelly_alpha=config["bet"]["kelly_alpha"],
        expires_millis_after=config["bet"]["expires_millis_after"],
        dry_run=config["bet"]["dry_run"],
        auto_sell_threshold=config["auto_sell_threshold"],
        db_path=config["db_path"],
    )
