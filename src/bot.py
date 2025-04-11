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
    get_my_account,
    get_market_positions,
)
from src.agent import init_pipeline
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
        account = get_my_account(self.manifold_api_key)
        self.user_id = account.id

    def subscribe_to_bets(self, market_id: str):
        """Subscribe to new bets for a specific market"""
        topic = f"contract/{market_id}/new-bet"
        self.subscribe_to_topics([topic])

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
                self.subscribe_to_bets(market_id)
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

                if not self.dry_run:
                    # add position to database after successful trade
                    self.db.add_position(
                        market_id=market.id,
                        outcome=bet_outcome,
                        shares=bet_amount,
                        price=probability_estimate,
                    )
                    self.subscribe_to_bets(market.id)

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
        self.get_my_positions()
        # Start ping thread to keep connection alive
        threading.Thread(target=self.ping_thread, daemon=True).start()

    def subscribe_to_topics(self, topics, unsubscribe=False):
        """Subscribe to WebSocket topics"""
        if self.ws and self.ws.sock and self.ws.sock.connected:
            sub_type = "unsubscribe" if unsubscribe else "subscribe"
            message = {"type": sub_type, "txid": self.txid, "topics": topics}
            self.ws.send(json.dumps(message))
            self.txid += 1
            self.logger.info(f"{sub_type}d to {topics}")

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
                self.logger.debug(f"Ping sent at {current_time}")
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
