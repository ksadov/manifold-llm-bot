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
        if market.outcomeType == OutcomeType.BINARY and self.can_trade(
            market, bankroll
        ):
            self.logger.info(f"Trading on market: {market}")
            try:
                if self.max_trade_time is not None:
                    start_time = time.time()

                probability_estimate, reasoning = self.get_probability_estimate(market)

                if (
                    self.max_trade_time is not None
                    and time.time() - start_time > self.max_trade_time
                ):
                    self.logger.warning(
                        f"Timeout processing market {market.id}: exceeded {self.max_trade_time} seconds"
                    )
                    return

                self.logger.info(
                    f"Probability estimate for market {market.id}: {probability_estimate}"
                )

                if (
                    self.max_trade_time is not None
                    and time.time() - start_time > self.max_trade_time
                ):
                    self.logger.warning(
                        f"Timeout processing market {market.id}: exceeded {self.max_trade_time} seconds"
                    )
                    return

                bet_amount, bet_outcome = kelly_bet(
                    probability_estimate,
                    market.probability,
                    self.kelly_alpha,
                    bankroll,
                    self.max_trade_amount,
                )
                if bet_amount > 0:
                    if (
                        self.max_trade_time is not None
                        and time.time() - start_time > self.max_trade_time
                    ):
                        self.logger.warning(
                            f"Timeout processing market {market.id}: exceeded {self.max_trade_time} seconds"
                        )
                        return

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
                        if (
                            self.max_trade_time is not None
                            and time.time() - start_time > self.max_trade_time
                        ):
                            self.logger.warning(
                                f"Timeout commenting on market {market.id}: exceeded {self.max_trade_time} seconds"
                            )
                            return

                        place_comment(market.id, reasoning, self.manifold_api_key)
                        self.logger.info(f"Commented on market: {market.id}")
            except Exception as e:
                self.logger.error(f"Error trading on market {market.id}: {e}")

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            msg = json.loads(message)
            if (
                msg.get("type") == "broadcast"
                and msg.get("topic") == "global/new-contract"
            ):
                self.logger.info(f"Received new market notification: {msg}")
                # Get the full market details
                market_id = msg.get("data", {}).get("id")
                if market_id:
                    response = requests.get(API_BASE + "market/" + market_id)
                    if response.status_code == 200:
                        market = FullMarket(**response.json())
                        self.trade_on_market(market)
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")

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
        while self.is_running and self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                message = {"type": "ping", "txid": self.txid}
                self.ws.send(json.dumps(message))
                self.txid += 1
                time.sleep(30)  # Send ping every 30 seconds
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
    )
