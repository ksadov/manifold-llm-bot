import json
import time
import datetime

from typing import Optional
from pathlib import Path
from logging import Logger

from src.calculations import kelly_fraction
from src.manifold.types import FullMarket, OutcomeType
from src.search import Search
from src.manifold.utils import (
    get_newest,
    place_limit_order,
    place_comment,
    get_my_account,
)
from src.agent import init_dspy


class Bot:
    def __init__(
        self,
        logger: Logger,
        manifold_api_key: str,
        llm_config_path: str,
        search: Search,
        trade_loop_wait: int,
        get_newest_limit: int,
        market_filters: dict,
        max_trade_amount: Optional[int],
        comment_with_reasoning: bool,
        kelly_alpha: float,
        expires_millis_after: Optional[int],
        dry_run: bool,
    ):
        self.logger = logger
        self.manifold_api_key = manifold_api_key
        self.search = search
        self.trade_loop_wait = trade_loop_wait
        self.get_newest_limit = get_newest_limit
        self.market_filters = market_filters
        self.max_trade_amount = max_trade_amount
        self.comment_with_reasoning = comment_with_reasoning
        self.kelly_alpha = kelly_alpha
        self.last_search_timestamp = None
        self.expires_millis_after = expires_millis_after
        self.dry_run = dry_run

        self.predict_market = init_dspy(llm_config_path, search, logger)

    def get_probability_estimate(self, market: FullMarket):
        prediction = self.predict_market(
            question=market.question,
            description=market.textDescription,
            current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
        )
        return prediction.answer, prediction.reasoning

    def can_trade(self, market: FullMarket):
        for exclude_group in self.market_filters.get("exclude_groups", []):
            if exclude_group in market.groupSlugs:
                return False
        return True

    def trade_on_new_markets(self):
        markets = get_newest(self.get_newest_limit)
        self.logger.debug(f"Found {len(markets)} new markets")
        # Filter out markets that have already been traded on
        markets = [
            market
            for market in markets
            if self.last_search_timestamp is None
            or market.createdTime > self.last_search_timestamp
        ]
        for market in markets:
            if market.outcomeType == OutcomeType.BINARY and self.can_trade(market):
                """
                self.logger.debug(f"Trading on market: {market}")
                probability_estimate, reasoning = self.get_probability_estimate(market)
                self.logger.debug(
                    f"Probability estimate for market {market.id}: {probability_estimate}"
                )
                """
                probability_estimate = 0.5
                reasoning = "Testing"
                bet_amount = max(
                    kelly_fraction(
                        probability_estimate, market.probability, self.kelly_alpha
                    )
                    * get_my_account(self.manifold_api_key).balance,
                    self.max_trade_amount,
                )
                bet = place_limit_order(
                    market.id,
                    probability_estimate,
                    bet_amount,
                    self.manifold_api_key,
                    expires_millis_after=self.expires_millis_after,
                    dry_run=self.dry_run,
                )
                self.logger.info(f"Placed trade: {bet}")
                if self.comment_with_reasoning:
                    place_comment(market.id, reasoning, self.manifold_api_key)
                    self.logger.info(f"Commented on market: {market.id}")
        if markets:
            self.last_search_timestamp = markets[0].createdTime

    def run(self):
        while True:
            try:
                self.trade_on_new_markets()
            except Exception as e:
                raise e
            time.sleep(self.trade_loop_wait)


def init_from_config(config_path: Path, logger: Logger) -> Bot:
    # Load config from file
    with open(config_path) as f:
        config = json.load(f)
    secrets_json_path = Path(config["secrets_path"])
    # Load secrets from file
    with open(secrets_json_path) as f:
        secrets = json.load(f)
    # Initialize search
    search = Search(
        secrets["google_api_key"],
        secrets["google_cse_cx"],
        config["max_search_results"],
        config["max_html_length"],
    )
    return Bot(
        logger=logger,
        manifold_api_key=secrets["manifold_api_key"],
        llm_config_path=config["llm_config_path"],
        search=search,
        trade_loop_wait=config["trade_loop_wait"],
        get_newest_limit=config["get_newest_limit"],
        market_filters=config["market_filters"],
        max_trade_amount=config["bet"]["max_trade_amount"],
        comment_with_reasoning=config["comment_with_reasoning"],
        kelly_alpha=config["bet"]["kelly_alpha"],
        expires_millis_after=config["bet"]["expires_millis_after"],
        dry_run=config["bet"]["dry_run"],
    )
