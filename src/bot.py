import json
import time
from typing import Optional
from pathlib import Path
from logging import Logger
from venv import logger

from src.manifold.types import FullMarket, OutcomeType
from src.search import Search
from src.manifold.utils import get_newest, place_trade, place_comment
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
    ):
        self.logger = logger
        self.manifold_api_key = manifold_api_key
        self.search = search
        self.trade_loop_wait = trade_loop_wait
        self.get_newest_limit = get_newest_limit
        self.market_filters = market_filters
        self.max_trade_amount = max_trade_amount
        self.comment_with_reasoning = comment_with_reasoning
        self.last_search_timestamp = None

        self.predict_market = init_dspy(llm_config_path, search, logger)

    def get_probability_estimate(self, market: FullMarket):
        prediction = self.predict_market(
            question=market.question, description=market.textDescription
        )
        return prediction.predicted_probability, prediction.reasoning

    def can_trade(self, market: FullMarket):
        for exclude_group in self.market_filters.get("exclude_groups", []):
            if exclude_group in market.groupSlugs:
                return False
        return True

    def trade_on_new_markets(self):
        markets = get_newest(self.get_newest_limit)
        logger.debug(f"Found {len(markets)} new markets")
        # Filter out markets that have already been traded on
        markets = [
            market
            for market in markets
            if self.last_search_timestamp is None
            or market.createdTime > self.last_search_timestamp
        ]
        for market in markets:
            if market.outcomeType == OutcomeType.BINARY and self.can_trade(market):
                logger.debug(f"Trading on market: {market}")
                probability_estimate, reasoning = self.get_probability_estimate(market)
                logger.debug(
                    f"Probability estimate for market {market.id}: {probability_estimate}"
                )
                """
                bet = place_trade(
                    market.id,
                    self.manifold_api_key,
                    probability_estimate,
                    self.max_trade_amount,
                )
                if self.comment_with_reasoning:
                    place_comment(market.id, reasoning, self.manifold_api_key)
                """
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
        max_trade_amount=config.get("max_trade_amount"),
        comment_with_reasoning=config["comment_with_reasoning"],
    )
