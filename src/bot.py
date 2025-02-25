import json
import time
from typing import Optional
from pathlib import Path

from src.manifold.types import FullMarket, OutcomeType
from src.search import Search
from src.manifold.utils import get_newest, place_trade, place_comment
from src.agent import init_dspy


class Bot:
    def __init__(
        self,
        manifold_api_key: str,
        llm_config_path: str,
        search: Search,
        trade_loop_wait: int,
        get_newest_limit: int,
        max_trade_amount: Optional[int],
        comment_with_reasoning: bool,
    ):
        self.manifold_api_key = manifold_api_key
        self.search = search
        self.trade_loop_wait = trade_loop_wait
        self.get_newest_limit = get_newest_limit
        self.max_trade_amount = max_trade_amount
        self.comment_with_reasoning = comment_with_reasoning
        self.last_search_timestamp = None

        self.predict_market = init_dspy(llm_config_path, search)

    def get_probability_estimate(self, market: FullMarket):
        print(f"Analyzing market: {market}")
        prediction = self.predict_market(
            question=market.question, description=market.textDescription
        )
        print(f"end prediction: {prediction}")
        return prediction.predicted_probability, prediction.reasoning

    def trade_on_new_markets(self):
        markets = get_newest(self.get_newest_limit)
        print("Found markets:")
        for market in markets:
            print(market)
        # Filter out markets that have already been traded on
        markets = [
            market
            for market in markets
            if self.last_search_timestamp is None
            or market.createdTime > self.last_search_timestamp
        ]
        for market in markets:
            # only trade on binary markets
            print(market.outcomeType)
            if market.outcomeType == OutcomeType.BINARY:
                print(f"Trading on market: {market}")
                probability_estimate, reasoning = self.get_probability_estimate(market)
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


def init_from_config(config_path: Path) -> Bot:
    # Load config from file
    with open(config_path) as f:
        config = json.load(f)
    secrets_json_path = Path(config["secrets_path"])
    # Load secrets from file
    with open(secrets_json_path) as f:
        secrets = json.load(f)
    # Initialize search
    search = Search(
        secrets["google_cse_key"], secrets["google_cse_cx"], config["max_html_length"]
    )
    return Bot(
        manifold_api_key=secrets["manifold_api_key"],
        llm_config_path=config["llm_config_path"],
        search=search,
        trade_loop_wait=config["trade_loop_wait"],
        get_newest_limit=config["get_newest_limit"],
        max_trade_amount=config.get("max_trade_amount"),
        comment_with_reasoning=config["comment_with_reasoning"],
    )
