import requests
from typing import Optional
from src.manifold.types import LiteMarket, FullMarket, OutcomeType, Bet
from src.manifold.constants import API_BASE


def get_newest(
    limit: int, outcome_types: Optional[list[OutcomeType]] = None
) -> list[LiteMarket]:
    """
    Get the newest markets from the Manifold API.

    Args:
        limit: The number of markets to attempt to retrieve.
        outcome_types: The types of markets to retrieve.

    Returns:
        list[LiteMarket]: The newest markets.
    """
    query = {"sort": "created-time", "limit": limit}
    response = requests.get(API_BASE + "markets", params=query)
    response.raise_for_status()
    litemarkets = [LiteMarket(**market) for market in response.json()]
    fullmarkets = []
    for l in litemarkets:
        if not l.isResolved and (
            outcome_types is None or l.outcomeType in outcome_types
        ):
            response = requests.get(API_BASE + "market/" + l.id)
            fullmarkets.append(FullMarket(**response.json()))
    return fullmarkets


def place_trade(
    market_id: str,
    probability: float,
    max_trade_amount: int,
    manifold_api_key: str,
    expiresAt: Optional[int] = None,
    dry_run: bool = False,
) -> Bet:
    """
    Place trade based on the market and computed probability of YES
    """
    post_dict = {
        "amount": max_trade_amount,
        "probability": probability,
        "marketId": market_id,
        "limitProb": probability,
    }
    if expiresAt is not None:
        post_dict["expiresAt"] = expiresAt
    if dry_run:
        post_dict["dryRun"] = True
    headers = {"Authorization": f"Bearer {manifold_api_key}"}
    response = requests.post(API_BASE + "trade", json=post_dict, headers=headers)
    response.raise_for_status()
    return Bet(**response.json())


def place_comment(
    market_id: str,
    comment: str,
    manifold_api_key: str,
) -> None:
    """
    Place comment on the market
    """
    post_dict = {
        "contractId": market_id,
        "markdown": comment,
    }
    headers = {"Authorization": f"Bearer {manifold_api_key}"}
    response = requests.post(API_BASE + "comment", json=post_dict, headers=headers)
    response.raise_for_status()
