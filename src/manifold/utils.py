import requests
from typing import Optional, List
from src.manifold.types import (
    LiteMarket,
    FullMarket,
    OutcomeType,
    Bet,
    User,
    MarketPosition,
)
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


def place_limit_order(
    market_id: str,
    probability: float,
    amount: int,
    binary_outcome: str,
    manifold_api_key: str,
    expires_millis_after: Optional[int] = None,
    dry_run: bool = False,
) -> Bet:
    """
    Place limit order based on the market and computed probability of YES
    """
    post_dict = {
        "amount": amount,
        "contractId": market_id,
        "limitProb": probability,
        "outcome": binary_outcome,
    }
    if expires_millis_after:
        post_dict["expiresMillisAfter"] = expires_millis_after
    if dry_run:
        post_dict["dryRun"] = True
    headers = {"Authorization": f"Key {manifold_api_key}"}
    response = requests.post(API_BASE + "bet", json=post_dict, headers=headers)
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


def get_my_account(manifold_api_key: str) -> User:
    """
    Get the User associated with the Manifold API key
    """
    header = {"Authorization": f"Key {manifold_api_key}"}
    response = requests.get(API_BASE + "me", headers=header)
    response.raise_for_status()
    return User(**response.json())


def get_market_positions(market_id: str, **kwargs) -> List[MarketPosition]:
    """
    Get the positions for a market
    """
    response = requests.get(
        API_BASE + "market/" + market_id + "/positions", params=kwargs
    )
    response.raise_for_status()
    market_positions = []
    for position in response.json():
        print("position", position)
        market_positions.append(MarketPosition(**position))
    return market_positions


def get_bets(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    contract_id: Optional[str] = None,
    contract_slug: Optional[str] = None,
    limit: Optional[int] = 1000,
    before: Optional[str] = None,
    after: Optional[str] = None,
    before_time: Optional[int] = None,
    after_time: Optional[int] = None,
    kinds: Optional[List[str]] = None,
    order: Optional[str] = "desc",
) -> List[Bet]:
    """
    Get bets from the Manifold API
    """
    param_dict = {
        "userId": user_id,
        "username": username,
        "contractId": contract_id,
        "contractSlug": contract_slug,
        "limit": limit,
        "before": before,
        "after": after,
        "beforeTime": before_time,
        "afterTime": after_time,
        "kinds": kinds,
        "order": order,
    }
    response = requests.get(API_BASE + "bets", params=param_dict)
    response.raise_for_status()
    print(response.json()[0])
    return [Bet(**bet) for bet in response.json()]


def has_stake(market_position: MarketPosition) -> bool:
    """
    Check if the market position has a stake
    """
    return market_position.maxSharesOutcome is not None
