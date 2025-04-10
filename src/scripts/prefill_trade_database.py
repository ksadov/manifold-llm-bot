import argparse
import sqlite3
import requests
import json
from pathlib import Path
from typing import List
from datetime import datetime
import time

from src.manifold.utils import get_my_account, get_market_positions, get_bets, has_stake
from src.trade_database import MarketPositionDB
from src.manifold.types import Bet


def get_whole_bet_history(user_id: str) -> List[Bet]:
    """Get all bets for a user"""
    # we get a max of 1000 bets at a time
    reached_end = False
    before_id = None
    bets = []
    while not reached_end:
        market_bets = get_bets(user_id=user_id, limit=1000, before=before_id)
        bets.extend(market_bets)
        if len(market_bets) < 1000:
            reached_end = True
        else:
            before_id = market_bets[-1].betId
    return bets


def populate_market_positions(user_id: str, db: MarketPositionDB):
    """Get all positions for a user"""
    bets = get_whole_bet_history(user_id)
    print(f"Got {len(bets)} bets")
    market_ids = set(bet.contractId for bet in bets)
    print(f"Got {len(market_ids)} market ids")
    for counter, market_id in enumerate(market_ids):
        market_position = get_market_positions(market_id, userId=user_id)
        for position in market_position:
            if has_stake(position):
                db.add_position(market_id, position)
        if counter != 0 and counter % 500 == 0:
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--secrets_path", "-s", type=Path, default="config/secrets/basic_secrets.json"
    )
    parser.add_argument("--db_path", "-d", type=Path, required=True)
    args = parser.parse_args()
    with open(args.secrets_path, "r") as f:
        secrets = json.load(f)
    api_key = secrets["manifold_api_key"]

    db = MarketPositionDB(args.db_path)
    user_id = get_my_account(api_key).id

    # Initial population of database
    populate_market_positions(user_id, db)

    # Example: Print all positions
    positions = db.get_all_positions()
    for pos in positions:
        print(f"Market {pos.market_id}: {pos.shares} shares of {pos.outcome}")


if __name__ == "__main__":
    main()
