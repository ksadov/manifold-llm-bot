import argparse
import sqlite3
import requests
import json
from pathlib import Path
from typing import List
from datetime import datetime
import time

from src.manifold.utils import get_my_account, get_market_positions, get_bets, has_stake
from src.manifold.types import MarketPosition, Bet
from src.manifold.constants import API_BASE
from pydantic import BaseModel


class SavedPosition(BaseModel):
    market_id: str
    outcome: str
    shares: float
    entry_time: datetime
    last_updated: datetime


class MarketPositionDB:
    def __init__(self, db_path: str = "market_positions.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database with necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    market_id TEXT PRIMARY KEY,
                    outcome TEXT NOT NULL,
                    shares REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL
                )
            """
            )
            conn.commit()

    def add_position(self, market_id: str, market_position: MarketPosition):
        """Add or update a market position"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO positions
                (market_id, outcome, shares, entry_time, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    market_id,
                    market_position.maxSharesOutcome,
                    market_position.totalShares[market_position.maxSharesOutcome],
                    market_position.lastBetTime,
                    datetime.now(),
                ),
            )
            conn.commit()

    def remove_position(self, market_id: str):
        """Remove a market position"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions WHERE market_id = ?", (market_id,))
            conn.commit()

    def get_all_positions(self) -> List[SavedPosition]:
        """Get all tracked market positions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions")
            rows = cursor.fetchall()
            return [
                SavedPosition(
                    market_id=row[0],
                    outcome=row[1],
                    shares=row[2],
                    entry_time=row[3],
                    last_updated=row[4],
                )
                for row in rows
            ]


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
