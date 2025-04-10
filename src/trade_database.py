import sqlite3
from typing import List
from datetime import datetime

from src.manifold.types import MarketPosition
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

    def get_position(self, market_id: str) -> SavedPosition:
        """Get a market position"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM positions WHERE market_id = ?",
                (market_id,),
            )
            row = cursor.fetchone()
            if row:
                print("row", row)
                return SavedPosition(
                    market_id=row[0],
                    outcome=row[1],
                    shares=row[2],
                    entry_time=row[3],
                    last_updated=row[4],
                )
        return None

    def get_all_positions(self) -> List[SavedPosition]:
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
