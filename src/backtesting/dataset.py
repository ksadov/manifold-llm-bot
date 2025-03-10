import dspy
import pandas as pd
import random
import math
import datetime
import json

from pathlib import Path
from typing import Optional, Iterable, Tuple

from src.agent import MarketPrediction


def make_example(
    question: str,
    description: str,
    creatorUsername: str,
    resolution: str,
    tradeHistory: list[dict],
    comments: list[dict],
    timestamp: Optional[int] = None,
):
    comments = json.loads(comments)
    if timestamp is None:
        random_snapshot = tradeHistory[math.floor(random.random() * len(tradeHistory))]
        timestamp, probability = (
            random_snapshot["snapshotTime"],
            random_snapshot["probability"],
        )
    else:
        probability = 0.5
        most_recent_time = tradeHistory[0]["snapshotTime"]
        for trade in tradeHistory:
            if (
                trade["snapshotTime"] <= timestamp
                and trade["snapshotTime"] > most_recent_time
            ):
                most_recent_time = trade["snapshotTime"]
                probability = trade["probability"]

    comments_pre_snapshot = [
        comment for comment in comments if comment["createdTime"] < timestamp
    ]
    # divide timestamp by 1000 to convert from milliseconds to seconds
    formatted_timestamp = datetime.datetime.fromtimestamp(timestamp / 1000).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return dspy.Example(
        question=question,
        description=description,
        current_date=formatted_timestamp,
        creatorUsername=creatorUsername,
        comments=comments_pre_snapshot,
        probability=probability,
        resolution=resolution,
        datetime_timestamp=datetime.datetime.fromtimestamp(timestamp / 1000),
    ).with_inputs(
        "question",
        "formatted_timestamp",
        "description",
        "creatorUsername",
        "comments",
        "current_date",
    )


def can_use(
    createdTime: int,
    groupSlugs: Iterable[str],
    cutoff_time: datetime.datetime,
    exclude_groups: Iterable[str],
    trade_history: list[dict],
    min_num_trades: Optional[int] = None,
) -> bool:
    cutoff_time = cutoff_time.timestamp() * 1000
    if createdTime < cutoff_time:
        return False
    for exclude_group in exclude_groups:
        if exclude_group in groupSlugs:
            return False
    if min_num_trades is not None and len(trade_history) < min_num_trades:
        return False
    return True


def load_examples(
    parquet_path: Path,
    cutoff_date: Optional[datetime.datetime],
    exclude_groups: Iterable[str],
    trade_from_start: bool,
    max_examples: Optional[int] = None,
    min_num_trades: Optional[int] = None,
):
    df = pd.read_parquet(parquet_path)
    examples = []
    for i, row in df.iterrows():
        if max_examples is not None and len(examples) >= max_examples:
            break
        trade_history = json.loads(row["tradeHistory"])
        if can_use(
            row["createdTime"],
            row["groupSlugs"],
            cutoff_date,
            exclude_groups,
            trade_history,
            min_num_trades,
        ):
            if trade_from_start:
                timestamp = row["createdTime"]
            else:
                timestamp = None
            examples.append(
                make_example(
                    row["question"],
                    row["description"],
                    row["creatorUsername"],
                    row["resolution"],
                    trade_history,
                    row["comments"],
                    timestamp,
                )
            )
    return examples


def test(parquet_path):
    df = pd.read_parquet(parquet_path)
    representative_entry = df.iloc[-10]
    print("Parquet entry:")
    print(representative_entry)
    cutoff_time_string = "2022-02-01"
    cutoff_time = datetime.datetime.strptime(cutoff_time_string, "%Y-%m-%d")
    exclude_groups = ["sex-and-love"]
    trade_from_start = True
    max_examples = 3
    min_num_trades = 10
    dataset = load_examples(
        parquet_path,
        cutoff_time,
        exclude_groups,
        trade_from_start,
        max_examples=max_examples,
        min_num_trades=min_num_trades,
    )
    print("Dataset length:", len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])


def main():
    parquet_path = Path("processed_data/test.parquet")
    test(parquet_path)


if __name__ == "__main__":
    main()
