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
    question,
    description,
    creatorUsername,
    resolution,
    tradeHistory,
    comments,
):
    tradeHistory = json.loads(tradeHistory)
    comments = json.loads(comments)
    random_snapshot = tradeHistory[math.floor(random.random() * len(tradeHistory))]
    timestamp, probability = (
        random_snapshot["snapshotTime"],
        random_snapshot["probability"],
    )
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
    )


def can_use(
    createdTime: int,
    groupSlugs: Iterable[str],
    cutoff_time: datetime.datetime,
    exclude_groups: Iterable[str],
) -> bool:
    cutoff_time = cutoff_time.timestamp() * 1000
    if createdTime < cutoff_time:
        return False
    for exclude_group in exclude_groups:
        if exclude_group in groupSlugs:
            return False
    return True


def load_examples(
    parquet_path: Path,
    cutoff_date: Optional[datetime.datetime],
    exclude_groups: Iterable[str],
    max_examples: Optional[int] = None,
):
    df = pd.read_parquet(parquet_path)
    examples = []
    for i, row in df.iterrows():
        if max_examples is not None and len(examples) >= max_examples:
            break
        if can_use(
            row["createdTime"],
            row["groupSlugs"],
            (cutoff_date if cutoff_date else datetime.datetime.now()),
            exclude_groups,
        ):
            examples.append(
                make_example(
                    row["question"],
                    row["description"],
                    row["creatorUsername"],
                    row["resolution"],
                    row["tradeHistory"],
                    row["comments"],
                )
            )
    return examples


def test(parquet_path):
    df = pd.read_parquet(parquet_path)
    representative_entry = df.iloc[50123]
    print("Parquet entry:")
    print(representative_entry)
    cutoff_time_string = "2022-02-01"
    cutoff_time = datetime.datetime.strptime(cutoff_time_string, "%Y-%m-%d").timestamp()
    exclude_groups = ["sex-and-love"]
    dataset = load_examples(parquet_path, cutoff_time, exclude_groups, max_examples=5)
    print("Dataset length:", len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])


def main():
    parquet_path = Path(
        "/Users/ksadov/Documents/manifold2025/bot/bot_dspy/manifold_dataset.parquet"
    )
    test(parquet_path)


if __name__ == "__main__":
    main()
