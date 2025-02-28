from math import log
from pathlib import Path
import json
import datetime
import argparse
from typing import Optional
import time
import os

from src.backtesting.dataset import load_examples
from src.agent import init_dspy
from src.search import Search
from src.logging import create_logger
from tqdm import tqdm


def validate_directional(example, pred, trace=None) -> int:
    pred_answer = pred["answer"]
    resolution = example["resolution"]
    if resolution == "YES" and pred_answer > 0.5:
        return 1
    elif resolution == "NO" and pred_answer < 0.5:
        return 1
    elif resolution == "YES" and pred_answer < 0.5:
        return -1
    elif resolution == "NO" and pred_answer > 0.5:
        return -1
    else:
        return 0


def validate_probability(example, pred, trace=None) -> float:
    pred_answer = pred["answer"]
    actual_probability = example["probability"]
    return (pred_answer - actual_probability) ** 2


def init_search(config_path: Path, cutoff_date: datetime.datetime) -> Search:
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
        cutoff_date=cutoff_date,
    )
    return search


def backtest_evaluate(
    config_path: Path,
    dev_parquet_path: Path,
    max_examples: Optional[int],
    log_level: str,
):
    with open(config_path) as f:
        config = json.load(f)
    llm_config_path = Path(config["llm_config_path"])
    with open(llm_config_path) as f:
        llm_config = json.load(f)
    # specified in 2021-01-01 format
    if "knowledge_cutoff" in llm_config:
        cutoff_date = datetime.datetime.strptime(
            llm_config["knowledge_cutoff"], "%Y-%m-%d"
        )
    else:
        cutoff_date = None
    examples = load_examples(
        dev_parquet_path,
        cutoff_date=cutoff_date,
        exclude_groups=config["market_filters"]["exclude_groups"],
        max_examples=max_examples,
    )
    search = init_search(config_path, cutoff_date)
    logger, logfile_name = create_logger(config["name"], eval=True, log_level=log_level)
    evalfile_name = f"logs/eval/{logfile_name.split('.')[0]}.json"
    os.makedirs("logs/eval", exist_ok=True)
    logger.info(
        f"Config: {config_path}, dev_parquet_path: {dev_parquet_path}, max_examples: {max_examples}",
    )
    logger.info("Config: {config}")
    logger.info(f"Loaded {len(examples)} examples")
    predict_market = init_dspy(llm_config_path, search, logger)
    scores = {"directional": [], "probability": [], "time": []}
    for example in tqdm(examples):
        start_time = time.time()
        search.set_cutoff_date(example["datetime_timestamp"])
        pred = predict_market(
            question=example["question"],
            description=example["description"],
            current_date=example["current_date"],
        )
        scores["time"].append(time.time() - start_time)
        scores["directional"].append(validate_directional(example, pred, trace=None))
        scores["probability"].append(validate_probability(example, pred, trace=None))
    print("Directional score:", sum(scores["directional"]) / len(scores["directional"]))
    print("Probability score:", sum(scores["probability"]) / len(scores["probability"]))
    print(
        "Average time per prediction:",
        round(sum(scores["time"]) / len(scores["time"]), 3),
    )
    with open(evalfile_name, "w") as f:
        json.dump(scores, f)
        logger.info(f"Saved evaluation results to {evalfile_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "--dev_parquet_path",
        type=Path,
        required=True,
        help="Path to the dev parquet file",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--use_directional",
        action="store_true",
        help="Whether to use directional evaluation",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
    )
    args = parser.parse_args()
    backtest_evaluate(
        args.config_path,
        args.dev_parquet_path,
        args.max_examples,
        args.log_level,
    )


if __name__ == "__main__":
    main()
