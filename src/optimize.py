import argparse
from math import e
import dspy
from typing import Optional
from pathlib import Path
import json
import datetime
from src.backtesting.dataset import load_examples
from src.agent import init_dspy
from src.backtesting.evaluation import validate_probability, init_search

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


def optimize(
    config_path: Path,
    train_parquet_path: Path,
    max_examples: Optional[int],
    num_threads: int,
    save_path: Path,
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
    search = init_search(config_path, cutoff_date)
    predict_market = init_dspy(
        llm_config_path,
        search,
        config["unified_web_search"],
        config["use_python_interpreter"],
    )
    trainset = load_examples(
        train_parquet_path,
        cutoff_date=cutoff_date,
        exclude_groups=config["market_filters"]["exclude_groups"],
        max_examples=max_examples,
    )
    kwargs = dict(num_threads=num_threads, display_progress=True, display_table=1)
    tp = dspy.COPRO(metric=validate_probability, auto="light")
    optimized_predict_market = tp.compile(
        predict_market, trainset=trainset, eval_kwargs=kwargs
    )
    optimized_predict_market.save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--train_parquet_path", type=Path, required=True)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--save_path", type=Path, required=True)
    args = parser.parse_args()
    optimize(
        args.config_path,
        args.train_parquet_path,
        args.max_examples,
        args.num_threads,
        args.save_path,
    )


if __name__ == "__main__":
    main()
