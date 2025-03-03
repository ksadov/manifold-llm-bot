from math import log
from pathlib import Path
import json
import datetime
from typing import Optional
import os
import dspy
from logging import Logger
from typing import List, Tuple
import math

from src.backtesting.dataset import load_examples
from src.agent import init_dspy
from src.search import init_search
from src.logging import create_logger


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


def soft_cross_entropy(example, pred, epsilon=1e-15):
    """
    Compute the cross entropy loss for soft targets.

    Parameters:
    - y_true: Array of ground truth probabilities (values in [0, 1]).
    - p_pred: Array of predicted probabilities (values in [0, 1]).
    - epsilon: Small value to avoid log(0).

    Returns:
    - Array of loss values for each instance.
    """
    p_pred = pred["answer"]
    y_true = example["probability"]
    # Clip predictions to avoid log(0)
    p_pred = min(p_pred, epsilon, 1 - epsilon)
    loss = -(y_true * math.log(p_pred) + (1 - y_true) * math.log(1 - p_pred))
    # for our optimizer, higher is better
    flipped_loss = -loss
    return flipped_loss


def setup_pipeline(
    config_path: Path,
    parquet_path: Path,
    max_examples: Optional[int],
    log_level: str,
    split: str,
    timeout: Optional[int] = None,
) -> Tuple[List[dspy.Example], dspy.ReAct, Logger, Optional[str]]:
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
        parquet_path,
        cutoff_date=cutoff_date,
        exclude_groups=config["market_filters"]["exclude_groups"],
        max_examples=max_examples,
    )
    logger, logfile_name = create_logger(config["name"], eval=True, log_level=log_level)
    if split == "dev":
        evalfile_name = f"logs/eval/{logfile_name.split('.')[0]}.json"
        os.makedirs("logs/eval", exist_ok=True)
    else:
        evalfile_name = None
    logger.info(
        f"Config: {config_path}, parquet_path: {parquet_path}, max_examples: {max_examples}, "
    )
    logger.info(f"Config: {config}")
    logger.info(f"Loaded {len(examples)} examples")
    search = init_search(config_path, cutoff_date)

    # Initialize prediction function
    predict_market = init_dspy(
        llm_config_path,
        search,
        config["unified_web_search"],
        config["use_python_interpreter"],
        logger,
        timeout=timeout,
    )
    return examples, predict_market, logger, evalfile_name


def jsonify_eval_outputs(result_triples: List[dict], evalfile_name: str):
    """
    result_triples is a list of (example, prediction, score) tuples
    """
    results = []
    for example, prediction, score in result_triples:
        results.append(
            {
                "example": str(example.toDict()),
                "prediction": str(prediction.toDict()),
                "score": score,
            }
        )
    with open(evalfile_name, "w") as f:
        json.dump(results, f)


def evaluate(
    config_path: Path,
    dev_parquet_path: Path,
    max_examples: Optional[int],
    log_level: str,
    split: str,
    num_threads: int,
    timeout: Optional[int] = None,
):
    examples, predict_market, logger, evalfile_name = setup_pipeline(
        config_path, dev_parquet_path, max_examples, log_level, split, timeout
    )
    evaluator = dspy.evaluate.Evaluate(
        devset=examples,
        num_threads=num_threads,
        display_progress=True,
        display_table=5,
        return_outputs=True,
        max_errors=len(examples),
    )
    overall_score, result_triples = evaluator(predict_market, metric=soft_cross_entropy)
    logger.info(f"Overall score: {overall_score}")
    # get average directional score
    directional_scores = [validate_directional(*triple) for triple in result_triples]
    avg_directional_score = sum(directional_scores) / len(directional_scores)
    logger.info(f"Average directional score: {avg_directional_score}")
    if evalfile_name:
        jsonify_eval_outputs(result_triples, evalfile_name)
    return overall_score


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--dev_parquet_path", type=Path, required=True)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()
    evaluate(
        args.config_path,
        args.dev_parquet_path,
        args.max_examples,
        args.log_level,
        args.split,
        args.num_threads,
        args.timeout,
    )


if __name__ == "__main__":
    main()
