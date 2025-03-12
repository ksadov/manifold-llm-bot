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

from src.agent import init_dspy
from src.tools.search import init_search
from src.logging import create_logger


def score_stats(scores):
    """
    Return mean and 95% confidence interval for a list of scores.
    """
    if not scores:
        return 0, 0
    n = len(scores)
    mean = sum(scores) / n

    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in scores) / n
    std_dev = variance**0.5

    # Calculate standard error of the mean
    std_error = std_dev / (n**0.5)

    # Calculate 95% confidence interval (1.96 is the z-score for 95% CI)
    confidence = 1.96 * std_error

    return mean, confidence


def brier_score(example, pred, trace=None):
    """
    Compute the Brier score.

    Parameters:
    - y_true: ground truth probability (values in [0, 1]).
    - p_pred: predicted probability (values in [0, 1]).

    Returns:
    - Brier score, or None if the resolution is not YES or NO or the prediction is not
      in [0, 1].
    """
    p_pred = pred["answer"]
    resolution_value = (
        1
        if example["resolution"] == "YES"
        else 0 if example["resolution"] == "NO" else None
    )
    if resolution_value is None or p_pred > 1:
        return None
    return (p_pred - resolution_value) ** 2


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


def soft_cross_entropy(example, pred, trace=None):
    """
    Compute the cross entropy loss for soft targets.

    Parameters:
    - y_true: ground truth probability (values in [0, 1]).
    - p_pred: predicted probability (values in [0, 1]).
    - epsilon: Small value to avoid log(0).

    Returns:
    - flipped loss, because dspy optimizes for higher values.
    """
    epsilon = 1e-15
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
    log_level: str,
    split: str,
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

    logger, logfile_name = create_logger(config["name"], split, log_level=log_level)
    if split == "eval":
        evalfile_name = f"logs/eval/{logfile_name.split('.')[0]}.json"
        os.makedirs("logs/eval", exist_ok=True)
    else:
        evalfile_name = None
    logger.info(f"Config: {config_path}")
    logger.info(f"Config: {json.dumps(config, indent=4)}")
    search = init_search(config_path, cutoff_date)

    # Initialize prediction function
    predict_market = init_dspy(
        llm_config_path,
        config["dspy_program_path"],
        search,
        config["unified_web_search"],
        config["use_python_interpreter"],
        logger,
    )
    return (
        predict_market,
        logger,
        evalfile_name,
        cutoff_date,
        config["market_filters"]["exclude_groups"],
    )
