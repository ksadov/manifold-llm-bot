import json

from matplotlib import use
import dspy

from pathlib import Path
from typing import List, Optional
from src.backtesting.dataset import load_examples

from src.backtesting.metrics import (
    soft_cross_entropy,
    validate_directional,
    brier_score,
    score_stats,
)
from src.agent import init_pipeline


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
    parquet_path: Path,
    max_examples: Optional[int],
    log_level: str,
    num_threads: int,
    use_brier: bool,
    trade_from_start: bool,
    min_num_trades: int,
):
    predict_market, logger, evalfile_name, cutoff_date, exclude_groups = init_pipeline(
        config_path, log_level, "eval"
    )
    examples = load_examples(
        parquet_path,
        cutoff_date,
        exclude_groups,
        trade_from_start,
        use_brier,
        max_examples,
        min_num_trades,
    )
    evaluator = dspy.evaluate.Evaluate(
        devset=examples,
        num_threads=num_threads,
        display_progress=True,
        display_table=5,
        return_outputs=True,
        max_errors=len(examples),
    )
    overall_score, result_triples = evaluator(
        predict_market, metric=brier_score if use_brier else soft_cross_entropy
    )
    logger.info(f"Overall score: {overall_score}")
    # filter out examples with no prediction
    result_triples = [
        triple
        for triple in result_triples
        if hasattr(triple[1], "answer") and triple[1].answer is not None
    ]
    logger.info(f"Failed to predict {len(examples) - len(result_triples)} examples")
    if len(result_triples) == 0:
        logger.error("No examples to evaluate")
        return
    score_mean, score_confidence = score_stats([triple[2] for triple in result_triples])
    logger.info(f"Score: mean {score_mean}, 95% CI +-{score_confidence}")
    directional_scores = [validate_directional(*triple) for triple in result_triples]
    directional_mean, directional_confidence = score_stats(directional_scores)
    logger.info(
        f"Directional: mean {directional_mean}, 95% CI +-{directional_confidence}"
    )
    if evalfile_name:
        jsonify_eval_outputs(result_triples, evalfile_name)
    return overall_score


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument(
        "--parquet_path", type=Path, default="processed_data/test.parquet"
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--random_snapshot", action="store_true")
    parser.add_argument("--min_num_trades", type=int, default=10)
    parser.add_argument("--score_type", type=str, default="brier")
    args = parser.parse_args()
    assert args.score_type in ["brier", "cross_entropy"], "Invalid score type"
    evaluate(
        args.config_path,
        args.parquet_path,
        args.max_examples,
        args.log_level,
        args.num_threads,
        args.score_type == "brier",
        not args.random_snapshot,
        args.min_num_trades,
    )


if __name__ == "__main__":
    main()
