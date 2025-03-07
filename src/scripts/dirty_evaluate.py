from tqdm import tqdm

from pathlib import Path
from typing import List, Optional
from src.backtesting.dataset import load_examples

from src.evaluation import (
    setup_pipeline,
    soft_cross_entropy,
    validate_directional,
)
from src.scripts.evaluate import jsonify_eval_outputs


def score_stats(scores):
    """
    Return mean and STD of scores
    """
    mean = sum(scores) / len(scores)
    std = (sum((score - mean) ** 2 for score in scores) / len(scores)) ** 0.5
    return mean, std


def evaluate(
    config_path: Path,
    parquet_path: Path,
    max_examples: Optional[int],
    log_level: str,
    num_threads: int,
    timeout: Optional[int],
):
    predict_market, logger, evalfile_name, cutoff_date, exclude_groups = setup_pipeline(
        config_path, log_level, "eval"
    )
    examples = load_examples(
        parquet_path,
        cutoff_date,
        exclude_groups,
        max_examples,
    )
    predictions = []
    cross_entropy_scores = []
    directional_scores = []
    for example in tqdm(examples):
        prediction = predict_market(
            question=example.question,
            description=example.description,
            current_date=example.current_date,
            creatorUsername=example.creatorUsername,
            comments=example.comments,
        )
        predictions.append(prediction)
        cross_entropy_scores.append(soft_cross_entropy(example, prediction))
        directional_scores.append(validate_directional(example, prediction))
    result_triples = list(zip(examples, predictions, cross_entropy_scores))
    jsonify_eval_outputs(result_triples, evalfile_name)
    logger.info("Evaluation results saved to %s", evalfile_name)
    cross_entropy_mean, cross_entropy_std = score_stats(cross_entropy_scores)
    directional_mean, directional_std = score_stats(directional_scores)
    logger.info(
        f"Soft cross entropy: mean {cross_entropy_mean}, std {cross_entropy_std}"
    )
    logger.info(f"Directional: mean {directional_mean}, std {directional_std}")
    return result_triples


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument(
        "--parquet_path", type=Path, default="processed_data/test.parquet"
    )
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()
    evaluate(
        args.config_path,
        args.parquet_path,
        args.max_examples,
        args.log_level,
        args.num_threads,
        args.timeout,
    )


if __name__ == "__main__":
    main()
