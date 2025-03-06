import json
import dspy

from pathlib import Path
from typing import List, Optional
from src.backtesting.dataset import load_examples

from src.evaluation import (
    setup_pipeline,
    soft_cross_entropy,
    validate_directional,
)


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
    # filter out examples with no prediction
    result_triples = [
        triple
        for triple in result_triples
        if hasattr(triple[1], "answer") and triple[1].answer is not None
    ]
    logger.info(f"Failed to predict {len(examples) - len(result_triples)} examples")
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
    parser.add_argument(
        "--parquet_path", type=Path, default="processed_data/test.parquet"
    )
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()
    evaluate(
        args.config_path,
        args.parquet_path,
        args.max_examples,
        args.log_level,
        args.num_threads,
    )


if __name__ == "__main__":
    main()
