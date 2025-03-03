import argparse
import dspy
from typing import Optional
from pathlib import Path
from src.evaluation import setup_pipeline, soft_cross_entropy, validate_directional


def metric_for_optimizer(example, pred, trace=None):
    """
    via https://github.com/stanfordnlp/dspy/issues/1978
    If we're "bootstrapping" for optimization, return a boolean value
    """
    if trace is not None:
        directional_value = validate_directional(example, pred, trace)
        print(f"Directional value: {directional_value}")
        return directional_value > 0
    else:
        score = soft_cross_entropy(example, pred, trace)
        print(f"Soft cross entropy score: {score}")
        return score


def optimize(
    config_path: Path,
    train_parquet_path: Path,
    max_examples: Optional[int],
    num_threads: int,
    save_filename: Path,
    log_level: str,
    timeout: Optional[int] = None,
):
    trainset, predict_market, logger, _ = setup_pipeline(
        config_path, train_parquet_path, max_examples, log_level, "train", timeout
    )
    tp = dspy.MIPROv2(
        metric=metric_for_optimizer,
        auto="light",
        num_threads=num_threads,
    )
    optimized_predict_market = tp.compile(
        predict_market, trainset=trainset, max_bootstrapped_demos=0, max_labeled_demos=0
    )
    save_dir = Path("dspy_programs")
    save_dir.mkdir(exist_ok=True)
    optimized_predict_market.save(save_dir / save_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument(
        "--train_parquet_path", type=Path, default="processed_data/train.parquet"
    )
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--save_filename", type=Path, required=True)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()
    optimize(
        args.config_path,
        args.train_parquet_path,
        args.max_examples,
        args.num_threads,
        args.save_filename,
        args.log_level,
        args.timeout,
    )


if __name__ == "__main__":
    main()
