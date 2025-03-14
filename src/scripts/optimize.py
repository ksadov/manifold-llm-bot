import argparse

from src.scripts import trade
import dspy
from typing import Optional
from pathlib import Path
from src.backtesting.metrics import (
    soft_cross_entropy,
    validate_directional,
    brier_score,
)
from src.backtesting.dataset import load_examples
from src.agent import init_pipeline


def optimizer_cross_entropy(example, pred, trace=None):
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


def optimizer_brier(example, pred, trace=None):
    if trace is not None:
        directional_value = validate_directional(example, pred, trace)
        print(f"Directional value: {directional_value}")
        return directional_value > 0
    else:
        score = brier_score(example, pred, trace)
        print(f"Brier score: {score}")
        return score


def optimize(
    config_path: Path,
    train_parquet_path: Path,
    val_parquet_path: Path,
    max_train_examples: Optional[int],
    max_val_examples: Optional[int],
    num_threads: int,
    save_filename: Path,
    log_level: str,
    optimizer: str,
    trade_from_start: bool,
    use_brier: bool,
):
    predict_market, _, _, cutoff_date, exclude_groups = init_pipeline(
        config_path,
        log_level,
        "optimize",
    )
    trainset = load_examples(
        train_parquet_path,
        cutoff_date,
        exclude_groups,
        trade_from_start,
        use_brier,
        max_train_examples,
    )
    valset = load_examples(
        val_parquet_path,
        cutoff_date,
        exclude_groups,
        trade_from_start,
        use_brier,
        max_val_examples,
    )
    if optimizer == "MIPROv2":
        tp = dspy.MIPROv2(
            metric=optimizer_brier if use_brier else optimizer_cross_entropy,
            auto="light",
            num_threads=num_threads,
        )
        optimized_predict_market = tp.compile(
            predict_market,
            trainset=trainset,
            valset=valset,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
        )
    elif optimizer == "COPRO":
        tp = dspy.COPRO(
            metric=optimizer_brier if use_brier else optimizer_cross_entropy,
            verbose=True,
        )
        kwargs = dict(num_threads=num_threads, display_progress=True, display_table=0)
        optimized_predict_market = tp.compile(
            predict_market,
            trainset=trainset,
            eval_kwargs=kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}, expected MIPROv2 or COPRO")
    save_dir = Path("dspy_programs")
    save_dir.mkdir(exist_ok=True)
    optimized_predict_market.save(save_dir / save_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument(
        "--train_parquet_path", type=Path, default="processed_data/train.parquet"
    )
    parser.add_argument(
        "--val_parquet_path", type=Path, default="processed_data/val.parquet"
    )
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--save_filename", type=Path, required=True)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--optimizer", type=str, default="MIPROv2")
    parser.add_argument("--random_snapshot", action="store_true")
    parser.add_argument("--score_type", type=str, default="brier")
    args = parser.parse_args()
    assert args.optimizer in [
        "MIPROv2",
        "COPRO",
    ], f"Unknown optimizer: {args.optimizer}"
    assert args.score_type in [
        "brier",
        "soft_cross_entropy",
    ], f"Unknown score type: {args.score_type}"
    optimize(
        args.config_path,
        args.train_parquet_path,
        args.val_parquet_path,
        args.max_train_examples,
        args.max_val_examples,
        args.num_threads,
        args.save_filename,
        args.log_level,
        args.optimizer,
        not args.random_snapshot,
        args.score_type == "brier",
    )


if __name__ == "__main__":
    main()
