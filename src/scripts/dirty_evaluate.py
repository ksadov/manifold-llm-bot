from tqdm import tqdm
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import signal
import sys
from src.agent import init_pipeline
from src.backtesting.dataset import load_examples
from src.backtesting.metrics import (
    soft_cross_entropy,
    validate_directional,
    score_stats,
)
from src.backtesting.metrics import brier_score as brier_score_fn
from src.scripts.evaluate import jsonify_eval_outputs


class TimeoutException(Exception):
    pass


# Flag to track if Ctrl+C was pressed
ctrl_c_pressed = False


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) signal."""
    global ctrl_c_pressed
    if not ctrl_c_pressed:
        print(
            "\nCtrl+C detected. Finishing current tasks and preparing to exit gracefully..."
        )
        ctrl_c_pressed = True
    else:
        # If Ctrl+C is pressed a second time, exit immediately
        print("\nForced exit. Results may be incomplete.")
        sys.exit(1)


def run_with_timeout(func, args=None, kwargs=None, timeout=None):
    """
    Run a function with a timeout.

    Args:
        func: The function to run.
        args: Arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
        timeout: The timeout in seconds. If None, no timeout is applied.

    Returns:
        The result of the function.

    Raises:
        TimeoutException: If the function does not complete within the timeout.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    if timeout is None:
        return func(*args, **kwargs)

    result_container = []
    exception_container = []

    def worker():
        try:
            result_container.append(func(*args, **kwargs))
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=worker)
    thread.daemon = True

    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException("Function execution timed out")

    if exception_container:
        raise exception_container[0]

    return result_container[0]


def process_example(args):
    """
    Process a single example with timeout and error handling.
    Returns a tuple (example, prediction, cross_entropy_score, directional_score, l1_score, status, error_msg, elapsed)
    where status is one of "success", "timeout", or "error"
    """
    example, predict_market, use_brier, timeout = args
    error_msg = None
    status = "success"

    start_time = time.time()

    try:
        # Define a function that will process this example
        def process_func():
            prediction = predict_market(
                question=example.question,
                description=example.description,
                current_date=example.current_date,
                creatorUsername=example.creatorUsername,
                comments=example.comments,
                cutoff_date=example.cutoff_date,
            )
            if use_brier:
                score = brier_score_fn(example, prediction)
            else:
                score = soft_cross_entropy(example, prediction)
            directional_score = validate_directional(example, prediction)
            return prediction, score, directional_score

        # Run the function with a timeout
        prediction, score, directional_score = run_with_timeout(
            process_func, timeout=timeout
        )

    except TimeoutException:
        prediction = None
        score = None
        directional_score = None
        status = "timeout"
        error_msg = "Evaluation timed out"

    except Exception as e:
        prediction = None
        score = None
        directional_score = None
        status = "error"
        error_msg = str(e)

    elapsed = time.time() - start_time
    if status == "timeout" and timeout is not None:
        elapsed = timeout  # Cap at timeout value

    return (
        example,
        prediction,
        score,
        directional_score,
        status,
        error_msg,
        elapsed,
    )


def evaluate(
    config_path: Path,
    parquet_path: Path,
    max_examples: Optional[int],
    log_level: str,
    num_threads: int,
    trade_from_start: bool,
    use_brier: bool,
    min_num_trades: Optional[int],
    timeout: Optional[int],
):
    # Set up the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    predict_market, logger, evalfile_name, cutoff_date, exclude_groups = init_pipeline(
        config_path, log_level, "eval"
    )
    examples = load_examples(
        parquet_path,
        cutoff_date,
        exclude_groups,
        trade_from_start,
        yes_no_resolution=use_brier,
        max_examples=max_examples,
        min_num_trades=min_num_trades,
    )

    predictions = []
    scores = []
    directional_scores = []
    result_triples = []
    timeouts_count = 0
    errors_count = 0
    processing_times = []
    failed_examples = []
    completed_count = 0
    total_examples = len(examples)

    # Prepare arguments for parallel processing
    process_args = [
        (example, predict_market, use_brier, timeout) for example in examples
    ]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures_to_args = {
            executor.submit(process_example, arg): arg for arg in process_args
        }

        # Track progress using tqdm
        progress_bar = tqdm(total=total_examples, desc="Processing examples")

        try:
            # Process completed futures as they come in
            for future in as_completed(futures_to_args):
                if ctrl_c_pressed:
                    # Cancel any pending futures when Ctrl+C is pressed
                    for f in futures_to_args.keys():
                        if not f.done():
                            f.cancel()
                    logger.info("Ctrl+C detected. Processing partial results...")
                    break

                (
                    example,
                    prediction,
                    score,
                    directional_score,
                    status,
                    error_msg,
                    elapsed,
                ) = future.result()

                completed_count += 1
                processing_times.append(elapsed)
                progress_bar.update(1)

                if status == "timeout":
                    timeouts_count += 1
                    logger.warning(
                        f"Example timed out after {timeout} seconds: {example.question[:50]}..."
                    )
                    failed_examples.append(
                        {
                            "question": example.question,
                            "status": status,
                            "error": error_msg,
                            "elapsed": elapsed,
                        }
                    )
                    continue
                elif status == "error":
                    errors_count += 1
                    logger.error(
                        f"Example errored: {example.question[:50]}... Error: {error_msg}"
                    )
                    failed_examples.append(
                        {
                            "question": example.question,
                            "status": status,
                            "error": error_msg,
                            "elapsed": elapsed,
                        }
                    )
                    continue

                predictions.append(prediction)

                if score is not None:
                    scores.append(score)

                if directional_score is not None:
                    directional_scores.append(directional_score)

                if prediction is not None:
                    result_triples.append((example, prediction, score))

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            progress_bar.close()

    # Save results
    logger.info(f"Saving results from {len(result_triples)} completed examples")
    jsonify_eval_outputs(result_triples, evalfile_name)
    logger.info("Evaluation results saved to %s", evalfile_name)

    # Calculate statistics
    if scores:
        score_mean, score_confidence = score_stats(scores)
    directional_mean, directional_confidence = score_stats(directional_scores)

    # Save failed examples to a separate file
    if failed_examples:
        import json
        from datetime import datetime

        failed_file = (
            Path(evalfile_name).parent
            / f"failed_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(failed_file, "w") as f:
            json.dump(failed_examples, f, indent=2)
        logger.info(f"Failed examples saved to {failed_file}")

    # Log completion status
    if ctrl_c_pressed:
        logger.info(
            f"Evaluation stopped early due to Ctrl+C. Processed {completed_count}/{total_examples} examples."
        )
    else:
        logger.info(
            f"Processed all {total_examples} examples with {num_threads} threads"
        )

    # Log results
    logger.info(
        f"Timed out: {timeouts_count} examples ({timeouts_count/completed_count*100:.2f}%)"
    )
    logger.info(
        f"Errors: {errors_count} examples ({errors_count/completed_count*100:.2f}%)"
    )
    logger.info(
        f"Successful: {completed_count - timeouts_count - errors_count} examples ({(completed_count - timeouts_count - errors_count)/completed_count*100:.2f}%)"
    )

    if processing_times:
        logger.info(
            f"Average processing time: {sum(processing_times)/len(processing_times):.2f} seconds"
        )

    if scores:
        score_type = "Brier" if use_brier else "Cross-entropy"
        logger.info(f"{score_type}: mean {score_mean}, 95% CI +-{score_confidence}")
    logger.info(
        f"Directional: mean {directional_mean}, 95% CI +-{directional_confidence}"
    )

    return result_triples


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
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--random_snapshot", action="store_true")
    parser.add_argument("--score_type", type=str, default="brier")
    parser.add_argument("--min_num_trades", type=int, default=10)
    args = parser.parse_args()

    assert args.score_type in ["brier", "cross_entropy"], "Invalid score type"

    evaluate(
        args.config_path,
        args.parquet_path,
        args.max_examples,
        args.log_level,
        args.num_threads,
        not args.random_snapshot,
        args.score_type == "brier",
        args.min_num_trades,
        args.timeout,
    )


if __name__ == "__main__":
    main()
