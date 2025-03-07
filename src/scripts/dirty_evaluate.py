from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from src.backtesting.dataset import load_examples
from src.evaluation import (
    setup_pipeline,
    soft_cross_entropy,
    validate_directional,
)
from src.scripts.evaluate import jsonify_eval_outputs


class TimeoutException(Exception):
    pass


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


def score_stats(scores):
    """
    Return mean and STD of scores
    """
    if not scores:
        return 0, 0
    mean = sum(scores) / len(scores)
    std = (sum((score - mean) ** 2 for score in scores) / len(scores)) ** 0.5
    return mean, std


def process_example(args):
    """
    Process a single example with timeout and error handling.
    Returns a tuple (example, prediction, cross_entropy_score, directional_score, status, error_msg, elapsed)
    where status is one of "success", "timeout", or "error"
    """
    example, predict_market, timeout = args
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
            )
            cross_entropy_score = soft_cross_entropy(example, prediction)
            directional_score = validate_directional(example, prediction)
            return prediction, cross_entropy_score, directional_score

        # Run the function with a timeout
        prediction, cross_entropy_score, directional_score = run_with_timeout(
            process_func, timeout=timeout
        )

    except TimeoutException:
        prediction = None
        cross_entropy_score = None
        directional_score = None
        status = "timeout"
        error_msg = "Evaluation timed out"

    except Exception as e:
        prediction = None
        cross_entropy_score = None
        directional_score = None
        status = "error"
        error_msg = str(e)

    elapsed = time.time() - start_time
    if status == "timeout" and timeout is not None:
        elapsed = timeout  # Cap at timeout value

    return (
        example,
        prediction,
        cross_entropy_score,
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
    result_triples = []
    timeouts_count = 0
    errors_count = 0
    processing_times = []
    failed_examples = []

    # Prepare arguments for parallel processing
    process_args = [(example, predict_market, timeout) for example in examples]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_example, arg) for arg in process_args]

        # Use tqdm to show progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing examples"
        ):
            (
                example,
                prediction,
                cross_entropy_score,
                directional_score,
                status,
                error_msg,
                elapsed,
            ) = future.result()

            processing_times.append(elapsed)

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

            if cross_entropy_score is not None:
                cross_entropy_scores.append(cross_entropy_score)

            if directional_score is not None:
                directional_scores.append(directional_score)

            if prediction is not None:
                result_triples.append((example, prediction, cross_entropy_score))

    # Save results
    jsonify_eval_outputs(result_triples, evalfile_name)
    logger.info("Evaluation results saved to %s", evalfile_name)

    # Calculate statistics
    cross_entropy_mean, cross_entropy_std = score_stats(cross_entropy_scores)
    directional_mean, directional_std = score_stats(directional_scores)

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

    # Log results
    logger.info(f"Processed {len(examples)} examples with {num_threads} threads")
    logger.info(
        f"Timed out: {timeouts_count} examples ({timeouts_count/len(examples)*100:.2f}%)"
    )
    logger.info(
        f"Errors: {errors_count} examples ({errors_count/len(examples)*100:.2f}%)"
    )
    logger.info(
        f"Successful: {len(examples) - timeouts_count - errors_count} examples ({(len(examples) - timeouts_count - errors_count)/len(examples)*100:.2f}%)"
    )
    logger.info(
        f"Average processing time: {sum(processing_times)/len(processing_times):.2f} seconds"
    )
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
