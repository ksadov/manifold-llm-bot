from math import log
from pathlib import Path
import json
import datetime
import argparse
from typing import Optional, Dict, Any
import time
import os
import concurrent.futures
from tqdm.auto import tqdm
import threading

from src.backtesting.dataset import load_examples
from src.agent import init_dspy
from src.search import Search
from src.logging import create_logger


class TimeoutError(Exception):
    """Raised when a function takes too long to execute."""

    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Function timed out")


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=None):
    """
    Run a function with a timeout.

    Args:
        func: The function to run
        args: The positional arguments to pass to the function
        kwargs: The keyword arguments to pass to the function
        timeout_seconds: The timeout in seconds

    Returns:
        The result of the function if it completes within the timeout

    Raises:
        TimeoutError: If the function takes longer than timeout_seconds to complete
    """
    if timeout_seconds is None:
        return func(*args, **kwargs)

    # Use a different approach based on the platform
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True

    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

    if exception[0]:
        raise exception[0]

    return result[0]


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


def validate_probability(example, pred, trace=None) -> float:
    pred_answer = pred["answer"]
    actual_probability = example["probability"]
    return (pred_answer - actual_probability) ** 2


def init_search(config_path: Path, cutoff_date: datetime.datetime) -> Search:
    # Load config from file
    with open(config_path) as f:
        config = json.load(f)
    secrets_json_path = Path(config["secrets_path"])
    # Load secrets from file
    with open(secrets_json_path) as f:
        secrets = json.load(f)
    # Initialize search
    search = Search(
        secrets["google_api_key"],
        secrets["google_cse_cx"],
        config["max_search_results"],
        config["max_html_length"],
        cutoff_date=cutoff_date,
    )
    return search


def process_example(
    example: Dict[str, Any],
    predict_market,
    search: Search,
    timeout_seconds: Optional[int] = None,
    logger=None,
) -> Dict[str, Any]:
    """Process a single example and return scores."""
    start_time = time.time()

    try:
        search.set_cutoff_date(example["datetime_timestamp"])

        # Run predict_market with a timeout
        pred = run_with_timeout(
            predict_market,
            kwargs={
                "question": example["question"],
                "description": example["description"],
                "current_date": example["current_date"],
            },
            timeout_seconds=timeout_seconds,
        )
        logger.info("Result for question: " + example["question"])
        logger.info(pred)

        elapsed_time = time.time() - start_time

        return {
            "id": example.get("id", "unknown"),
            "time": elapsed_time,
            "directional": validate_directional(example, pred, trace=None),
            "probability": validate_probability(example, pred, trace=None),
            "skipped": False,
        }

    except TimeoutError as e:
        elapsed_time = time.time() - start_time
        if logger:
            logger.warning(
                f"Example {example.get('id', 'unknown')} timed out after {elapsed_time:.2f} seconds: {str(e)}"
            )
        return {
            "id": example.get("id", "unknown"),
            "time": elapsed_time,
            "directional": 0,
            "probability": 0,
            "skipped": True,
            "error": str(e),
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        if logger:
            logger.warning(
                f"Error processing example {example.get('id', 'unknown')}: {str(e)}"
            )
        return {
            "id": example.get("id", "unknown"),
            "time": elapsed_time,
            "directional": 0,
            "probability": 0,
            "skipped": True,
            "error": str(e),
        }


def backtest_evaluate(
    config_path: Path,
    dev_parquet_path: Path,
    max_examples: Optional[int],
    log_level: str,
    num_workers: int,
    timeout_seconds: Optional[int] = None,
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
    examples = load_examples(
        dev_parquet_path,
        cutoff_date=cutoff_date,
        exclude_groups=config["market_filters"]["exclude_groups"],
        max_examples=max_examples,
    )
    logger, logfile_name = create_logger(config["name"], eval=True, log_level=log_level)
    evalfile_name = f"logs/eval/{logfile_name.split('.')[0]}.json"
    os.makedirs("logs/eval", exist_ok=True)
    logger.info(
        f"Config: {config_path}, dev_parquet_path: {dev_parquet_path}, max_examples: {max_examples}, "
        f"num_workers: {num_workers}, timeout_seconds: {timeout_seconds}",
    )
    logger.info(f"Config: {config}")
    logger.info(f"Loaded {len(examples)} examples")

    # Initialize a search instance for each worker
    logger.info(f"Initializing {num_workers} search instances")
    search_instances = [
        init_search(config_path, cutoff_date) for _ in range(num_workers)
    ]

    # Initialize prediction function
    predict_market = init_dspy(
        llm_config_path,
        search_instances[0],
        config["unified_web_search"],
        config["use_python_interpreter"],
        logger,
    )

    # Prepare worker function with shared resources
    all_results = []

    # Define a worker initialization function that assigns a search instance to each worker
    def init_worker(worker_id):
        # This makes the search instance accessible to the worker
        process_example.search = search_instances[worker_id % num_workers]
        process_example.predict_market = predict_market
        process_example.logger = logger

    # Define a simplified worker function that uses thread-local resources
    def worker_fn(example, worker_id):
        return process_example(
            example,
            process_example.predict_market,
            process_example.search,
            timeout_seconds=timeout_seconds,
            logger=process_example.logger,
        )

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers, initializer=init_worker, initargs=(0,)
    ) as executor:
        # Create a dictionary to track futures and their corresponding worker IDs
        futures = {}

        # Submit initial batch of tasks
        for i, example in enumerate(examples[:num_workers]):
            future = executor.submit(worker_fn, example, i % num_workers)
            futures[future] = i % num_workers

        # Set up progress bar for the total number of examples
        with tqdm(total=len(examples), desc="Processing examples") as pbar:
            completed = 0
            next_example_idx = num_workers
            skipped_count = 0

            # Process results as they complete and submit new tasks
            while futures:
                # Wait for the next future to complete
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in done:
                    worker_id = futures.pop(future)

                    try:
                        result = future.result()
                        all_results.append(result)

                        if result.get("skipped", False):
                            skipped_count += 1

                    except Exception as e:
                        logger.error(f"Unexpected error in worker: {e}")
                        # Add placeholder for failed example
                        all_results.append(
                            {
                                "time": 0,
                                "directional": 0,
                                "probability": 0,
                                "skipped": True,
                                "error": str(e),
                            }
                        )
                        skipped_count += 1

                    # Submit next task if there are more examples
                    if next_example_idx < len(examples):
                        future = executor.submit(
                            worker_fn, examples[next_example_idx], worker_id
                        )
                        futures[future] = worker_id
                        next_example_idx += 1

                    # Update progress bar
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix({"skipped": skipped_count})

    # Filter out skipped examples for metrics calculation
    valid_results = [
        result for result in all_results if not result.get("skipped", False)
    ]

    # Aggregate results
    if valid_results:
        scores = {
            "directional": [result["directional"] for result in valid_results],
            "probability": [result["probability"] for result in valid_results],
            "time": [result["time"] for result in valid_results],
        }

        logger.info(f"Processed {len(all_results)} examples, {skipped_count} skipped")
        logger.info(
            f"Directional score: {sum(scores['directional']) / len(scores['directional'])}"
        )
        logger.info(
            f"Probability score error: {sum(scores['probability']) / len(scores['probability'])}"
        )
        logger.info(
            f"Average time per prediction: {round(sum(scores['time']) / len(scores['time']), 3)}"
        )
    else:
        logger.info("All examples were skipped, no valid results to report")
        scores = {"directional": [], "probability": [], "time": []}

    # Include skipped examples in the saved results
    full_results = {
        "scores": scores,
        "total_examples": len(all_results),
        "skipped_examples": skipped_count,
        "skipped_percentage": (
            round(skipped_count / len(all_results) * 100, 2) if all_results else 0
        ),
        "all_results": all_results,
    }

    with open(evalfile_name, "w") as f:
        json.dump(full_results, f)
        logger.info(f"Saved evaluation results to {evalfile_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "--dev_parquet_path",
        type=Path,
        required=True,
        help="Path to the dev parquet file",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--use_directional",
        action="store_true",
        help="Whether to use directional evaluation",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers to use",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each prediction (None means no timeout)",
    )
    args = parser.parse_args()
    backtest_evaluate(
        args.config_path,
        args.dev_parquet_path,
        args.max_examples,
        args.log_level,
        args.num_workers,
        args.timeout,
    )


if __name__ == "__main__":
    main()
