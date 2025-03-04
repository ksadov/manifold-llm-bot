import concurrent.futures


class TimeoutError(Exception):
    """Raised when a function call times out"""

    pass


def run_with_timeout(func, timeout, *args, **kwargs):
    """Execute a function with a timeout."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
