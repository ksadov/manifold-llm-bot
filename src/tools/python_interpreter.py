import sys
import threading
from typing import Dict, Any, Optional
import traceback
import io
import time
import ctypes


class TimeoutError(Exception):
    """Exception raised when code execution exceeds the time limit."""

    pass


class PythonInterpreter:
    """
    A class that provides a sandboxed environment for executing Python code
    with optional time limits and variable injection.
    """

    def __init__(self, time_limit: int = 5):
        """
        Initialize the interpreter with an optional time limit.

        Args:
            time_limit: Maximum execution time in seconds (default: 5)
        """
        self.time_limit = time_limit
        self.stdout = None
        self.stderr = None

    def _terminate_thread(self, thread):
        """
        Attempt to terminate a thread. This is not a clean solution but can work as a last resort.
        For CPython, this uses the ctypes module to raise an exception in the target thread.

        Note: This is implementation-specific and may not work on all Python implementations.
        """
        if not thread.is_alive():
            return

        exc_type = TimeoutError
        exc = exc_type(f"Code execution timed out after {self.time_limit} seconds")

        # Get the thread identifier and raise an exception in that thread
        thread_id = thread.ident
        if thread_id and hasattr(ctypes, "pythonapi"):
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), ctypes.py_object(exc)
            )
            if res == 0:
                raise ValueError("Invalid thread ID")
            elif res > 1:
                # If more than one thread was affected, something went wrong
                # Cancel the exception in all threads to avoid undefined behavior
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread_id), None
                )
                raise SystemError("PyThreadState_SetAsyncExc failed")

    def execute(
        self, code: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the provided Python code with optional variable injection.

        Args:
            code: The Python code to execute
            variables: Optional dictionary of variables to inject into the execution environment

        Returns:
            Dictionary containing execution results:
                - 'stdout': Captured standard output
                - 'stderr': Captured standard error
                - 'variables': Dictionary of variables after execution
                - 'error': Exception information if an error occurred

        Raises:
            TimeoutError: If execution time exceeds the set time limit
        """
        # Prepare shared objects for thread communication
        result = {"variables": {}, "stdout": "", "stderr": "", "error": None}

        # Prepare execution environment
        local_vars = {} if variables is None else variables.copy()

        # Thread execution function
        def execute_code():
            nonlocal result

            # Capture stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                # Execute the code
                exec(code, {}, local_vars)

                # Collect results
                result["variables"] = local_vars
                result["stdout"] = sys.stdout.getvalue()
                result["stderr"] = sys.stderr.getvalue()

            except Exception as e:
                # Capture any exceptions
                result["error"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                result["stderr"] = sys.stderr.getvalue()

            finally:
                # Restore stdout and stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        # Create and start the execution thread
        execution_thread = threading.Thread(target=execute_code)
        execution_thread.daemon = (
            True  # Allow program to exit even if thread is running
        )
        execution_thread.start()

        # Wait for execution to complete or timeout
        execution_thread.join(self.time_limit if self.time_limit > 0 else None)

        # If thread is still running after timeout, terminate it
        if execution_thread.is_alive():
            try:
                self._terminate_thread(execution_thread)
                raise TimeoutError(
                    f"Code execution timed out after {self.time_limit} seconds"
                )
            except (ValueError, SystemError) as e:
                # If we couldn't terminate the thread, report that
                result["error"] = {
                    "type": "ThreadTerminationError",
                    "message": f"Thread termination failed: {str(e)}",
                    "traceback": traceback.format_exc(),
                }

        return result


# Example usage
if __name__ == "__main__":
    interpreter = PythonInterpreter(time_limit=2)

    # Example 1: Simple code execution
    result = interpreter.execute("print('Hello, World!')")
    print("Example 1 - stdout:", result["stdout"])

    # Example 2: With variables
    variables = {"x": 10, "y": 20}
    result = interpreter.execute("z = x + y\nprint(f'Sum is {z}')", variables)
    print("Example 2 - stdout:", result["stdout"])
    print("Example 2 - variables:", result["variables"])

    # Example 3: Code that times out
    result = interpreter.execute(
        "import time\ntime.sleep(5)\nprint('This should not print')"
    )
    print("Example 3 - error:", result.get("error"))

    # Example 4: Code with errors
    result = interpreter.execute("print(undefined_variable)")
    print(
        "Example 4 - error type:", result["error"]["type"] if result["error"] else None
    )
    print(
        "Example 4 - error message:",
        result["error"]["message"] if result["error"] else None,
    )
