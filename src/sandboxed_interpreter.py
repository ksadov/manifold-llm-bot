import multiprocessing
import resource
import traceback
from typing import Any, Dict, Optional
import builtins


class SandboxedPythonInterpreter:
    def __init__(self, memory_limit_mb: int = 50, time_limit_seconds: int = 2):
        # Memory limit in bytes (50 MB default)
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.time_limit = time_limit_seconds

    def _run_code(
        self,
        code: str,
        variables: Optional[Dict[str, Any]],
        result_queue: multiprocessing.Queue,
    ):
        try:
            # Apply memory limit (Unix-only)
            resource.setrlimit(
                resource.RLIMIT_AS, (self.memory_limit, self.memory_limit)
            )
        except Exception as e:
            # In production, you might want to handle/log this differently
            pass

        # Create a restricted execution environment
        exec_globals = {}
        if variables:
            exec_globals.update(variables)

        # Define a whitelist of safe built-in functions
        safe_builtins = {
            "abs": builtins.abs,
            "all": builtins.all,
            "any": builtins.any,
            "bin": builtins.bin,
            "bool": builtins.bool,
            "chr": builtins.chr,
            "complex": builtins.complex,
            "dict": builtins.dict,
            "dir": builtins.dir,
            "divmod": builtins.divmod,
            "enumerate": builtins.enumerate,
            "filter": builtins.filter,
            "float": builtins.float,
            "format": builtins.format,
            "frozenset": builtins.frozenset,
            "hash": builtins.hash,
            "hex": builtins.hex,
            "int": builtins.int,
            "isinstance": builtins.isinstance,
            "issubclass": builtins.issubclass,
            "iter": builtins.iter,
            "len": builtins.len,
            "list": builtins.list,
            "map": builtins.map,
            "max": builtins.max,
            "min": builtins.min,
            "next": builtins.next,
            "oct": builtins.oct,
            "ord": builtins.ord,
            "pow": builtins.pow,
            "print": builtins.print,
            "range": builtins.range,
            "repr": builtins.repr,
            "reversed": builtins.reversed,
            "round": builtins.round,
            "set": builtins.set,
            "slice": builtins.slice,
            "sorted": builtins.sorted,
            "str": builtins.str,
            "sum": builtins.sum,
            "tuple": builtins.tuple,
            "type": builtins.type,
            "zip": builtins.zip,
        }
        # Prevent importing modules by omitting __import__
        exec_globals["__builtins__"] = safe_builtins

        try:
            exec(code, exec_globals)
            # Return the updated globals (or a subset as needed)
            result_queue.put((True, exec_globals))
        except Exception:
            # Send the traceback for error reporting
            tb = traceback.format_exc()
            result_queue.put((False, tb))

    def execute(
        self, code: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the provided code in a sandboxed environment.

        :param code: The code to execute.
        :param variables: Optional initial variables to include in the execution environment.
        :return: A dictionary of globals resulting from execution.
        :raises TimeoutError: If execution exceeds the time limit.
        :raises RuntimeError: If an error occurs during code execution.
        """
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_code, args=(code, variables, result_queue)
        )
        process.start()
        process.join(self.time_limit)

        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError("Execution exceeded the time limit.")

        # Retrieve results from the queue
        if not result_queue.empty():
            success, result = result_queue.get()
        else:
            raise RuntimeError("No result returned from the sandboxed process.")

        if not success:
            raise RuntimeError("Error during execution:\n" + result)

        return result


# Example usage:
if __name__ == "__main__":
    interpreter = SandboxedPythonInterpreter(memory_limit_mb=50, time_limit_seconds=2)
    code = """
test_array = [1, 2, 3, 4, 5]
test_sum = sum(test_array)
test_max = max(test_array)
"""
    try:
        output = interpreter.execute(code)
        print("Sandboxed globals:", output)
    except Exception as e:
        print("Error:", e)
