import dspy
from logging import Logger
import json

from typing import Any, Dict, Optional
from dspy.utils.callback import BaseCallback

from src.search import Search


class MarketPrediction(dspy.Signature):
    """Given a question and description, predict the likelihood that the market will resolve YES."""

    question: str = dspy.InputField()
    description: str = dspy.InputField()
    current_date: str = dspy.InputField()
    answer: float = dspy.OutputField()


class AgentLoggingCallback(BaseCallback):
    def __init__(self, python_logger: Logger):
        super().__init__()
        self.python_logger = python_logger

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        self.python_logger.info(f"Starting DSPy module {instance} with inputs:")
        self.python_logger.info(inputs)

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        if exception is not None:
            self.python_logger.error("DSPy Formatter Exception:")
            self.python_logger.error(exception)

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        if exception is not None:
            self.python_logger.error("DSPy Parser Exception:")
            self.python_logger.error(exception)

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        self.python_logger.info(f"Starting tool {instance} with inputs:")
        self.python_logger.info(inputs)

    def on_tool_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        self.python_logger.info(f"Tool {call_id} finished with outputs:")
        self.python_logger.info(outputs)
        if exception is not None:
            self.python_logger.error("DSPy Tool Exception:")
            self.python_logger.error(exception)

    def on_lm_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        self.python_logger.info(f"LM {call_id} finished with outputs:")
        self.python_logger.info(outputs)
        if exception is not None:
            self.python_logger.error("DSPy LM Exception:")
            self.python_logger.error(exception)


def init_dspy(
    llm_config_path: str, search: Search, logger: Optional[Logger] = None
) -> dspy.ReAct:
    with open(llm_config_path) as f:
        llm_config = json.load(f)
    # DSPY expects OpenAI-compatible endpoints to have the prefix openai/
    # even if we're not using an OpenAI model
    lm = dspy.LM(
        f'openai/{llm_config["model"]}',
        api_key=llm_config["api_key"],
        api_base=llm_config["api_base"],
        **llm_config["prompt_params"],
    )
    if logger is not None:
        dspy.configure(lm=lm, callbacks=[AgentLoggingCallback(logger)])
    else:
        dspy.configure(lm=lm)

    def evaluate_math(expression: str) -> float:
        return dspy.PythonInterpreter({}).execute(expression)

    def web_search(query: str) -> list[dict]:
        results = search.get_results(query, retrieve_html=True)
        result_dicts = [result.to_dict() for result in results]
        return result_dicts

    predict_market = dspy.ReAct(MarketPrediction, tools=[web_search, evaluate_math])
    if logger is not None:
        logger.info("DSPy initialized")
        logger.info(llm_config)
    return predict_market
