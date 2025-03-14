import dspy
from logging import Logger
import json
from pathlib import Path

from typing import Any, Dict, Optional
from dspy.utils.callback import BaseCallback

from src.tools.search import Search
from src.tools.python_interpreter import PythonInterpreter


def make_search_tools(search: Search, unified_search: bool) -> list:
    def get_relevant_urls(query: str) -> list[dict]:
        results = search.get_results(query)
        result_dicts = [result.to_dict() for result in results]
        return result_dicts

    def retrieve_web_content(url_list: list[str] | dict) -> list[dict]:
        if isinstance(url_list, dict) and "items" in url_list:
            url_list = list(url_list["items"])
        cleaned_html = [search.retrieve_cleaned_html(url) for url in url_list]
        result_dicts = [
            {"url": url, "cleaned_html_content": html}
            for url, html in zip(url_list, cleaned_html)
        ]
        return result_dicts

    if unified_search:

        def web_search(query: str) -> list[dict]:
            relevant_urls = get_relevant_urls(query)
            urls = [result["link"] for result in relevant_urls]
            web_content = retrieve_web_content(urls)
            result_dicts = [
                {**relevant_url, **content}
                for relevant_url, content in zip(relevant_urls, web_content)
            ]
            return result_dicts

        search_tools = [web_search]

    else:
        search_tools = [get_relevant_urls, retrieve_web_content]

    return search_tools


class MarketPrediction(dspy.Signature):
    """Given a question and description, predict the likelihood (between 0 and 1) that the market will resolve YES."""

    question: str = dspy.InputField()
    description: str = dspy.InputField()
    creatorUsername: str = dspy.InputField()
    comments: list[dict] = dspy.InputField()
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
        self.python_logger.debug(f"Starting DSPy module {instance} with inputs:")
        self.python_logger.debug(json.dumps(inputs, indent=4))

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
        self.python_logger.debug(f"Starting tool {instance} with inputs:")
        self.python_logger.debug(json.dumps(inputs, indent=4))

    def on_tool_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        self.python_logger.debug(f"Tool {call_id} finished with outputs:")
        self.python_logger.debug(outputs)
        if exception is not None:
            self.python_logger.error("DSPy Tool Exception:")
            self.python_logger.error(exception)

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        self.python_logger.debug(f"Starting LM {instance} with inputs:")
        self.python_logger.debug(json.dumps(inputs, indent=4))

    def on_lm_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        self.python_logger.debug(f"LM {call_id} finished with outputs:")
        self.python_logger.debug(json.dumps(outputs, indent=4))
        if exception is not None:
            self.python_logger.error("DSPy LM Exception:")
            self.python_logger.error(exception)


def init_dspy(
    llm_config_path: Path,
    dspy_program_path: Optional[Path],
    search: Search,
    unified_web_search: bool,
    use_python_interpreter: bool,
    logger: Optional[Logger] = None,
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

    tools = make_search_tools(search, unified_web_search)

    if use_python_interpreter:

        def eval_python(code: str) -> Dict[str, Any]:
            interpreter = PythonInterpreter()
            result = interpreter.execute(code)
            return result

        tools.append(eval_python)

    predict_market = dspy.ReAct(
        MarketPrediction,
        tools=tools,
    )
    if dspy_program_path is not None:
        predict_market.load(dspy_program_path)
        logger.info(f"Loaded DSPy program from {dspy_program_path}")
    if logger is not None:
        logger.info("DSPy initialized")
    return predict_market
