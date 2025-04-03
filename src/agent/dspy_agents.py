import dspy
from logging import Logger
import json
from pathlib import Path
from typing import Any, Dict, Optional

from typing import Any, Dict, Optional
from dspy.utils.callback import BaseCallback

from src.tools.search import Search, make_search_tools
from src.tools.python_interpreter import (
    PythonInterpreter,
    eval_python as eval_python_tool,
)


class MarketPrediction(dspy.Signature):
    """Given a question and description, predict the likelihood (between 0 and 1) that the market will resolve YES."""

    question: str = dspy.InputField()
    description: str = dspy.InputField()
    creatorUsername: str = dspy.InputField()
    comments: list[dict] = dspy.InputField()
    current_date: str = dspy.InputField()
    answer: float = dspy.OutputField()


def stringify_for_logging(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=4)
    except Exception:
        return str(obj)


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
        self.python_logger.debug(stringify_for_logging(inputs))

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
        self.python_logger.debug(stringify_for_logging(inputs))

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
        self.python_logger.debug(stringify_for_logging(inputs))

    def on_lm_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        self.python_logger.debug(f"LM {call_id} finished with outputs:")
        self.python_logger.debug(stringify_for_logging(outputs))
        if exception is not None:
            self.python_logger.error("DSPy LM Exception:")
            self.python_logger.error(exception)


class GetSources(dspy.Signature):
    """Search the web and retrieve a list of HTML content potentially relevant to making a prediction on the given prediction market question or associated base rates. The output answer should be a list of strings, where each string is the cleaned HTML content from some URL."""

    question: str = dspy.InputField()
    description: str = dspy.InputField()
    creatorUsername: str = dspy.InputField()
    comments: list[dict] = dspy.InputField()
    current_date: str = dspy.InputField()
    answer: list[str] = dspy.OutputField()


class FillInScratchPad(dspy.Signature):
    """Fill in the double-bracketed sections of the template according to the instructions, using relevant information from the sources. Then return the filled-in reasoning template as well as your final answer."""

    template: str = dspy.InputField()
    sources: list[str] = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: float = dspy.OutputField()


class PredictWithScratchpad(dspy.Module):
    def __init__(self, search_tools: list, template: str):
        super().__init__()
        self.template = template
        self.get_sources = dspy.ReAct(GetSources, tools=search_tools)
        self.fill_in_scratch_pad = dspy.Predict(FillInScratchPad)

    def forward(
        self,
        question: str,
        description: str,
        creatorUsername: str,
        comments: list[dict],
        current_date: str,
    ) -> dict:
        sources = self.get_sources(
            question=question,
            description=description,
            creatorUsername=creatorUsername,
            comments=comments,
            current_date=current_date,
        )
        filled_in = self.fill_in_scratch_pad(template=self.template, sources=sources)
        return filled_in


class PredictWithSearchCutoff(dspy.Module):
    def __init__(
        self,
        search: Search,
        unified_search: bool,
        use_python_interpreter: bool,
        scratchpad_template: Optional[str],
    ):
        super().__init__()
        self.search = search
        self.unified_search = unified_search
        self.use_python_interpreter = use_python_interpreter
        self.tools = make_search_tools(search, unified_search)
        if use_python_interpreter and scratchpad_template is not None:
            raise ValueError(
                "Cannot use both Python interpreter and scratchpad prompts"
            )
        elif use_python_interpreter:
            self.tools.append(eval_python_tool)
        if scratchpad_template is not None:
            self.predict_market = PredictWithScratchpad(
                search_tools=self.tools,
                template=scratchpad_template,
            )
        else:
            self.predict_market = dspy.ReAct(
                MarketPrediction,
                tools=self.tools,
            )

    def forward(
        self,
        question: str,
        description: str,
        creatorUsername: str,
        comments: list[dict],
        current_date: str,
        cutoff_date: Optional[str] = None,
    ) -> dict:
        if cutoff_date is not None:
            self.search.set_cutoff_date(cutoff_date)
        return self.predict_market.forward(
            question=question,
            description=description,
            creatorUsername=creatorUsername,
            comments=comments,
            current_date=current_date,
        )


def init_dspy(
    llm_config: dict,
    dspy_program_path: Optional[Path],
    search: Search,
    unified_web_search: bool,
    use_python_interpreter: bool,
    scratchpad_template_path: Optional[Path],
    logger: Optional[Logger] = None,
) -> dspy.ReAct:
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

    predict_market = PredictWithSearchCutoff(
        search,
        unified_web_search,
        use_python_interpreter,
        scratchpad_template_path.read_text() if scratchpad_template_path else None,
    )
    if dspy_program_path is not None:
        predict_market.load(dspy_program_path)
        logger.info(f"Loaded DSPy program from {dspy_program_path}")
    if logger is not None:
        logger.info("DSPy initialized")
    return predict_market
