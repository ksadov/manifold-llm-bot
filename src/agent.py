import dspy
import datetime
import json
import mlflow

from typing import Any, Dict, Optional
from dspy.utils.callback import BaseCallback

from src.manifold.types import FullMarket
from src.search import Search


class MarketPrediction(dspy.Signature):
    """Given a question and description, predict the likelihood that the market will resolve YES."""

    question: str = dspy.InputField()
    description: str = dspy.InputField()
    predicted_probability: str = dspy.OutputField()


class AgentLoggingCallback(BaseCallback):
    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        print(f"Starting call {call_id} with inputs:")
        print(inputs)
        print(f"Module: {instance}")
        print("=" * 80)

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        print(f"Starting adapter {instance} with inputs:")
        print(inputs)

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        print("Adapter Exception:")
        print(exception)

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        print(f"Starting parser {instance} with inputs:")
        print(inputs)

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        print("Parser Exception:")
        print(exception)

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        print(f"Starting tool {instance} with inputs:")
        print(inputs)

    def on_tool_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        print("Tool Exception:")
        print(exception)

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: Dict[str, Any],
    ):
        pass

    def on_lm_end(
        self,
        call_id: str,
        outputs: Optional[Dict[str, Any]],
        exception: Optional[Exception] = None,
    ):
        print("LM Exception:")
        print(exception)

    def on_module_end(self, call_id, outputs, exception):
        print("Module end Outputs:")
        print(outputs)
        print("Module end Exception:")
        print(exception)
        print("=" * 80)


def init_dspy(llm_config_path: str, search: Search):
    mlflow.set_experiment(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    mlflow.dspy.autolog()
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
    dspy.configure(lm=lm, callbacks=[AgentLoggingCallback()])

    def evaluate_math(expression: str) -> float:
        return dspy.PythonInterpreter({}).execute(expression)

    def web_search(query: str) -> list[dict]:
        results = search.get_results(query, retrieve_html=True)
        result_dicts = [result.to_dict() for result in results]
        return result_dicts

    predict_market = dspy.ReAct(MarketPrediction, tools=[web_search, evaluate_math])
    print("Initialized DSPY with config:")
    print(llm_config)
    return predict_market
