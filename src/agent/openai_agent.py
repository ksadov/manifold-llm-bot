from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from logging import Logger

from src.tools.search import Search, make_search_tools
from src.tools.python_interpreter import eval_python
from typing import Optional, Any, Dict
from pathlib import Path
from collections.abc import Callable


class MarketPrediction(BaseModel):
    reasoning: str
    answer: float


def format_prompt(
    scratchpad_template: Optional[str],
    question: str,
    description: str,
    creatorUsername: str,
    comments: list[dict],
    current_date: str,
) -> str:
    if scratchpad_template is not None:
        template_instruction = f"Fill in the double-bracketed sections of the template according to the instructions, using relevant information from the web if needed. Then return the filled-in reasoning template as well as your final answer.\n\n{scratchpad_template}\n\n"
    else:
        template_instruction = ""

    return f"{template_instruction}Question: {question}\nDescription: {description}\nCreator Username: {creatorUsername}\nComments: {comments}\nCurrent Date: {current_date}"


def init_openai(
    llm_config: dict,
    search: Search,
    logger: Logger,
    unified_web_search: bool,
    use_python_interpreter: bool,
    scratchpad_template_path: Optional[Path],
) -> Callable:
    instruction = "You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can’t answer, pick the base rate, but return a number between 0 and 1."

    raw_search_tools = make_search_tools(search, unified_web_search)

    if unified_web_search:

        @function_tool
        def web_search(query: str) -> list[dict]:
            return raw_search_tools[0](query)

        search_tools = [web_search]
    else:

        @function_tool
        def get_relevant_urls(query: str) -> list[dict]:
            return raw_search_tools[0](query)

        @function_tool
        def retrieve_web_content(url_list: list[str] | dict) -> list[dict]:
            return raw_search_tools[1](url_list)

        search_tools = [get_relevant_urls, retrieve_web_content]

    if use_python_interpreter:

        @function_tool
        def eval_python(code: str) -> Dict[str, Any]:
            return eval_python(code)

        search_tools.append(eval_python)

    agent = Agent(
        name="Oracle",
        instructions=instruction,
        output_type=MarketPrediction,
        tools=search_tools,
        model=llm_config["model"],
    )

    print(agent)

    template = None
    if scratchpad_template_path is not None:
        template = scratchpad_template_path.read_text()

    def predict_market(
        question: str,
        description: str,
        creatorUsername: str,
        comments: list[dict],
        current_date: str,
    ) -> MarketPrediction:
        prompt = format_prompt(
            template, question, description, creatorUsername, comments, current_date
        )
        result = Runner.run_sync(agent, prompt)
        logger.debug(result.raw_responses)
        return result.final_output

    logger.info("OpenAI agent initialized")

    return predict_market
