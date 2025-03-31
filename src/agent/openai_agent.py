from agents import Agent, Runner, function_tool
from pydantic import BaseModel

from src.tools.search import Search
from src.logging import Logger
from typing import Optional
from pathlib import Path
from collections.abc import Callable


class Answer(BaseModel):
    reasoning: str
    probability: float


def format_prompt(
    question: str,
    description: str,
    creatorUsername: str,
    comments: list[dict],
    current_date: str,
) -> str:
    return f"Question: {question}\nDescription: {description}\nCreator Username: {creatorUsername}\nComments: {comments}\nCurrent Date: {current_date}"


def init_openai(
    llm_config: dict,
    search: Search,
    unified_web_search: bool,
    use_python_interpreter: bool,
    scratchpad_template_path: Optional[Path],
    logger: Optional[Logger] = None,
) -> Callable:
    # TODO: finish this
    instruction = "You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can’t answer, pick the base rate, but return a number between 0 and 1."
    lm = dspy.LM(
        f'openai/{llm_config["model"]}',
        api_key=llm_config["api_key"],
        api_base=llm_config["api_base"],
        **llm_config["prompt_params"],
    )

    @function_tool
    def get_relevant_urls(query: str) -> list[dict]:
        results = search.get_results(query)
        result_dicts = [result.to_dict() for result in results]
        return result_dicts

    @function_tool
    def retrieve_web_content(url_list: list[str] | dict) -> list[dict]:
        if isinstance(url_list, dict) and "items" in url_list:
            url_list = list(url_list["items"])
        cleaned_html = [search.retrieve_cleaned_html(url) for url in url_list]
        result_dicts = [
            {"url": url, "cleaned_html_content": html}
            for url, html in zip(url_list, cleaned_html)
        ]
        return result_dicts

    agent = Agent(
        name="Oracle",
        instructions=instruction,
        output_type=Answer,
        tools=[get_relevant_urls, retrieve_web_content],
    )

    def predict_market(
        question: str,
        description: str,
        creatorUsername: str,
        comments: list[dict],
        current_date: str,
    ) -> Answer:
        prompt = format_prompt(
            question, description, creatorUsername, comments, current_date
        )
        return Runner.run_sync(agent, prompt)

    return predict_market
