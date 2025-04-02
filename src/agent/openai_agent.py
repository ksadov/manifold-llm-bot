from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from logging import Logger

from src.tools.search import Search
from typing import Optional
from pathlib import Path
from collections.abc import Callable


class MarketPrediction(BaseModel):
    reasoning: str
    answer: float


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
    logger: Logger,
    unified_web_search: bool,
    use_python_interpreter: bool,
    scratchpad_template_path: Optional[Path],
) -> Callable:
    instruction = "You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can’t answer, pick the base rate, but return a number between 0 and 1."

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
        output_type=MarketPrediction,
        tools=[get_relevant_urls, retrieve_web_content],
        model=llm_config["model"],
    )

    print(agent)

    def predict_market(
        question: str,
        description: str,
        creatorUsername: str,
        comments: list[dict],
        current_date: str,
    ) -> MarketPrediction:
        prompt = format_prompt(
            question, description, creatorUsername, comments, current_date
        )
        result = Runner.run_sync(agent, prompt)
        logger.debug(result.raw_responses)
        return result.final_output

    logger.info("OpenAI agent initialized")

    return predict_market
