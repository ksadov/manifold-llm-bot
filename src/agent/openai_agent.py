import asyncio

from agents import Agent, Runner, function_tool
from logging import Logger

from src.tools.search import Search, make_search_tools
from src.tools.python_interpreter import eval_python as eval_python_tool
from typing import Optional, Any, Dict
from pathlib import Path
from collections.abc import Callable
from src.agent.utils import MarketPrediction, format_prompt, DEFAULT_INSTRUCTION


def init_openai(
    llm_config: dict,
    search: Search,
    logger: Logger,
    unified_web_search: bool,
    use_python_interpreter: bool,
    scratchpad_template_path: Optional[Path],
) -> Callable:

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
            return eval_python_tool(code)

        search_tools.append(eval_python)

    agent = Agent(
        name="Oracle",
        instructions=DEFAULT_INSTRUCTION,
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
        cutoff_date: Optional[str] = None,
    ) -> MarketPrediction:
        if cutoff_date is not None:
            search.set_cutoff_date(cutoff_date)
        prompt = format_prompt(
            template, question, description, creatorUsername, comments, current_date
        )

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = Runner.run_sync(agent, prompt)
        finally:
            loop.close()

        logger.debug(result.raw_responses)
        return result.final_output

    logger.info("OpenAI agent initialized")

    return predict_market
