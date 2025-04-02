# this is all broken, do not use until the Google gen AI library is good
from google import genai
from google.genai import types
from typing import Callable, Optional, Any, Dict
from src.tools.search import Search
from logging import Logger
from pathlib import Path
from src.agent.utils import MarketPrediction, format_prompt, DEFAULT_INSTRUCTION
import json

from src.tools.search import Search, make_search_tools
from src.tools.python_interpreter import eval_python as eval_python_tool


def init_google(
    llm_config: dict,
    search: Search,
    logger: Logger,
    unified_web_search: bool,
    use_python_interpreter: bool,
    scratchpad_template_path: Optional[Path],
) -> Callable:
    client = genai.Client(
        api_key=llm_config["api_key"],
    )
    raw_search_tools = make_search_tools(search, unified_web_search)

    if unified_web_search:

        def web_search(query: str) -> list[dict]:
            """
            Search the web for query.

            Args:
                query: The query to search the web for.

            Returns:
                A list of dictionaries containing the search results.
            """
            return raw_search_tools[0](query)

        search_tools = [web_search]
    else:

        def get_relevant_urls(query: str) -> list[dict]:
            """
            Get the relevant URLs for the query.

            Args:
                query: The query to get the relevant URLs for.

            Returns:
                A list of URLs that are relevant to the query.
            """
            return raw_search_tools[0](query)

        def retrieve_web_content(url_list: list[str]) -> list[dict]:
            """
            Retrieve the web content for the URLs.

            Args:
                url_list: The list of URLs to retrieve the web content for.
            """
            return raw_search_tools[1](url_list)

        search_tools = [get_relevant_urls, retrieve_web_content]

    if use_python_interpreter:

        def eval_python(code: str) -> Dict[str, Any]:
            return eval_python_tool(code)

        search_tools.append(eval_python)

    template = None
    if scratchpad_template_path is not None:
        template = scratchpad_template_path.read_text

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
        response = client.models.generate_content(
            model=llm_config["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=DEFAULT_INSTRUCTION,
                tools=search_tools,
                response_mime_type="text/plain",
            ),
        )
        logger.info(response)
        # we need to parse the response text into a MarketPrediction object
        # until Google fixes their schemas lol
        reponse_parse_prompt = (
            "Parse the following text into a MarketPrediction object:\n\n"
            + response.candidates[0].content
        )
        response_parse_response = client.models.generate_content(
            model=llm_config["model"],
            contents=reponse_parse_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MarketPrediction,
            ),
        )
        logger.info(response_parse_response)
        # parse the response into a MarketPrediction object
        parsed = json.loads(response_parse_response.candidates[0].content)
        return MarketPrediction(**parsed)

    logger.info("Google agent initialized")

    return predict_market
