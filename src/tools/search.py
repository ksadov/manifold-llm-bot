from calendar import c
import html
from typing import Optional
import requests
import datetime
import json
from pathlib import Path
import dspy

from src.logging import create_logger


class CleanHTML(dspy.Signature):
    "Extract the main text content from this HTML, preserving paragraph structure but removing all HTML tags, scripts, styles, and extraneous formatting."

    html: str = dspy.InputField()
    clean_text: str = dspy.OutputField()


class SearchResult:
    def __init__(self, item: dict):
        self.title = item.get("og:title", item["title"])
        self.link = item["link"]
        self.snippet = item.get("og:description", item["snippet"])
        self.retrieved_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet,
            "retrieved_timestamp": self.retrieved_timestamp,
        }

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self)


class Search:
    def __init__(
        self,
        google_cse_key: str,
        google_cse_cx: str,
        num_search_results: int,
        max_html_length: int,
        cutoff_date: Optional[str] = None,
    ):
        self.api_key = google_cse_key
        self.cx = google_cse_cx
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        self.max_html_length = max_html_length
        self.num_search_results = num_search_results
        self.lm = dspy.LM("gemini/gemini-2.0-flash-lite", api_key=google_cse_key)
        self.html_cleaner = dspy.Predict(CleanHTML)
        if cutoff_date:
            self.date_restriction_string = f"date:r::{self.format_date(cutoff_date)}"
        else:
            self.date_restriction_string = None

    def format_date(self, date: str) -> str:
        return datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

    def set_cutoff_date(self, cutoff_date: str):
        self.date_restriction_string = f"date:r::{self.format_date(cutoff_date)}"
        return self

    def get_results(self, query: str) -> list[SearchResult]:
        response_params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": self.num_search_results,
        }
        if self.date_restriction_string:
            response_params["sort"] = self.date_restriction_string
        res = requests.get(
            self.endpoint,
            params=response_params,
        )
        res.raise_for_status()
        results = [SearchResult(item) for item in res.json()["items"]]
        return results

    def ai_clean_html(self, html: str) -> str:
        if self.max_html_length is not None and len(html) > self.max_html_length:
            html = html[: self.max_html_length]
        with dspy.context(lm=self.lm):
            clean_text = self.html_cleaner(html=html)
        return clean_text

    def retrieve_cleaned_html(self, url):
        try:
            response = requests.get(url)
            clean_html = self.ai_clean_html(response.text)
        except Exception as e:
            clean_html = f"Error retrieving or processing HTML: {e}"
        return clean_html


def init_search(config_path: Path) -> Search:
    # Load config from file
    with open(config_path) as f:
        config = json.load(f)
    secrets_json_path = Path(config["secrets_path"])
    # Load secrets from file
    with open(secrets_json_path) as f:
        secrets = json.load(f)
    # Initialize search
    search = Search(
        secrets["google_api_key"],
        secrets["google_cse_cx"],
        config["max_search_results"],
        config["max_html_length"],
    )
    return search


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


def test():
    query = "prediction markets"
    secret_path = "config/secrets/basic_secrets.json"
    cutoff_date = datetime.datetime(2005, 1, 1).strftime("%Y-%m-%d")
    with open(secret_path) as f:
        secrets = json.load(f)
    search = Search(
        secrets["google_api_key"],
        secrets["google_cse_cx"],
        3,
        None,
        cutoff_date,
    )

    results = search.get_results(query)

    for result in results:
        clean_html = search.retrieve_cleaned_html(result.link)
        print(result)
        print(clean_html)


if __name__ == "__main__":
    test()
