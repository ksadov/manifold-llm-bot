from calendar import c
from typing import Optional
import requests
import datetime
from google import genai
import json
from pathlib import Path


def ai_clean_html(client, html, max_tokens):
    config = genai.types.GenerateContentConfig(max_output_tokens=max_tokens)
    prompt = f"Extract only the human-readable text from this HTML document (excluding CSS, Javascript, HTML tags, etc) and format it with Markdown syntax:\n\n{html}"
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=prompt, config=config
    )
    return response.text


class SearchResult:
    def __init__(self, item: dict, ai_client, max_html_length: Optional[int]):
        self.title = item.get("og:title", item["title"])
        self.link = item["link"]
        self.snippet = item.get("og:description", item["snippet"])
        self.retrieved_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ai_client = ai_client

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
        cutoff_date: Optional[datetime.datetime] = None,
    ):
        self.api_key = google_cse_key
        self.cx = google_cse_cx
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        self.max_html_length = max_html_length
        self.num_search_results = num_search_results
        self.ai_client = genai.Client(api_key=google_cse_key)
        if cutoff_date:
            self.date_restriction_string = f"date:r::{cutoff_date.strftime('%Y%m%d')}"
        else:
            self.date_restriction_string = None

    def set_cutoff_date(self, cutoff_date: datetime.datetime):
        self.date_restriction_string = f"date:r::{cutoff_date.strftime('%Y%m%d')}"
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
        results = [
            SearchResult(item, self.ai_client, self.max_html_length)
            for item in res.json()["items"]
        ]
        return results

    def retrieve_cleaned_html(self, url):
        try:
            response = requests.get(url)
            clean_html = ai_clean_html(
                self.ai_client, response.text, self.max_html_length
            )
        except Exception as e:
            clean_html = f"Error retrieving or processing HTML: {e}"
        return clean_html


def init_search(config_path: Path, cutoff_date: datetime.datetime) -> Search:
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
        cutoff_date=cutoff_date,
    )
    return search


def test():
    query = "prediction markets"
    secret_path = "config/secrets/basic_secrets.json"
    cutoff_date = datetime.datetime(2005, 1, 1)
    with open(secret_path) as f:
        secrets = json.load(f)
    search = Search(
        secrets["google_api_key"],
        secrets["google_cse_cx"],
        3,
        10000,
        cutoff_date,
    )

    results = search.get_results(query)
    print(results)

    clean_html = search.retrieve_cleaned_html(results[0].link)
    print(clean_html)


if __name__ == "__main__":
    test()
