from calendar import c
from typing import Optional
import requests
import datetime
from google import genai
import json


def ai_clean_html(client, html, max_tokens):
    config = genai.types.GenerateContentConfig(max_output_tokens=max_tokens)
    prompt = f"Extract only the human-readable text from this HTML document and format with Markdown syntax:\n\n{html}"
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
        self.html = None
        self.max_html_length = max_html_length
        self.ai_client = ai_client

    def to_dict(self):
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet,
            "retrieved_timestamp": self.retrieved_timestamp,
            "cleaned_html": self.html,
        }

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self)

    def add_html(self):
        html_results = requests.get(self.link)
        html_results.raise_for_status()
        cleaned = ai_clean_html(self.ai_client, html_results.text, self.max_html_length)
        self.html = cleaned


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

    def get_results(self, query: str, retrieve_html: bool) -> list[SearchResult]:
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
        if retrieve_html:
            for result in results:
                try:
                    result.add_html()
                except Exception as e:
                    result.html = f"Error retrieving HTML content: {e}"
        return results


def test():
    query = "prediction markets"
    secret_path = "config/secrets/basic_secrets.json"
    cutoff_date = datetime.datetime(2005, 1, 1)
    with open(secret_path) as f:
        secrets = json.load(f)
    search = Search(
        secrets["google_api_key"], secrets["google_cse_cx"], 3, 10000, cutoff_date
    )
    results = search.get_results(query, True)
    print(results)


if __name__ == "__main__":
    test()
