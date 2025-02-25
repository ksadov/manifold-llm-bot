import re
import time
from typing import Optional
import requests
import datetime
import bs4
from bs4.element import Comment


def clean_html(html_text):
    """
    Clean the HTML content returned from a request.

    Args:
        html_text (str): The raw HTML text to be cleaned

    Returns:
        str: Cleaned HTML text
    """
    # Import required libraries
    from bs4 import BeautifulSoup

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove extra whitespace and normalize
    cleaned_text = soup.get_text(separator=" ", strip=True)
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text


class SearchResult:
    def __init__(self, item: dict, max_html_length: Optional[int]):
        self.title = item.get("og:title", item["title"])
        self.link = item["link"]
        self.snippet = item.get("og:description", item["snippet"])
        self.retrieved_timestamp = datetime.datetime.now()
        self.html = None
        self.max_html_length = max_html_length
        self.html_truncated = False

    def to_dict(self):
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet,
            "retrieved_timestamp": self.retrieved_timestamp,
            "cleaned_html": self.html,
            "html_truncated": self.html_truncated,
        }

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self)

    def add_html(self):
        html_results = requests.get(self.link)
        html_results.raise_for_status()
        cleaned = clean_html(html_results.text)
        if self.max_html_length is not None and len(cleaned) > self.max_html_length:
            self.html = cleaned[: self.max_html_length]
            self.html_truncated = True
        else:
            self.html = cleaned


class Search:
    def __init__(self, google_cse_key: str, google_cse_cx: str, max_html_length: int):
        self.api_key = google_cse_key
        self.cx = google_cse_cx
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        self.max_html_length = max_html_length

    def get_results(self, query: str, retrieve_html: bool) -> list[SearchResult]:
        res = requests.get(
            self.endpoint,
            params={"key": self.api_key, "cx": self.cx, "q": query},
        )
        res.raise_for_status()
        results = [
            SearchResult(item, self.max_html_length) for item in res.json()["items"]
        ]
        if retrieve_html:
            for result in results:
                try:
                    result.add_html()
                except Exception as e:
                    result.html = f"Error retrieving HTML content: {e}"
        print("results done!")
        return results


def main():
    query = "prediction markets"
    search = Search()
    results = search.results(query)
    print(results)


if __name__ == "__main__":
    main()
