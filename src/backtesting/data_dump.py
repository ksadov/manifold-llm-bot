import json

from typing import List


def load_market_data(data_path: str):
    print("Loading data from", data_path)
    with open(data_path) as f:
        data = json.load(f)
    print(data[1000])
    print(data[9999])
    print(data[-1])
    print(data[-5000])


def test():
    data_path = "/Users/ksadov/Documents/manifold2025/bot/data_dump_2024-07/manifold-comments-20240706hi.json"
    load_market_data(data_path)


if __name__ == "__main__":
    test()
