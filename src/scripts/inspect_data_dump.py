import json
import argparse
import random


def inspect_data(data_path: str, n_data_points: int):
    print("Loading data from", data_path)
    with open(data_path) as f:
        data = json.load(f)

    # pick 5 random data points to print
    data_indexes = [random.randint(0, len(data)) for _ in range(n_data_points)]
    for index in data_indexes:
        print(f"Data point {index}:")
        print(json.dumps(data[index], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to the data file")
    parser.add_argument(
        "--n_data_points",
        type=int,
        default=5,
        help="Number of data points to inspect",
    )
    args = parser.parse_args()
    inspect_data(args.data_path, args.n_data_points)


if __name__ == "__main__":
    main()
