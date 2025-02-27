import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
import ast


def convert_to_valid_json(line):
    """Convert Python-style dictionary strings to valid JSON."""
    try:
        # Try to parse the line as a Python literal using ast.literal_eval
        # This is safer than eval() and handles Python dict syntax properly
        python_obj = ast.literal_eval(line)
        # Convert to proper JSON
        return json.dumps(python_obj)
    except (SyntaxError, ValueError):
        # If ast.literal_eval fails, fall back to string replacement
        if "'" in line:
            # First, escape any existing double quotes
            line = line.replace('"', '\\"')
            # Then replace all single quotes with double quotes
            line = line.replace("'", '"')
        return line


def load_json_file(filepath):
    print("Loading data from", filepath)
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def process_data(markets_filepath, trades_filepath, comments_filepath, output_filepath):
    """Process the Manifold data into the desired format."""
    # Load data
    markets = load_json_file(markets_filepath)
    trades = load_json_file(trades_filepath)
    comments = load_json_file(comments_filepath)

    # Filter for binary markets
    binary_markets = [m for m in markets if m.get("outcomeType") == "BINARY"]
    print(f"Found {len(binary_markets)} binary markets")

    # Create lookup dictionaries
    market_id_to_data = {market["id"]: market for market in binary_markets}
    trades_by_market = defaultdict(list)
    comments_by_market = defaultdict(list)

    # Organize trades by market
    for trade in tqdm(trades, desc="Organizing trades"):
        contract_id = trade.get("contractId")
        if contract_id in market_id_to_data:
            # Extract only the needed fields
            trades_by_market[contract_id].append(
                {
                    "snapshotTime": trade.get("createdTime"),
                    "probability": trade.get("probAfter"),
                }
            )

    # Organize comments by market
    for comment in tqdm(comments, desc="Organizing comments"):
        contract_id = comment.get("contractId")
        if contract_id in market_id_to_data:
            # Extract comment text depending on the format
            if "text" in comment:
                comment_text = comment.get("text")
            elif "content" in comment:
                content = comment.get("content")
                # Handle different content structures
                if isinstance(content, dict):
                    # Complex nested structure
                    text_parts = []

                    def extract_text_from_content(content_obj):
                        """Recursively extract text from nested content structures"""
                        result = []
                        if isinstance(content_obj, dict):
                            if "text" in content_obj:
                                result.append(content_obj["text"])
                            if "content" in content_obj and isinstance(
                                content_obj["content"], list
                            ):
                                for item in content_obj["content"]:
                                    result.extend(extract_text_from_content(item))
                        elif isinstance(content_obj, list):
                            for item in content_obj:
                                result.extend(extract_text_from_content(item))
                        return result

                    text_parts = extract_text_from_content(content)
                    comment_text = " ".join(text_parts) if text_parts else str(content)
                else:
                    comment_text = str(content)  # Fallback for other structures
            else:
                comment_text = str(comment)  # Fallback

            comments_by_market[contract_id].append(
                {
                    "id": comment.get("id"),
                    "text": comment_text,
                    "userName": comment.get("userName"),
                    "createdTime": comment.get("createdTime"),
                }
            )

    # Construct the final dataset
    final_dataset = []

    for market_id, market_data in tqdm(
        market_id_to_data.items(), desc="Constructing final dataset"
    ):
        # Skip markets without trades for simplicity
        if market_id not in trades_by_market:
            continue

        # Sort trades by time
        sorted_trades = sorted(
            trades_by_market[market_id], key=lambda x: x["snapshotTime"]
        )

        # Create the market entry
        market_entry = {
            "question": market_data.get("question", ""),
            "createdTime": market_data.get("createdTime"),
            "description": market_data.get("description", ""),
            "creatorUsername": market_data.get("creatorUsername", ""),
            "id": market_id,
            "groupSlugs": market_data.get("groupSlugs", []),
            "resolution": market_data.get("resolution"),
            "resolutionTime": market_data.get("resolutionTime"),
            "tradeHistory": sorted_trades,
            "comments": comments_by_market.get(market_id, []),
        }

        # Handle description field which might be a dictionary
        if isinstance(market_entry["description"], dict):
            # Try to extract text from complex description structure
            content = market_entry["description"].get("content", [])
            text_parts = []

            def extract_text(content_list):
                texts = []
                for item in content_list:
                    if isinstance(item, dict):
                        if "text" in item:
                            texts.append(item["text"])
                        elif "content" in item and isinstance(item["content"], list):
                            texts.extend(extract_text(item["content"]))
                return texts

            if isinstance(content, list):
                text_parts = extract_text(content)
                market_entry["description"] = " ".join(text_parts)
            else:
                market_entry["description"] = str(market_entry["description"])

        final_dataset.append(market_entry)

    print(f"Created {len(final_dataset)} entries in the final dataset")

    # Convert to DataFrame and save as Parquet
    df = pd.DataFrame(final_dataset)

    # Convert complex columns to strings to ensure Parquet compatibility
    for col in ["tradeHistory", "comments"]:
        df[col] = df[col].apply(json.dumps)

    df.to_parquet(output_filepath, index=False)
    print(f"Saved dataset to {output_filepath}")

    # Also save a sample as JSON for inspection
    sample_size = min(100, len(final_dataset))
    with open(output_filepath.replace(".parquet", "_sample.json"), "w") as f:
        json.dump(final_dataset[:sample_size], f, indent=2)
    print(f"Saved sample to {output_filepath.replace('.parquet', '_sample.json')}")


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--markets_filepath", type=str)
    arg.add_argument("--trades_filepath", type=str)
    arg.add_argument("--comments_filepath", type=str)
    arg.add_argument("--output_filepath", type=str, default="manifold_dataset.parquet")
    args = arg.parse_args()

    process_data(
        markets_filepath=args.markets_filepath,
        trades_filepath=args.trades_filepath,
        comments_filepath=args.comments_filepath,
        output_filepath=args.output_filepath,
    )
