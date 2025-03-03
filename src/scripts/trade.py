import argparse
import json

from src.bot import init_from_config
from src.logging import create_logger


def main():
    parser = argparse.ArgumentParser(
        description="Run the prediction market trading bot"
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the bot configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    logger, _ = create_logger(config["name"], "trading", args.log_level)
    bot = init_from_config(config, logger)
    bot.run()


if __name__ == "__main__":
    main()
