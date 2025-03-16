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
    bot = init_from_config(args.config_path, args.log_level)
    bot.run()


if __name__ == "__main__":
    main()
