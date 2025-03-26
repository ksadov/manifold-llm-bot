import argparse

from src.bot import init_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Run the prediction market trading bot"
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the bot configuration file"
    )
    parser.add_argument(
        "--max_trade_time",
        type=int,
        default=None,
        help="Maximum trade time in seconds",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()
    bot = init_from_config(args.config_path, args.log_level, args.max_trade_time)
    bot.run()


if __name__ == "__main__":
    main()
