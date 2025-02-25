import argparse

from src.bot import init_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Run the prediction market trading bot"
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the bot configuration file"
    )
    args = parser.parse_args()
    bot = init_from_config(args.config_path)
    bot.run()


if __name__ == "__main__":
    main()
