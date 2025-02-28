import logging
import json
import os
import datetime
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def create_logger(bot_name: str, eval: bool, log_level: str) -> logging.Logger:
    logger = logging.getLogger(bot_name)
    logger.setLevel(log_level)

    os.makedirs("logs", exist_ok=True)

    logfile_name = f"{bot_name}-{'eval' if eval else ''}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    handler = RotatingFileHandler(
        f"logs/{logfile_name}", maxBytes=1_000_000, backupCount=5
    )
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also log to console at INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, logfile_name
