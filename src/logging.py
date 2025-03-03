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


def create_logger(bot_name: str, label: str, log_level: str) -> logging.Logger:
    logger = logging.getLogger(bot_name)
    logger.setLevel(log_level)
    print(f"Logging level: {log_level}")

    logfile_name = f"{bot_name}-{label}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    handler = RotatingFileHandler(
        f"logs/{logfile_name}", maxBytes=1_000_000, backupCount=5
    )
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, logfile_name
