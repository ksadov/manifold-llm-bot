import logging
import json
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def create_logger() -> logging.Logger:
    logger = logging.getLogger("BotLogger")
    logger.setLevel(logging.DEBUG)

    handler = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=5)
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also log to console at INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
