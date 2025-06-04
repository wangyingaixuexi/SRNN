from typing import List
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging


created_loggers: List[logging.Logger] = []

class LoggingConfig():
    _file = None
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    _level = logging.INFO

    @staticmethod
    def set_file(file: Path):
        LoggingConfig._file = file
        file_handler = logging.FileHandler(file, mode='w')
        file_handler.setFormatter(LoggingConfig.formatter)
        for logger in created_loggers:
            logger.addHandler(file_handler)

    @staticmethod
    def disable_console_output():
        for logger in created_loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    logger.removeHandler(handler)
        sys.stdout = open(os.devnull, 'w')

    @staticmethod
    def set_level(level):
        for logger in created_loggers:
            logger.setLevel(level)

def get_predefined_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(LoggingConfig.formatter)
    logger.addHandler(stream_handler)
    created_loggers.append(logger)

    return logger

def get_timestamp() -> str:
    current_time = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
    timestamp = current_time.strftime('%Y-%m-%dT%H.%M.%S')
    return timestamp
