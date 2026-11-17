"""
Standardized Logging Configuration.
Sets up JSON or standard formatting for logs across the entire RAG pipeline.
Ensures that errors, warnings, and evaluation metrics are properly piped to standard out
or log files for production monitoring.

Functions:
- setup_logger(name): Returns a pre-configured Python logger instance.
"""
import logging
import json
from pathlib import Path

log_path = Path(__file__).resolve().parent / "../data/log.jsonl"
log_path.parent.mkdir(parents=True, exist_ok=True)


class CustomJsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = f"{self.formatTime(record)} - {record.name} - {record.levelname} - {record.getMessage()} - {record.module}"
        
        return log_record

logger = logging.getLogger("AIRAG")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(CustomJsonFormatter())
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
file_handler.setFormatter(CustomJsonFormatter())
logger.addHandler(file_handler)

