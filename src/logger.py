import logging
from config import load_config


def get_logger(name: str):
    config = load_config()

    log_level = config.get("logging", {}).get("level", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return logging.getLogger(name)