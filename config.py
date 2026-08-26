from pathlib import Path
import yaml


CONFIG_PATH = Path("config.yaml")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.yaml was not found.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config