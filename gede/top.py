# coding=utf-8

import os
import logging
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

VERSION = "0.3.24"

# Set log format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
rh = RichHandler()
rh.setFormatter(formatter)

logger = logging.getLogger("gede")
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.ERROR)
# logger.setLevel(logging.CRITICAL)
logger.addHandler(rh)

agent_logger = logging.getLogger("agents")
agent_logger.addHandler(rh)

custom_theme = Theme(
    {
        "info": "dim cyan",
        "system": "dim",
        "input": "dim bold",
        "warning": "magenta",
        "danger": "bold red",
    }
)
console = Console(theme=custom_theme)


# DEFAULT_MODEL_PATH = "voice_engine:doubao-seed-1-6"
DEFAULT_MODEL_PATH = "openrouter:google/gemini-3-pro-preview"


def gede_dir():
    return os.path.join(Path.home(), ".gede")


def gede_cache_dir():
    cache_dir = os.path.join(gede_dir(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def gede_data_dir():
    data_dir = os.path.join(gede_dir(), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir
