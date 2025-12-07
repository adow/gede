# coding=utf-8

import os
import logging
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

VERSION = "0.3.16"

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


def gede_dir():
    return os.path.join(Path.home(), ".gede")
