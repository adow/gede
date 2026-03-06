# coding=utf-8
#
# top.py
# 全局配置与路径管理
#

import os
from pathlib import Path

from rich.console import Console
from rich.theme import Theme

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


def gede_instructions_dir():
    instruction_dir = os.path.join(gede_dir(), "instructions")
    if not os.path.exists(instruction_dir):
        os.makedirs(instruction_dir)
    return instruction_dir


def gede_prompts_dir():
    prompts_dir = os.path.join(gede_dir(), "prompts")
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
    return prompts_dir


def gede_config_dir():
    config_dir = os.path.join(gede_dir(), "config")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return config_dir


def gede_mcp_config_path():
    """返回 MCP 配置文件的完整路径"""
    return os.path.join(gede_config_dir(), "mcp.json")
