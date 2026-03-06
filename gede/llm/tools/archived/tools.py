# coding=utf-8

from agents import Tool

from .web_search import web_search_summary_tool
from .time_tool import now_tool
from .read_url_tool import read_page_tool


AVAILABLE_INNER_TOOLS_DICT = {
    "web_search": {
        "tool": web_search_summary_tool,
        "description": "search internet for recent information",
    },
    "now": {"tool": now_tool, "description": "get current date and time"},
    "read_page": {
        "tool": read_page_tool,
        "description": "read content from a given URL",
    },
}


AVAILABLE_INNER_TOOLS_SELECTOR = [
    (one[0] + ":" + one[1].get("description", ""), one[0])
    for one in AVAILABLE_INNER_TOOLS_DICT.items()
]

DEFAULT_INNER_TOOLS = ["web_search", "now", "read_page"]


def get_tools(*name: str):
    tools: list[Tool] = []
    for one in name:
        item = AVAILABLE_INNER_TOOLS_DICT.get(one)
        if item:
            tool = item.get("tool")
            if tool:
                tools.append(tool)
    return tools
