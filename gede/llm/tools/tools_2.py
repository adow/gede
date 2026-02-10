# coding=utf-8
#
#

from my_llmkit.chat.tools import ToolFunctions
from .now_tool import now_tool
from .web_serach_2 import web_search_tool
from .read_url_2 import read_url_tool

# 所有内置可用工具
AVAILABLE_INNER_TOOLS = ToolFunctions(now_tool, web_search_tool, read_url_tool)

AVAILABLE_INNER_TOOLS_SELECTOR = [
    (one.name + ":" + one.tool_param.description.splitlines()[0], one.name)
    for one in AVAILABLE_INNER_TOOLS.tool_function_list
]

DEFAULT_INNER_TOOLS_USED = ["now_tool"]


def get_tools(*name: str):
    tool_functions: ToolFunctions = ToolFunctions()
    for one_name in name:
        for tool in AVAILABLE_INNER_TOOLS.tool_function_list:
            if one_name == tool.name:
                tool_functions.tool_function_list.append(tool)
                break
    return tool_functions
