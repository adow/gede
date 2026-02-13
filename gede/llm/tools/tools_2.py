# coding=utf-8
#
#

from my_llmkit.chat.tools import ToolFunctions
from .now_tool import now
from .web_serach_tool2 import web_search
from .read_url_tool2 import read_url

# 所有内置可用工具
AVAILABLE_INNER_TOOLS = ToolFunctions(now, web_search, read_url)

AVAILABLE_INNER_TOOLS_SELECTOR = [
    (one.name + ":" + one.tool_param.description.splitlines()[0], one.name)
    for one in AVAILABLE_INNER_TOOLS.tool_function_list
]


def get_tools(*name: str):
    tool_functions: ToolFunctions = ToolFunctions()
    for one_name in name:
        for tool in AVAILABLE_INNER_TOOLS.tool_function_list:
            if one_name == tool.name:
                tool_functions.tool_function_list.append(tool)
                break
    return tool_functions
