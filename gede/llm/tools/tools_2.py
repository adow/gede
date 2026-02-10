# coding=utf-8
#
#

from .now_tool import now_tool
from my_llmkit.chat.tools import ToolFunctions

# 所有内置可用工具
AVAILABLE_INNER_TOOLS = ToolFunctions(now_tool)

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
