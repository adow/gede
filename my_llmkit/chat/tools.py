# coding=utf-8
#
# 工具定义与执行器
#
import inspect
import json
import logging
from typing import Callable, Optional

from agents.function_schema import function_schema

from ..mcp.mcp_client import MCPServerType
from .types import UnifiedToolCall, UnifiedToolParam


class ToolFunction:
    def __init__(self, func: Callable):
        self.func = func
        schema = function_schema(func)
        self.name = schema.name
        self.tool_param = UnifiedToolParam(
            name=schema.name,
            description=schema.description or "",
            parameters=schema.params_json_schema,
        )


class ToolFunctions:
    def __init__(self, *funcs: Callable):
        self.tool_function_list = [ToolFunction(f) for f in funcs]

    @property
    def tool_params(self) -> list[UnifiedToolParam]:
        """返回统一格式的工具参数列表"""
        return [tf.tool_param for tf in self.tool_function_list]

    @property
    def tool_definitions(self) -> dict[str, Callable]:
        return {tf.name: tf.func for tf in self.tool_function_list}


# mcp


def make_mcp_tool_name(server_name: str, tool_name: str) -> str:
    return f"__mcp_{server_name}_{tool_name}"


class ToolExecutor:
    """公共工具执行器，供流式和非流式共用"""

    def __init__(
        self,
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
    ):
        self.tools = tools
        self.mcp_servers = mcp_servers

    def get_tool_func(self, tool_name: str) -> Optional[Callable]:
        if not self.tools:
            return None
        return self.tools.tool_definitions.get(tool_name)

    def get_mcp_server(self, server_name: str) -> Optional[MCPServerType]:
        if not self.mcp_servers:
            return None
        return self.mcp_servers.get(server_name)

    async def execute(self, tool_call: UnifiedToolCall) -> tuple[str, Optional[str]]:
        """
        执行单个工具调用

        Returns:
            tuple[str, Optional[str]]: (工具结果, 错误信息)
        """
        function_name = tool_call.name
        function_args = tool_call.arguments

        # 确保 arguments 是有效 JSON
        if not function_args or function_args.strip() == "":
            function_args = "{}"

        # 调用 MCP 工具
        if function_name.startswith("_mcp_"):
            return await self._execute_mcp_tool(function_name, function_args)
        else:
            return await self._execute_builtin_tool(function_name, function_args)

    async def _execute_mcp_tool(
        self, function_name: str, function_args: str
    ) -> tuple[str, Optional[str]]:
        """执行 MCP 工具"""
        names = [one for one in function_name.split("_", maxsplit=3) if one.strip()]
        if len(names) < 3:
            error = f"Invalid MCP tool name: {function_name}"
            logging.error(error)
            return f"Error: {error}", error

        _, server_name, tool_name = names
        logging.info(f"Calling MCP Tool: Server={server_name}, Tool={tool_name}")

        server = self.get_mcp_server(server_name)
        if not server:
            error = f"MCP Server {server_name} not found"
            logging.error(error)
            return f"Error: {error}", error

        try:
            kwargs = json.loads(function_args)
            result = await server.call_tool(tool_name, kwargs)
            return result.model_dump_json(), None
        except Exception as e:
            error = f"Error calling MCP tool {tool_name}: {e}"
            logging.error(error)
            return f"Error: {error}", error

    async def _execute_builtin_tool(
        self, function_name: str, function_args: str
    ) -> tuple[str, Optional[str]]:
        """执行内置工具"""
        func = self.get_tool_func(function_name)
        if not func:
            error = f"Tool {function_name} not found"
            return f"Error: {error}", error

        try:
            logging.info(f"Executing tool: {function_name}")
            kwargs = json.loads(function_args)
            result = func(**kwargs)

            if inspect.isawaitable(result):
                result = await result

            if not isinstance(result, str):
                return json.dumps(result, ensure_ascii=False), None
            return result, None
        except Exception as e:
            error = f"Error executing {function_name}: {e}"
            logging.error(error)
            return f"Error: {str(e)}", error
