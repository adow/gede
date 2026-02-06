# coding=utf-8
#
#

from dataclasses import dataclass, field
import dataclasses
from typing import List, Optional

from prompt_toolkit import PromptSession
from rich.console import Console

from my_llmkit.mcp import MCPServerBase, MCPServerType
from my_llmkit.chat import ToolFunctions
from .chatcore2 import ChatModel
from .display import NotificationRenderer, InfoRenderer


@dataclass
class Context:
    console: Console
    prompt_session: PromptSession

    # Current chat session
    current_chat: "ChatModel"

    message: Optional[str] = None

    # Display renderers
    notification_display: NotificationRenderer = field(init=False)
    info_display: InfoRenderer = field(init=False)

    mcp_servers: dict[str, MCPServerBase] = field(default_factory=dict)

    tools: ToolFunctions = field(default_factory=ToolFunctions)

    def __init__(
        self,
        console: Console,
        prompt_session: PromptSession,
        current_chat: "ChatModel",
        message: Optional[str] = None,
        mcp_servers: Optional[dict[str, MCPServerBase]] = None,
        tools: Optional[ToolFunctions] = None,
    ):
        self.console = console
        self.prompt_session = prompt_session
        self.current_chat = current_chat
        self.message = message
        self.mcp_servers = mcp_servers if mcp_servers is not None else {}
        self.tools = tools if tools is not None else ToolFunctions()
        # 初始化渲染器
        self.notification_display = NotificationRenderer(console)
        self.info_display = InfoRenderer(console)

    async def print_chat_info(self):
        """打印聊天信息面板"""
        tools_info = (
            "[bold]Using Tools[/bold]: "
            + ",".join([one.name for one in self.tools.tool_function_list])
            if self.tools
            else "None"
        )
        server_names = self.mcp_servers.keys()
        if server_names:
            mcp_info = "[bold]Using MCP Servers[/bold]: " + ",".join(server_names)
        else:
            mcp_info = "[bold]Using MCP Servers[/bold]: None"

        chat_info = await self.current_chat.info
        self.info_display.chat_info(chat_info, tools_info, mcp_info)

    def print_tool_info(self, description: str):
        """打印工具信息面板"""
        # TODO: 打印当前的工具信息
        self.info_display.tool_info(description)

    def print_instruction(self):
        """打印系统指令"""
        self.info_display.instruction(self.current_chat.instruction)

    def print_rule(self, title: str):
        self.info_display.rule(title)

    def print_model_settings(self):
        self.info_display.model_settings(self.current_chat.user_model_settings)
