# coding=utf-8
#
# base.py
# Base command class and context
#

from __future__ import annotations  # 启用延迟注解解析

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
from contextlib import AsyncExitStack

from prompt_toolkit import PromptSession
from rich.console import Console
from agents.mcp import MCPServer

from ..llm.mcp.mcp_manager import MCPServerManager

# 仅用于类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from ..chatcore import ChatModel


@dataclass
class CommandConext:
    console: Console

    # Current user input message
    message: str

    # Current chat session
    current_chat: ChatModel  # 现在可以直接使用，因为有 TYPE_CHECKING
    mcp_manager: MCPServerManager
    prompt_session: PromptSession

    tools: list[str] = field(default_factory=list)

    mcp_servers: list[MCPServer] = field(default_factory=list)

    stack: Optional[AsyncExitStack] = None

    def __init__(
        self,
        console: Console,
        message: str,
        current_chat: ChatModel,  # 这里也可以直接使用
        prompt_session: PromptSession,
        tools: List[str] = [],
        mcp_servers: List[MCPServer] = [],
        stack: Optional[AsyncExitStack] = None,
    ):
        self.console = console
        self.message = message
        self.current_chat = current_chat
        self.tools = tools
        self.mcp_servers = mcp_servers
        self.mcp_manager = MCPServerManager.get_manager_from_config()
        self.prompt_session = prompt_session
        self.stack = stack

    def print_chat_info(self):
        from rich.panel import Panel

        tools_info = (
            "[bold]Using Tools[/bold]: " + ",".join(self.tools)
            if self.tools
            else "None"
        )
        servers = self.mcp_manager.get_running_servers()
        server_names = [one.name for one in servers]
        if server_names:
            mcp_info = "[bold]Using MCP Servers[/bold]: " + ",".join(server_names)
        else:
            mcp_info = "[bold]Using MCP Servers[/bold]: None"

        chat_info = self.current_chat.info + "\n" + tools_info + "\n" + mcp_info

        self.console.print(
            Panel(chat_info, title="Chat Info", expand=True),
            style="info",
        )

    def print_tool_info(self, description: str):
        from rich.panel import Panel

        self.console.print()
        self.console.print(Panel(f"🧰 {description}", expand=False), style="warning")


class CommandBase:
    def __init__(self, context: CommandConext):
        self.context = context
        self.console = context.console
        self.message = context.message.strip().lower()

    def do_command(self) -> bool:
        """
        Execute command
        Returns: whether to continue subsequent execution
        """
        raise NotImplementedError("Subclasses must implement do_command method")

    async def do_command_async(self) -> bool:
        return True

    def print_instruction(self):
        self.console.print(
            f"[bold]System:[/bold] {self.context.current_chat.instruction}",
            style="system",
        )

    @property
    def doc_title(self) -> str:
        return ""

    @property
    def doc_description(self) -> str:
        return ""

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return None
