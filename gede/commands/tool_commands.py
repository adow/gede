# coding=utf-8
#
# tool_commands.py
# Tool-related commands
#

from typing import Optional, List, Any

import inquirer

from .base import CommandBase


class SelectToolsCommand(CommandBase):
    def do_command(self) -> bool:
        from ..llm.tools.tools import (
            AVAILABLE_INNER_TOOLS_SELECTOR,
        )

        command = "/select-tools"
        if self.message == command:
            question = [
                inquirer.Checkbox(
                    "tools",
                    message="Select tools to enable (use SPACE to select/deselect, ENTER to confirm)",
                    choices=AVAILABLE_INNER_TOOLS_SELECTOR,
                    default=self.context.tools,
                )
            ]

            answers: Optional[dict[str, Any]] = inquirer.prompt(question)
            if answers and "tools" in answers:
                selected_tools: List[str] = answers["tools"]
                self.context.console.print(
                    f"Select {len(selected_tools)} tools", style="info"
                )

                # Extract tool objects
                self.context.tools = selected_tools

            else:
                self.console.print("No tools selected", style="warning")

            return False
        return True

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/select-tools"

    @property
    def doc_title(self) -> str:
        return "/select-tools \nSelect tools to enable"

    @property
    def doc_description(self) -> str:
        return """Choose from a list of available tools to enable for the current chat session. Use SPACE to select or deselect tools, and ENTER to confirm your selection. The selected tools will be available for use during the chat."""


class SelectMCPCommand(CommandBase):
    async def connect(self, selected_mcp: list[str]):
        from ..top import logger

        for name, item in self.context.mcp_manager.server_items.items():
            if name in selected_mcp:
                if not item.selected:
                    # Use stack to manage server lifecycle
                    if self.context.stack:
                        # Connect only once
                        if not item.connected:
                            await self.context.stack.enter_async_context(item.server)
                            item.connected = True
                            logger.info('Connected to MCP server "%s"', name)
                        else:
                            logger.info(
                                'Using existing connection to MCP server "%s"', name
                            )
                        await item.select_server(self.context.console)
                    logger.debug(f'MCP server "{name}" connected')
            else:
                if item.selected:
                    item.selected = False
                    logger.debug(f'MCP server "{name}" disconnected')

    async def do_command_async(self) -> bool:
        command = "/select-mcp"
        if self.message == command:
            choices: list[str] = []
            default_selected: list[str] = []
            for name, item in self.context.mcp_manager.server_items.items():
                choices.append(name)
                if item.selected:
                    default_selected.append(name)
            if not choices:
                self.context.console.print("No MCP server found", style="warning")
                return False
            question = [
                inquirer.Checkbox(
                    "mcp",
                    message="Select MCP servers to connect (use SPACE to select/deselect, ENTER to confirm)",
                    choices=choices,
                    default=default_selected,
                )
            ]
            answers = inquirer.prompt(question)
            if answers and "mcp" in answers:
                selected_mcp: List[str] = answers["mcp"]
                self.context.console.print(f"Select mcp: {selected_mcp}", style="info")
                await self.connect(selected_mcp)

            return False

        return True

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/select-mcp"

    @property
    def doc_title(self) -> str:
        return "/select-mcp \nSelect MCP servers to connect"

    @property
    def doc_description(self) -> str:
        return """Choose from a list of available MCP servers to connect for the current chat session. Use SPACE to select or deselect servers, and ENTER to confirm your selection. The selected MCP servers will be connected and their tools will be available for use during the chat."""
