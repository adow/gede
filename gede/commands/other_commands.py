# coding=utf-8
#
# other_commands.py
# Other miscellaneous commands
#

import os
import logging
from pathlib import Path
from typing import Optional

from rich.panel import Panel

from .base import CommandBase
from ..top import gede_dir, gede_prompts_dir
from ..version import get_app_version

logger = logging.getLogger(__name__)


class CleanupCommand(CommandBase):
    async def do_command_async(self) -> bool:
        from .common import cleanup_screen

        if self.message == "/cleanup":
            cleanup_screen()
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/cleanup \nClear the terminal screen"

    @property
    def doc_description(self) -> str:
        return """Clear the terminal screen to provide a clean workspace. This command works in standard terminals and is optimized for use within tmux sessions."""

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/cleanup"


class SelectPromptCommand(CommandBase):
    def load_prompts(self):
        prompts_dir = gede_prompts_dir()
        if not os.path.exists(prompts_dir):
            os.makedirs(prompts_dir)
            file_path = os.path.join(prompts_dir, "hello.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Hello.")

        prompts: list[str] = []
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".txt") or filename.endswith(".md"):
                filepath = os.path.join(prompts_dir, filename)
                with open(filepath, "r") as f:
                    content = f.read().strip()
                    if content:
                        prompts.append(content)
        logger.debug(f"Loaded {len(prompts)} prompts from {prompts_dir}")
        return prompts

    async def do_command_async(self) -> bool:
        import inquirer

        if self.message == "/select-prompt":
            prompts = self.load_prompts()
            question = [
                inquirer.List(
                    "Prompt",
                    message="Select a predefined prompt",
                    choices=prompts,
                    carousel=True,
                )
            ]
            answers = inquirer.prompt(question)
            if answers and "Prompt" in answers:
                selected_prompt = answers["Prompt"]
                self.context.message = selected_prompt
                self.console.print(f"[bold]You:[/bold] {selected_prompt}\n")
                # Return True to continue execution
                return True
            else:
                return False
        return True

    @property
    def doc_title(self) -> str:
        return "/select-prompt\nChoose from predefined prompts"

    @property
    def doc_description(self) -> str:
        return "Select a predefined prompt as your input message. Prompts are loaded from ~/.gede/prompts/ directory. The selected prompt is immediately sent to the AI as if you typed it yourself."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/select-prompt"


class HelpCommand(CommandBase):
    async def do_command_async(self) -> bool:
        cmd = "/help"
        if self.message.startswith(cmd):
            # Import here to avoid circular imports
            from . import get_command_class_list, get_command_class_list_async
            from .chat_commands import (
                NewPublicChatCommand,
                NewPrivateChatCommand,
                QuitCommand,
                ChatInfoCommand,
                CloneChatCommand,
            )
            from .instruction_commands import (
                SetInstructionCommand,
                GetInstructionCommand,
                SelectInstructionCommand,
            )
            from .model_commands import (
                SelectLLMCommand,
                SetMessageNumCommand,
                SetModelSettingsCommand,
                GetModelSettingsCommand,
                SetModelReasoningCommand,
            )
            from .file_commands import (
                SaveCommand,
                LoadChatCommand,
                LoadPrivateChatCommand,
                ExportCommand,
            )
            from .tool_commands import SelectToolsCommand, SelectMCPCommand
            from .other_commands import (
                CleanupCommand,
                SelectPromptCommand,
            )

            keywords = self.message[len(cmd) :].strip()

            # Group command classes by category
            categories = {
                "Chat Management": [
                    NewPublicChatCommand,
                    NewPrivateChatCommand,
                    QuitCommand,
                    ChatInfoCommand,
                    CloneChatCommand,
                ],
                "Instruction & Prompt Management": [
                    GetInstructionCommand,
                    SetInstructionCommand,
                    SelectInstructionCommand,
                    SelectPromptCommand,
                ],
                "Model Settings": [
                    SelectLLMCommand,
                    SetMessageNumCommand,
                    SetModelSettingsCommand,
                    GetModelSettingsCommand,
                    SetModelReasoningCommand,
                ],
                "File Operations": [
                    SaveCommand,
                    LoadChatCommand,
                    LoadPrivateChatCommand,
                    ExportCommand,
                ],
                "Tools & MCP": [
                    SelectToolsCommand,
                    # SelectMCPCommand,
                ],
                "Utility": [
                    CleanupCommand,
                    HelpCommand,
                ],
            }

            # Build output
            output = ""
            for category, command_classes in categories.items():
                category_commands = []
                for command_class in command_classes:
                    command_instance = command_class(self.context)

                    # Get all command hints for this command
                    hints = []
                    hint_val = command_instance.command_hint
                    if hint_val:
                        if isinstance(hint_val, str):
                            hints.append(hint_val)
                        elif isinstance(hint_val, tuple):
                            hints.extend(list(hint_val))

                    # Format command entries
                    for hint in hints:
                        doc_title = command_instance.doc_title.strip()
                        # Extract subtitle (remove first line)
                        subtitle = ""
                        if chr(10) in doc_title:
                            parts = doc_title.split(chr(10), 1)
                            if len(parts) > 1:
                                subtitle = parts[1].strip()

                        cmd_entry = f"[bold]{hint}[/bold]"
                        if subtitle:
                            cmd_entry += f" [dim]{subtitle}[/dim]"

                        description = command_instance.doc_description.strip()

                        # Filter by keyword if provided
                        if not keywords or (
                            keywords.lower() in hint.lower()
                            or keywords.lower() in description.lower()
                        ):
                            category_commands.append((cmd_entry, description))

                # Only show category if it has matching commands
                if category_commands:
                    output += f"\n[bold yellow]{category}[/bold yellow]\n\n"
                    for cmd_entry, description in category_commands:
                        output += f"  {cmd_entry}\n"
                        if description:
                            # Clean up description (replace newlines with spaces)
                            wrapped_desc = description.replace("\n", " ")
                            output += f"  [dim]{wrapped_desc}[/dim]\n"
                        output += "\n"

            # Remove extra newlines at start/end
            output = output.strip()

            # Add usage tip if no keyword search
            if not keywords:
                output += "\n\n[dim]Use '/help KEYWORD' to search for specific commands.[/dim]"

            self.context.info_display.command_help(
                title="[bold]Gede Command Help[/bold]",
                subtitle=f"[dim]Version: {get_app_version()}[/dim]",
                description=output,
            )

            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/help <KEYWORD>\nShow this help message"

    @property
    def doc_description(self) -> str:
        return """Display a list of all available commands along with their descriptions to assist users in navigating and utilizing the chat application effectively.

If KEYWORD is provided, only commands containing the keyword will be shown.
        """

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/help"
