# coding=utf-8
#
# instruction_commands.py
# Instruction-related commands
#

from typing import Optional

import inquirer
from prompt_toolkit.patch_stdout import patch_stdout
from rich.prompt import Prompt

from .base import CommandBase
from ..chatcore import loaded_instructions


class SetInstructionCommand(CommandBase):
    async def get_multiline_input(self):
        self.context.console.print(
            "[dim]Multi-line mode. Press Esc+Enter to submit.[/dim]"
        )
        with patch_stdout():
            message = await self.context.prompt_session.prompt_async(
                "... ",
                multiline=True,  # Multi-line mode
                prompt_continuation="... ",  # Continuation prompt
            )
            return message.strip()

    async def do_command_async(self) -> bool:
        cmd = "/set-instruction"
        if self.message.startswith(cmd):
            args = self.message[len(cmd) :].strip()
            if not args:
                args = Prompt.ask("[bold dim]Input Instruction[/bold dim]")
                args = args.strip()
                # Get multi-line input
                if args == "\\":
                    args = await self.get_multiline_input()
            if args:
                self.context.current_chat.set_instruction(args)
                self.print_instruction()
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/set-instruction <INSTRUCTION>\nSet system instruction for current chat"

    @property
    def doc_description(self) -> str:
        return "Set the system message (instruction) that guides the AI's behavior. If INSTRUCTION is not provided, you'll be prompted to enter it. Use '\\\\' as the instruction value to enter multi-line mode (press Esc+Enter to submit)."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/set-instruction"


class GetInstructionCommand(CommandBase):
    def do_command(self) -> bool:
        if self.message == "/get-instruction":
            self.print_instruction()
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/get-instruction\nDisplay current system instruction"

    @property
    def doc_description(self) -> str:
        return "View the current system instruction (system message) that is guiding the AI's behavior in this chat."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/get-instruction"


class SelectInstructionCommand(CommandBase):
    def do_command(self) -> bool:
        if self.message == "/select-instruction":
            question = [
                inquirer.List(
                    "Instruction",
                    message="Select Instruction",
                    choices=loaded_instructions,
                    carousel=True,
                )
            ]
            answers = inquirer.prompt(question)
            if answers and "Instruction" in answers:
                instruction = answers["Instruction"]
                self.context.current_chat.set_instruction(instruction)
                self.print_instruction()

            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/select-instruction\nChoose from predefined instructions"

    @property
    def doc_description(self) -> str:
        return "Select a system instruction from the predefined list. Instructions are loaded from ~/.gede/instructions/ directory. A new instruction will immediately affect the AI's behavior in subsequent responses."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/select-instruction"
