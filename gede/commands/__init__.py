# coding=utf-8
#
# __init__.py
# Commands package
#

from typing import Type, List

from .base import CommandBase, CommandConext
from .common import cleanup_screen

# Import all command classes
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
    SetModelWebSearchCommand,
)
from .file_commands import (
    SaveCommand,
    LoadChatCommand,
    LoadPrivateChatCommand,
)
from .tool_commands import (
    SelectToolsCommand,
    SelectMCPCommand,
)
from .other_commands import (
    CleanupCommand,
    SelectPromptCommand,
    HelpCommand,
    ExportCommand,
)


def get_command_class_list() -> list[Type[CommandBase]]:
    """Get all synchronous command classes"""
    return [
        NewPublicChatCommand,
        NewPrivateChatCommand,
        QuitCommand,
        GetInstructionCommand,
        SelectInstructionCommand,
        SelectLLMCommand,
        ChatInfoCommand,
        SetMessageNumCommand,
        CleanupCommand,
        HelpCommand,
        SelectPromptCommand,
        SetModelSettingsCommand,
        GetModelSettingsCommand,
        SetModelReasoningCommand,
        SetModelWebSearchCommand,
        SelectToolsCommand,
        CloneChatCommand,
        ExportCommand,
    ]


def get_command_class_list_async() -> list[Type[CommandBase]]:
    """Get all asynchronous command classes"""
    return [
        SaveCommand,
        LoadChatCommand,
        LoadPrivateChatCommand,
        SetInstructionCommand,
        SelectMCPCommand,
    ]


async def do_command(context: CommandConext) -> bool:
    """
    Execute command
    """

    for one_command in get_command_class_list():
        command_instance = one_command(context)
        if command_instance.do_command():
            continue
        else:
            return False
    for one_command in get_command_class_list_async():
        command_instance = one_command(context)
        if await command_instance.do_command_async():
            continue
        else:
            return False

    if context.message.startswith("/"):
        context.console.print("Unknown command:" + context.message, style="danger")
        return False
    return True


def get_command_hints() -> List[str]:
    """
    Get all command hints
    Returns: Command hints list
    """
    from rich.console import Console
    from prompt_toolkit import PromptSession

    hints = []
    command_list = get_command_class_list() + get_command_class_list_async()
    context = CommandConext(
        console=Console(),
        message="",
        current_chat=None,  # type: ignore
        prompt_session=PromptSession(),
    )
    for one_command in command_list:
        command_instance = one_command(context)
        hint = command_instance.command_hint
        if hint:
            if isinstance(hint, str):
                hints.append(hint)
            elif isinstance(hint, tuple):
                hints.extend(list(hint))
    return hints


__all__ = [
    "CommandBase",
    "CommandConext",
    "cleanup_screen",
    "do_command",
    "get_command_hints",
    "get_command_class_list",
    "get_command_class_list_async",
    # Export command classes for backward compatibility
    "NewPublicChatCommand",
    "NewPrivateChatCommand",
    "QuitCommand",
    "ChatInfoCommand",
    "CloneChatCommand",
    "SetInstructionCommand",
    "GetInstructionCommand",
    "SelectInstructionCommand",
    "SelectLLMCommand",
    "SetMessageNumCommand",
    "SetModelSettingsCommand",
    "GetModelSettingsCommand",
    "SetModelReasoningCommand",
    "SetModelWebSearchCommand",
    "SaveCommand",
    "LoadChatCommand",
    "LoadPrivateChatCommand",
    "SelectToolsCommand",
    "SelectMCPCommand",
    "CleanupCommand",
    "SelectPromptCommand",
    "HelpCommand",
    "ExportCommand",
]
