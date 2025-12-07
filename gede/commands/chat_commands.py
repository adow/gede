# coding=utf-8
#
# chat_commands.py
# Chat-related commands
#

import sys
from typing import Optional

from .base import CommandBase
from .common import cleanup_screen


class NewPublicChatCommand(CommandBase):
    def do_command(self) -> bool:
        if self.message == "/new":
            from ..chatcore import ChatModel

            self.context.current_chat = ChatModel(is_private=False)
            self.console.rule("NEW CHAT")
            self.context.print_chat_info()
            self.print_instruction()
            self.console.print()
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/new\nStart a new public chat"

    @property
    def doc_description(self) -> str:
        return "Messages are saved in plain text and automatically saved after each response."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/new"


class NewPrivateChatCommand(CommandBase):
    def do_command(self) -> bool:
        if self.message == "/new-private":
            from ..chatcore import ChatModel

            self.context.current_chat = ChatModel(is_private=True)
            self.console.rule("NEW CHAT (Private)")
            self.context.print_chat_info()
            self.print_instruction()
            self.console.print()
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/new-private\nStart a new private chat"

    @property
    def doc_description(self) -> str:
        return "Messages are encrypted with a password you set when saving. Private chats must be manually saved using the /save command."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/new-private"


class QuitCommand(CommandBase):
    def do_command(self) -> bool:
        if self.message == "/quit":
            self.console.print("Exiting chat...", style="dim")
            if self.context.current_chat.is_private:
                cleanup_screen()
            sys.exit(0)
            # return False
        return True

    @property
    def doc_title(self) -> str:
        return "/quit\nExit the chat application"

    @property
    def doc_description(self) -> str:
        return "Safely exit the chat application. Unsaved private chats will remain in memory but not persist after exit."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/quit"


class ChatInfoCommand(CommandBase):
    def do_command(self) -> bool:
        if self.message == "/chat-info":
            self.context.print_chat_info()
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/chat-info\nDisplay current chat details"

    @property
    def doc_description(self) -> str:
        return "Display comprehensive information about the current chat session, including chat ID, title, privacy status, model, instruction, number of messages in context, and total messages. Also shows enabled tools and MCP servers."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/chat-info"


class CloneChatCommand(CommandBase):
    def do_command(self) -> bool:
        from copy import deepcopy

        command = "/clone-chat"
        if self.message == command:
            old_chat = self.context.current_chat
            from ..chatcore import ChatModel

            clone_chat = ChatModel(is_private=old_chat.is_private)
            clone_chat.set_instruction(old_chat.instruction)
            clone_chat.model_path = old_chat.model_path
            clone_chat.message_num_in_context = old_chat.message_num_in_context
            clone_chat.user_model_settings = deepcopy(old_chat.user_model_settings)
            self.context.current_chat = clone_chat
            self.console.rule(
                f"NEW CHAT{' (Private)' if clone_chat.is_private else ''}"
            )
            self.context.print_chat_info()
            self.print_instruction()
            self.console.print()
            return False
        return True

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/clone-chat"

    @property
    def doc_title(self) -> str:
        return "/clone-chat \nClone the current chat to a new chat session"

    @property
    def doc_description(self) -> str:
        return "Create a new chat session that inherits all settings from the current chat (instruction, model, message context size, and model parameters). Useful for starting a fresh conversation with the same configuration."
