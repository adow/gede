# coding=utf-8
#
# file_commands.py
# File-related commands
#

from typing import Optional

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.patch_stdout import patch_stdout
from rich.prompt import Prompt

from .base import CommandBase
from ..chatcore import (
    load_chats_files,
    load_chats_files_2,
)


class PublicChatFileCompleter(Completer):
    def get_completions(self, document, complete_event):
        # files = load_chats_files()
        files = load_chats_files_2()
        text = document.text.lower()
        for one in files:
            if text in one[1].lower():
                yield Completion(
                    text=one[1], start_position=-len(document.text), display=one[0]
                )


class PrivateChatFileCompleter(Completer):
    def get_completions(self, document, complete_event):
        files = load_chats_files(is_private=True)
        text = document.text.lower()
        for one in files:
            if text in one.lower():
                yield Completion(one, start_position=-len(document.text))


class SaveCommand(CommandBase):
    async def do_command_async(self) -> bool:
        if self.message == "/save":
            if self.context.current_chat.is_private:
                password = Prompt.ask(
                    "[bold dim]Input password for private chat[/bold dim]",
                    password=True,
                )
                password = password.strip()
                if not password:
                    self.console.print("Password cannot be empty.", style="warning")
                    return False
                self.context.current_chat.private_password = password

            await self.context.current_chat.geneate_title()

            # Need to generate filename before saving
            self.context.current_chat.generate_filename()
            filepath = self.context.current_chat.save()
            if filepath:
                self.console.print(
                    f"Chat saved successfully in {filepath}. Title: {self.context.current_chat.title}",
                    style="info",
                )
            else:
                self.console.print("Not saved", style="warning")
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/save\nPersist current chat to disk"

    @property
    def doc_description(self) -> str:
        return "Save the current chat to disk. Public chats auto-save with generated titles. Private chats require a password for encryption and must be manually saved. Files are stored in ~/.gede/chats/."

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/save"


class LoadChatCommand(CommandBase):
    async def pick_file(self) -> Optional[str]:
        completion = PublicChatFileCompleter()
        self.console.print(
            "Type to search for chat files, use arrow keys to select.", style="info"
        )
        try:
            with patch_stdout():
                filename: str = await self.context.prompt_session.prompt_async(
                    "Select Chat: ",
                    completer=completion,
                    complete_while_typing=True,
                    complete_in_thread=False,
                )
                # self.console.print("You selected: " + filename, style="info")
                return filename
        except KeyboardInterrupt:
            return None

    async def do_command_async(self) -> bool:
        from ..chatcore import ChatModel

        cmd = "/load-chat"
        if self.message.startswith(cmd):
            # filename = self.message[len(cmd) :].strip()
            filename = await self.pick_file()
            if not filename:
                self.console.print("Filename cannot be empty.", style="danger")
                return False
            chat = ChatModel.load_from_file(filename)
            if not chat:
                self.console.print(f"Load chat failed: {filename}", style="danger")
            else:
                self.context.current_chat = chat
                self.console.rule(f"LOAD CHAT: {chat.filename}")
                self.print_instruction()
                for message in chat.messages:
                    role = message.get("role")
                    content = message.get("content")
                    if role and content:
                        if role == "system":
                            continue
                        else:
                            role_name = "You" if role == "user" else "Assistant"
                            self.console.print(f"[bold]{role_name}: [/bold]{content}")
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/load-chat \nLoad a public chat from file"

    @property
    def doc_description(self) -> str:
        return """Load a public chat from a specified file. The chat file should be located in the 'chats/public' directory (default: ~/.gede/chats/public). """

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/load-chat"


class LoadPrivateChatCommand(CommandBase):
    async def pick_file(self) -> Optional[str]:
        completion = PrivateChatFileCompleter()
        self.console.print(
            "Type to search for chat files, use arrow keys to select.", style="info"
        )
        try:
            with patch_stdout():
                filename = await self.context.prompt_session.prompt_async(
                    "Select Private Chat: ",
                    completer=completion,
                    complete_while_typing=True,
                    complete_in_thread=False,
                )
                # self.console.print("You selected: " + filename, style="info")
                return filename
        except KeyboardInterrupt:
            return None

    async def do_command_async(self) -> bool:
        from ..chatcore import ChatModel

        cmd = "/load-private-chat"
        if self.message.startswith(cmd):
            # filename = self.message[len(cmd) :].strip()
            filename = await self.pick_file()
            if not filename:
                self.console.print("Filename cannot be empty.", style="danger")
                return False
            password = Prompt.ask(
                "[bold dim]Input password for private chat[/bold dim]", password=True
            )
            password = password.strip()
            if not password:
                self.console.print("Password cannot be empty.", style="warning")
            chat = ChatModel.load_from_file(
                filename, is_private=True, private_password=password
            )
            if not chat:
                self.console.print(f"Load chat failed: {filename}", style="danger")
            else:
                self.context.current_chat = chat
                self.console.rule(f"LOAD PRIVATE CHAT: {chat.filename}")
                self.print_instruction()
                for message in chat.messages:
                    role = message.get("role")
                    content = message.get("content")
                    if role and content:
                        if role == "system":
                            continue
                        else:
                            role_name = (
                                "You (Private)" if role == "user" else "Assistant"
                            )
                            self.console.print(f"[bold]{role_name}: [/bold]{content}\n")
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/load-private-chat \nLoad a private chat from file"

    @property
    def doc_description(self) -> str:
        return """Load a private chat from a specified file. You will be prompted to enter the password used to encrypt the chat messages. The chat file should be located in the 'chats/private' directory (default: ~/.gede/chats/private). """

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/load-private-chat"
