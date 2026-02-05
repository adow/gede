# coding=utf-8
#
# gede.py
#

import os
import json
import asyncio
import unicodedata
import argparse
from contextlib import AsyncExitStack

from agents.mcp import MCPServer
from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from agents import Agent, Runner, OpenAIChatCompletionsModel, Tool, set_tracing_disabled

from prompt_toolkit.shortcuts import CompleteStyle
from rich import print
from rich.panel import Panel
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from pyfiglet import figlet_format

from .top import logger, console, VERSION, gede_dir, gede_cache_dir
from . import config
from .commands import do_command, get_command_hints
from .chatcore2 import ChatModel
from .llm.tools.tools import get_tools
from .profiles import get_profile
from .context import Context
from .llm.providers2 import get_provider_from_model_path
from .display import MessageRenderer, NotificationRenderer


def clean_unicode_text(text):
    """Clean problematic Unicode characters from text"""
    # Remove surrogate pair characters
    text = "".join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))
    # Normalize Unicode characters
    text = unicodedata.normalize("NFC", text)
    return text


def input_history():
    filename = os.path.join(gede_cache_dir(), "input_history.txt")
    history = FileHistory(filename)
    return history


def create_prompt_style():
    """Create prompt style"""
    return Style.from_dict(
        {
            "username": "#87d7ff bold",
            "private": "#ffa500 bold",
            "symbol": "#00aaaa",
        }
    )


async def get_input_message(
    completer: WordCompleter,
    session: PromptSession,
    style: Style,
    is_private: bool = False,
):
    prompt_text_public = HTML("<username>You</username><symbol>: </symbol>")
    prompt_text_private = HTML("<private>You (Private)</private><symbol>: </symbol>")

    # Get input in single-line mode first
    with patch_stdout():
        message = await session.prompt_async(
            prompt_text_private if is_private else prompt_text_public,
            completer=completer,
            style=style,
            multiline=False,  # Default single-line mode
        )

    # If input is backslash, switch to multi-line mode
    if message.strip() == "\\":
        console.print("[dim]Multi-line mode. Press Esc+Enter to submit.[/dim]")
        with patch_stdout():
            message = await session.prompt_async(
                "... ",
                style=style,
                multiline=True,  # Multi-line mode
                prompt_continuation="... ",  # Continuation prompt
            )

    message = message.strip()
    message = clean_unicode_text(message)
    return message


async def chat(context: Context):
    input_message = context.current_chat.get_messages_to_talk()
    model_path = context.current_chat.model_path
    model_info = await context.current_chat.model

    # 根据 model_path 自动选择合适的 Provider
    provider = get_provider_from_model_path(model_path)
    if not provider:
        logger.error(f"Provider not found for model_path: {model_path}")
        context.notification.error(f"找不到模型路径对应的 Provider: {model_path}")
        return

    # 创建消息渲染器
    renderer = MessageRenderer(console)
    renderer.show_loading("Assistant is thinking")

    chat_client = provider.get_chat_client(model_info.model_id)
    runner = chat_client.run_stream(messages=input_message)

    full_answer_buffer = ""
    full_reasoning_buffer = ""

    async for event in runner.stream_event():
        # 渲染事件并收集内容
        content = renderer.render_event(event)

        # 收集完整的响应内容用于保存
        if event.type == "reasoning_content":
            full_reasoning_buffer += event.content
        elif event.type == "content":
            full_answer_buffer += event.content

    # 保存助手消息并完成渲染
    context.current_chat.append_assistant_message(full_answer_buffer)
    renderer.finish_message()


async def run_main():
    history = InMemoryHistory()
    completer = WordCompleter(get_command_hints(), ignore_case=True, sentence=True)
    style = create_prompt_style()
    session = PromptSession()

    current_chat = ChatModel()

    context = Context(
        current_chat=current_chat, console=console, prompt_session=session
    )

    context.notification.dim(
        "Tip: Type '\\' for multi-line input, or just type your message."
    )

    while True:
        message = await get_input_message(
            completer=completer, session=session, style=style
        )

        console.print()
        context.message = message
        should_continue = await do_command(context)
        # After command execution, do not continue
        if not should_continue:
            console.print()
            continue

        context.current_chat.append_user_message(message)
        # console.print(f"You entered: {message}")
        await chat(context)

        console.print()


if __name__ == "__main__":
    asyncio.run(run_main())
