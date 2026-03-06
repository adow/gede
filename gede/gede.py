# coding=utf-8
#
# gede.py
#

import os
import json
import logging
import asyncio
import unicodedata
import argparse
from typing import Optional

from prompt_toolkit.shortcuts import CompleteStyle
from rich import print
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout

from . import config
from my_llmkit.chat.tools import ToolFunctions
from my_llmkit.mcp.mcp_config import get_mcp_servers

from .top import (
    console,
    gede_cache_dir,
    gede_mcp_config_path,
)
from .version import get_app_version
from .commands import do_command, get_command_hints
from .chatcore import ChatModel
from .profiles import get_profile
from .context import Context
from .llm.providers import get_provider_from_model_path, prepare_models
from .display import MessageRenderer, NotificationRenderer, render_startup_logo
from .llm.tools.tools import get_tools

logger = logging.getLogger(__name__)


def apply_app_log_level(level: str):
    """Apply log level for app loggers."""
    logging.getLogger("gede").setLevel(level)
    logging.getLogger("my_llmkit").setLevel(level)


def apply_sdk_log_level(level: str):
    """Apply log level for SDK/network loggers."""
    logging.getLogger("openai").setLevel(level)
    logging.getLogger("anthropic").setLevel(level)
    logging.getLogger("httpx").setLevel(level)
    logging.getLogger("httpcore").setLevel(level)


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
    continuation_prompt = "... "

    lines = []
    is_first_line = True

    while True:
        # 根据是否是第一行选择提示符
        if is_first_line:
            prompt = prompt_text_private if is_private else prompt_text_public
        else:
            prompt = continuation_prompt

        # 获取一行输入
        with patch_stdout():
            line = await session.prompt_async(
                prompt,
                completer=completer if is_first_line else None,  # 只在第一行启用补全
                style=style,
                multiline=False,
            )

        # 如果第一行单独输入 \，切换到真正的多行编辑器（用于粘贴多行文本）
        if is_first_line and line.strip() == "\\":
            console.print("[dim]Multi-line mode. Press Esc+Enter to submit.[/dim]")
            with patch_stdout():
                message = await session.prompt_async(
                    "... ",
                    style=style,
                    multiline=True,
                    prompt_continuation="... ",
                )
            return clean_unicode_text(message.strip())

        # 检查是否以 \ 结尾（续行）
        if line.rstrip().endswith("\\"):
            # 去掉行尾的 \，保存该行
            lines.append(line.rstrip()[:-1].rstrip())
            is_first_line = False
            continue
        else:
            # 不以 \ 结尾，添加该行并结束输入
            lines.append(line)
            break

    # 合并所有行
    message = "\n".join(lines).strip()
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
        context.notification_display.error(
            f"找不到模型路径对应的 Provider: {model_path}"
        )
        return

    # 创建消息渲染器
    renderer = MessageRenderer(console)
    renderer.show_loading("Assistant is thinking")
    # logger.debug(
    #     "model_settings: "
    #     + json.dumps(
    #         context.current_chat.model_settings.to_json_dict(), ensure_ascii=False
    #     )
    # )

    chat_client = provider.get_chat_client(
        model_info.model_id, context.current_chat.model_settings
    )

    tools: Optional[ToolFunctions] = None
    if context.tools:
        tools = get_tools(*context.tools)

    runner = chat_client.run_stream(
        messages=input_message,
        tools=tools,
        mcp_servers=context.mcp_servers if context.mcp_servers else None,
    )

    full_answer_buffer = ""
    full_reasoning_buffer = ""

    try:
        async for event in runner.stream_event():
            # 渲染事件并收集内容
            content = renderer.render_event(event)

            # 收集完整的响应内容用于保存
            if event.type == "reasoning_content":
                full_reasoning_buffer += event.content
            elif event.type == "content":
                full_answer_buffer += event.content
    except Exception as e:
        logger.error(f"Error during chat streaming: {e}")
        context.notification_display.error(f"Error during chat: {e}")
        renderer.finish_message()
        return

    # 保存助手消息并完成渲染
    context.current_chat.append_assistant_message(full_answer_buffer)
    renderer.finish_message()


async def run_main():
    parser = argparse.ArgumentParser(description="Chat with an LLM.")
    # Parameter --log-level
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    # Parameter --model
    parser.add_argument(
        "--model",
        type=str,
        help="Specify default model (format: provider_id:model_id, e.g.: openai:gpt-4o)",
    )
    # Parameter --instruction
    parser.add_argument(
        "--instruction",
        type=str,
        help="Set system prompt (equivalent to executing /set-instruction command)",
    )
    # Parameter --private
    parser.add_argument(
        "--private",
        action="store_true",
        help="Start private session (equivalent to executing /new-private command)",
    )
    # Parameter --tools
    parser.add_argument(
        "--tools",
        type=str,
        help="Set enabled tools list (multiple tools separated by commas, e.g.: --tools web_search,now,read_page)",
    )
    args = parser.parse_args()

    render_startup_logo(console=console, app_name="Gede", version=get_app_version())

    history = InMemoryHistory()
    completer = WordCompleter(get_command_hints(), ignore_case=True, sentence=True)
    style = create_prompt_style()
    session = PromptSession()

    await prepare_models()
    current_chat = ChatModel(is_private=args.private)

    # args
    if args.model:
        current_chat.model_path = args.model
    if args.instruction:
        current_chat.instruction = args.instruction
    if args.log_level:
        log_level = args.log_level.upper()
        apply_app_log_level(log_level)
        apply_sdk_log_level(log_level)
    else:
        # Keep app logs at INFO by default. SDK logs follow env vars
        # (OPENAI_LOG / ANTHROPIC_LOG) initialized during module import.
        apply_app_log_level("INFO")

    # 尝试加载 MCP 服务器配置
    mcp_config_path = gede_mcp_config_path()
    stack = None
    mcp_servers = {}

    if os.path.exists(mcp_config_path):
        notification = NotificationRenderer(console)
        try:
            logger.debug(f"正在加载 MCP 配置: {mcp_config_path}")
            mcp_servers, stack = await get_mcp_servers(mcp_config_path)
            logger.debug(f"成功加载 {len(mcp_servers)} 个 MCP 服务器")
            notification.info(f"成功加载 {len(mcp_servers)} 个 MCP 服务器")

            # 显式初始化所有服务器，确保所有输出都在用户输入前完成
            for name, server in mcp_servers.items():
                try:
                    tools = await server.list_tools()
                    logger.debug(f"服务器 {name} 提供 {len(tools)} 个工具")
                except Exception as e:
                    logger.warning(f"初始化服务器 {name} 的工具列表失败: {e}")

            # 短暂延迟，确保所有子进程的 stderr 输出完成
            # await asyncio.sleep(0.3)
        except Exception as e:
            logger.warning(f"加载 MCP 配置失败: {e}")
            notification.warning(f"MCP 配置加载失败: {e}")
            notification.dim("继续运行但无 MCP 工具支持")
    else:
        logger.debug(f"MCP 配置文件不存在: {mcp_config_path}")

    context = Context(
        current_chat=current_chat,
        console=console,
        prompt_session=session,
        mcp_servers=mcp_servers,
    )

    if args.tools:
        tools = [t.strip() for t in args.tools.split(",")] if args.tools else []
        context.tools = tools

    await context.print_chat_info()

    context.notification_display.dim(
        "Tip: Type \\ for multi-line input, or end a line with \\ to continue input on the next line."
    )

    try:
        while True:
            message = await get_input_message(
                completer=completer,
                session=session,
                style=style,
                is_private=context.current_chat.is_private,
            )

            if not message:
                context.notification_display.warning("Input cannot be empty.")
                continue

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
    finally:
        # 清理 MCP 服务器连接
        if stack:
            await stack.aclose()


if __name__ == "__main__":
    apply_app_log_level("INFO")
    asyncio.run(run_main())
