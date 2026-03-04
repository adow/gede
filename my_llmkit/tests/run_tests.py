# coding=utf-8
#
# run_tests.py
#
# 测试在思考过程中使用工具
#
import os
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import Any
from my_llmkit.chat import (
    UnifiedMessage,
    ToolFunctions,
    TextContent,
    DocumentContent,
    ImageContent,
)
from my_llmkit.chat.base import LLMChatCompletion
from my_llmkit.chat.claude import ClaudeChatCompletion
from my_llmkit.chat.openai_compatible import OpenAICompatibleChatCompletion
from my_llmkit.mcp.mcp_config import MCPServersContext
from .tools import now_tool, get_weather_tool

from .utils import RunStreamResult, run_stream

logger = logging.getLogger(__name__)


# reasoning tools
async def run_stream_tool_test(client: LLMChatCompletion):
    """
    流式输出(含思考模式)调用工具
    """
    # prompt = "现在几点？"
    prompt = "明天天气怎么样"
    messages: list[UnifiedMessage] = [UnifiedMessage(role="user", content=prompt)]
    result = client.run_stream(
        messages=messages,
        tools=ToolFunctions(now_tool, get_weather_tool),
    )
    stream_result = await run_stream(result)

    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    content_buffer = stream_result.content.strip()
    # assert year in content_buffer and month in content_buffer and day in content_buffer
    assert "6°C" in content_buffer


async def run_tool_test(client: LLMChatCompletion):
    """
    非流式输出(含思考模式)调用工具
    """
    prompt = "现在几点？"
    # prompt = "明天天气怎么样"
    messages: list[UnifiedMessage] = [UnifiedMessage(role="user", content=prompt)]
    result = await client.run(
        messages=messages,
        tools=ToolFunctions(now_tool, get_weather_tool),
    )
    print("content", result.last_content)
    print("usages", result.usages)
    print("messages", result.messages)
    now = datetime.now()
    year = str(now.year)
    month = str(now.month)
    day = str(now.day)
    content = result.last_content or ""
    content_buffer = content.strip()
    assert year in content_buffer and month in content_buffer and day in content_buffer
    # assert "6°C" in content_buffer


# stream response format
class TimeResult(BaseModel):
    local_time: str
    utc_time: str
    tz: str
    weekday: str


def check_stream_response_format_result(
    output_result: BaseModel | dict[str, Any] | None, content: str
):
    now = datetime.now()
    year = str(now.year)
    month = str(now.month)
    day = str(now.day)
    content_buffer = content.strip()
    assert year in content_buffer and month in content_buffer and day in content_buffer
    output: Any = output_result
    logging.info("Output Result: %s, %s", output, type(output))
    assert output is not None
    if isinstance(output_result, dict):
        assert (
            year in output.get("local_time", "")
            and month in output.get("local_time", "")
            and day in output.get("local_time", "")
        )
    elif isinstance(output_result, BaseModel):
        assert (
            year in output.local_time
            and month in output.local_time
            and day in output.local_time
        )


async def run_openai_json_schema_test(client: OpenAICompatibleChatCompletion):
    """
    使用 OpenAI 兼容接口，
    测试在思考过程中使用格式化输出
    """
    prompt = "现在几点？"
    # prompt = "明天天气怎么样"
    messages: list[UnifiedMessage] = [UnifiedMessage(role="user", content=prompt)]
    result = client.run_stream(
        messages=messages,
        tools=ToolFunctions(
            now_tool,
            # get_weather_tool
        ),
        response_format=TimeResult,
    )
    stream_result = await run_stream(result)

    check_stream_response_format_result(result.output_result, stream_result.content)


async def run_openai_json_mode_test(client: OpenAICompatibleChatCompletion):
    """
    使用 OpenAI 兼容接口，
    在思考模式中使用 JSON Mode 格式输出结果
    """
    prompt = """现在几点？
请使用以下 JSON 格式来回答：

``` json
{
"local_time": "本地时间",
"utc_time": "UTC时间",
"tz": "时区",
"weekday": "星期几"
}

```

    """
    messages: list[UnifiedMessage] = [UnifiedMessage(role="user", content=prompt)]
    result = client.run_stream(
        messages=messages,
        tools=ToolFunctions(now_tool, get_weather_tool),
        response_format={
            "type": "json_object",
        },
    )
    stream_result = await run_stream(result)
    check_stream_response_format_result(result.output_result, stream_result.content)


async def run_qwen_json_mode_test(client: OpenAICompatibleChatCompletion):
    """
    千问在 JSON 模式下的输出格式测试，不会调用工具
    """
    prompt = """现在几点？
请使用以下 JSON 格式来回答：

``` json
{
"local_time": "本地时间",
"utc_time": "UTC时间",
"tz": "时区",
"weekday": "星期几"
}

```

    """
    messages: list[UnifiedMessage] = [UnifiedMessage(role="user", content=prompt)]
    result = client.run_stream(
        messages=messages,
        tools=ToolFunctions(now_tool, get_weather_tool),
        response_format={
            "type": "json_object",
        },
    )
    stream_result = await run_stream(result)

    check_stream_response_format_result(result.output_result, stream_result.content)


async def run_claude_json_schema_test(client: ClaudeChatCompletion):
    """
    使用 Claude 接口，
    在思考过程中使用格式化输出
    """
    prompt = "现在几点？"
    # prompt = "明天天气怎么样"
    messages: list[UnifiedMessage] = [UnifiedMessage(role="user", content=prompt)]
    result = client.run_stream(
        messages=messages,
        tools=ToolFunctions(
            now_tool,
            # get_weather_tool
        ),
        response_format=TimeResult,
    )
    stream_result = await run_stream(result)

    check_stream_response_format_result(result.output_result, stream_result.content)


# vision pdf input


async def run_openai_pdf_file_input_test(
    client: OpenAICompatibleChatCompletion,
):
    """测试 OpenAI 的 文件输入"""

    # 使用测试 PDF 文件（需要提前准备一个测试 PDF）
    # 这里假设您有一个测试 PDF 文件，或者可以创建一个简单的 base64 示例
    prompt = "这个文档的主要内容是什么？"

    messages: list[UnifiedMessage] = [
        UnifiedMessage(
            role="user",
            content=[
                TextContent(text=prompt),
                DocumentContent.from_file(
                    # 从本地文件加载（需要替换为实际路径）
                    "/Users/reynoldqin/Downloads/planning-with-files.pdf"
                ),
            ],
        )
    ]

    result = client.run_stream(messages=messages)
    stream_result = await run_stream(result)
    content = stream_result.content.strip()

    # 验证返回内容不为空
    assert len(content) > 0
    logging.info(f"OpenAI Document Response: {content}")


async def run_claude_pdf_url_input_test(
    client: ClaudeChatCompletion,
):
    """测试 Claude 的 URL 文档输入"""
    prompt = "这个文档的主要内容是什么？请简要总结。"

    messages: list[UnifiedMessage] = [
        UnifiedMessage(
            role="user",
            content=[
                TextContent(text=prompt),
                # 使用 URL 方式（Claude 支持）
                DocumentContent.from_url(
                    "https://tds-us-east-1.slashusr.xyz/planning-with-files.pdf"
                ),
            ],
        )
    ]

    result = client.run_stream(messages=messages)
    stream_result = await run_stream(result)
    content = stream_result.content.strip()

    # 验证返回内容包含文档相关信息
    assert len(content) > 0
    logging.info(f"Claude Document (URL) Response: {content}")


async def run_cluade_pdf_file_input_test(client: ClaudeChatCompletion):
    """测试从本地文件加载文档（需要实际的 PDF 文件）"""

    messages: list[UnifiedMessage] = [
        UnifiedMessage(
            role="user",
            content=[
                TextContent(text="总结这个文档的主要内容"),
                # 从本地文件加载（需要替换为实际路径）
                DocumentContent.from_file(
                    "/Users/reynoldqin/Downloads/planning-with-files.pdf"
                ),
            ],
        )
    ]

    result = client.run_stream(messages=messages)
    stream_result = await run_stream(result)
    content = stream_result.content.strip()

    assert len(content) > 0
    logging.info(f"Document from file Response: {content}")


# vision image input


async def run_openai_image_input_url_test(
    client: OpenAICompatibleChatCompletion,
):
    """测试 OpenAI 图片输入"""
    prompt = "图片里有什么"
    messages: list[UnifiedMessage] = [
        UnifiedMessage(
            role="user",
            content=[
                TextContent(text=prompt),
                ImageContent(image_url="https://tds-us-east-1.slashusr.xyz/1.png"),
                # ImageContent.from_file("/Users/reynoldqin/Downloads/1.png"),
            ],
        )
    ]
    result = client.run_stream(
        messages=messages,
    )
    stream_result = await run_stream(result)
    content = stream_result.content.strip()
    assert "魔法" in content or "奇幻" in content or "场景" in content


async def run_openai_image_input_file_test(
    client: OpenAICompatibleChatCompletion,
):
    """测试 OpenAI 图片输入"""
    prompt = "图片里有什么"
    messages: list[UnifiedMessage] = [
        UnifiedMessage(
            role="user",
            content=[
                TextContent(text=prompt),
                ImageContent.from_file("/Users/reynoldqin/Downloads/1.png"),
            ],
        )
    ]
    result = client.run_stream(
        messages=messages,
    )
    stream_result = await run_stream(result)
    content = stream_result.content.strip()
    assert "魔法" in content or "奇幻" in content or "场景" in content


async def run_claude_image_input_test(client: ClaudeChatCompletion):
    """测试 Claude 图片输入"""
    prompt = "图片里有什么"
    messages: list[UnifiedMessage] = [
        UnifiedMessage(
            role="user",
            content=[
                TextContent(text=prompt),
                ImageContent(image_url="https://tds-us-east-1.slashusr.xyz/1.png"),
                # ImageContent.from_file("/Users/reynoldqin/Downloads/1.png"),
            ],
        )
    ]
    result = client.run_stream(
        messages=messages,
    )
    stream_result = await run_stream(result)
    content = stream_result.content.strip()
    assert "魔法" in content or "奇幻" in content or "场景" in content
