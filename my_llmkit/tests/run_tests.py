# coding=utf-8
#
# 测试在思考过程中使用工具
#
import os
import logging
from datetime import datetime
from openai import OpenAI
from openai.types import Reasoning
from pydantic import BaseModel
from typing import Any
import pytest
from my_llmkit.chat import UnifiedMessage, ToolFunctions
from my_llmkit.chat.base import LLMChatCompletion
from my_llmkit.chat.claude import ClaudeChatCompletion
from my_llmkit.chat.openai_compatible import OpenAICompatibleChatCompletion
from my_llmkit.mcp.mcp_config import MCPServersContext
from .tools import now_tool, get_weather_tool
from .conftest import (
    make_claude_client,
    make_openai_client,
    make_qwen_client,
    gpt_5_2_openrouter,
    gpt_5_2_zenmux,
    gemini_3_pro_openrouter,
    gemini_3_pro_zenmux,
    kimi_k2_thinking_moonshot,
    kimi_k2_thinking_turbo_moonshot,
    kimi_k2_5_moonshot,
    deepseek_reasoner_deepseek,
    doubao_seed_1_6,
    doubao_seed_2_pro,
    ernie_x_1_1,
    grok_4_1_fast_openrouter,
    grok_4_fast_openrouter,
    grok_4_openrouter,
    grok_code_fast_1_openrouter,
    grok_4_1_fast_zenmux,
    grok_4_fast_zenmux,
    grok_4_zenmux,
    grok_code_fast_1_zenmux,
    claude_4_5_sonnet_zenmux,
    claude_4_5_sonnet_openrouter,
    claude_4_5_haiku_zenmux,
    claude_4_5_haiku_openrouter,
    qwen_plus,
)
from .utils import RunStreamResult, run_stream

logger = logging.getLogger(__name__)


# reasoning stream tools
async def run_tool_test(client: LLMChatCompletion):
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
    pass


# stream response format
class TimeResult(BaseModel):
    local_time: str
    utc_time: str
    tz: str
    weekday: str


def check_stream_response_format_result(output_result: Any, content: str):
    now = datetime.now()
    year = str(now.year)
    month = str(now.month)
    day = str(now.day)
    content_buffer = content.strip()
    assert year in content_buffer and month in content_buffer and day in content_buffer
    output: Any = output_result
    logging.info("Output Result: %s", output)
    assert output is not None
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


async def run_qwen_json_mode_test(model: str):
    """
    千问在 JSON 模式下的输出格式测试，不会调用工具
    """
    client = make_qwen_client(model, reasoning=False)
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


# tests
@pytest.mark.asyncio
async def test_gpt_5_2_zemux():
    client = make_openai_client(*gpt_5_2_zenmux, reasoning=Reasoning(effort="medium"))
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_gpt_5_2_openrouter():
    client = make_openai_client(
        *gpt_5_2_openrouter, reasoning=Reasoning(effort="medium")
    )
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_gemini_3_pro_openrouter():
    client = make_openai_client(
        *gemini_3_pro_openrouter, reasoning=Reasoning(effort="medium")
    )
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_kimi_k2_thinking_moonshot():
    client = make_openai_client(
        *kimi_k2_thinking_moonshot, reasoning=Reasoning(effort="medium")
    )
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_kimi_k2_5_moonshot():
    client = make_openai_client(
        *kimi_k2_5_moonshot, reasoning=Reasoning(effort="medium")
    )
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_deepseek_reasoner_deepseek():
    client = make_openai_client(*deepseek_reasoner_deepseek)
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_doubao_seed_2_pro():
    client = make_openai_client(*doubao_seed_2_pro)
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_ernie_x_1_1():
    client = make_openai_client(*ernie_x_1_1)
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_grok_4_1_fast_openrouter():
    client = make_openai_client(*grok_4_1_fast_openrouter)
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_qwen_plus():
    client = make_qwen_client("qwen-plus", reasoning=True)
    await run_tool_test(client)


@pytest.mark.asyncio
async def test_claude_4_5_sonnet_zenmux():
    client = make_claude_client(*claude_4_5_sonnet_zenmux, reasoning=True)
    await run_tool_test(client)
