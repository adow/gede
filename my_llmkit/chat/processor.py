# coding=utf-8
#
# 流式处理器
#
import json
import logging
from dataclasses import asdict
from typing import Any, AsyncIterator, Optional

from ..mcp.mcp_client import MCPServerType
from .events import (
    Usage,
    ChatCompleteionStreamContentEvent,
    ChatCompletionStreamReasoningContentEvent,
    ChatCompletionStreamToolCallStartEvent,
    ChatCompletionStreamToolCallResultEvent,
    ChatCompletionStreamUsageEvent,
)
from .tools import ToolExecutor, ToolFunctions
from .types import UnifiedChunk, UnifiedMessage, UnifiedToolCall


class ChatCompletionStreamProcessor:
    """
    流式响应处理器 - 使用统一的 UnifiedChunk 格式。
    不再依赖任何特定 SDK 的类型。
    """

    def __init__(
        self,
        response: AsyncIterator[UnifiedChunk],
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
    ):
        self.response = response
        self.messages = messages
        self.tools = tools
        self.mcp_servers = mcp_servers
        self.tool_executor = ToolExecutor(tools, mcp_servers)
        self.content = ""
        self.reasoning_content = ""
        self.tool_calls: list[UnifiedToolCall] = []
        self.usage: Optional[Usage] = None
        self.vendor_metadata: Optional[dict[str, Any]] = None

    async def _process_assistant_message(self):
        content_buffer = ""
        reasoning_content_buffer = ""
        final_tool_calls: list[UnifiedToolCall] = []

        async for chunk in self.response:
            logging.debug(
                "Unified Chunk: %s\n",
                json.dumps(asdict(chunk), indent=2, ensure_ascii=False),
            )

            # 处理 usage
            if chunk.usage:
                usage = Usage(
                    completion_tokens=chunk.usage.completion_tokens,
                    prompt_tokens=chunk.usage.prompt_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    reasoning_tokens=chunk.usage.reasoning_tokens,
                    cached_tokens=chunk.usage.cached_tokens,
                    model=chunk.usage.model,
                )
                self.usage = usage
                yield ChatCompletionStreamUsageEvent(usage=usage)

            # 处理 reasoning_content
            if chunk.reasoning_content:
                reasoning_content_buffer += chunk.reasoning_content
                yield ChatCompletionStreamReasoningContentEvent(
                    content=chunk.reasoning_content
                )

            # 处理 content
            if chunk.content:
                content_buffer += chunk.content
                yield ChatCompleteionStreamContentEvent(content=chunk.content)

            # 工具调用 - 直接使用已累积的完整状态
            if chunk.tool_calls:
                final_tool_calls = chunk.tool_calls

            # 保存厂商元数据
            if chunk.vendor_metadata:
                self.vendor_metadata = chunk.vendor_metadata

        self.content = content_buffer
        self.reasoning_content = reasoning_content_buffer
        self.tool_calls = final_tool_calls

        # 确保 arguments 是有效 JSON
        for tc in self.tool_calls:
            if not tc.arguments or tc.arguments.strip() == "":
                tc.arguments = "{}"

        # 构建 assistant 消息
        assistant_message = UnifiedMessage(
            role="assistant",
            content=content_buffer if content_buffer else None,
            reasoning_content=reasoning_content_buffer
            if reasoning_content_buffer
            else None,
            tool_calls=self.tool_calls if self.tool_calls else None,
            vendor_metadata=self.vendor_metadata,
        )
        self.messages.append(assistant_message)

    async def _process_tool_call(self):
        if not self.tool_calls:
            return

        for tool_call in self.tool_calls:
            # 发送开始事件
            yield ChatCompletionStreamToolCallStartEvent(
                function_name=tool_call.name,
                function_args=tool_call.arguments,
            )

            # 执行工具
            tool_result, _ = await self.tool_executor.execute(tool_call)

            # 发送结果事件
            yield ChatCompletionStreamToolCallResultEvent(
                function_name=tool_call.name,
                function_result=tool_result,
            )

            # 添加工具结果消息
            tool_message = UnifiedMessage(
                role="tool",
                content=tool_result,
                tool_call_id=tool_call.id,
            )
            self.messages.append(tool_message)

    async def stream_event(self):
        async for event in self._process_assistant_message():
            yield event
        async for event in self._process_tool_call():
            yield event
