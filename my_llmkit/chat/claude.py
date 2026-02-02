# coding=utf-8
#
# Claude 实现
#
import json
import logging
from typing import Any, AsyncIterator, Optional, Type, Union

import anthropic
from pydantic import BaseModel

from my_llmkit.chat.model_settings import ModelSettings

from ..mcp.mcp_client import MCPServerType
from .base import LLMChatCompletion
from .tools import ToolFunctions
from .types import (
    ImageContent,
    DocumentContent,
    TextContent,
    UnifiedChunk,
    UnifiedMessage,
    UnifiedResponse,
    UnifiedToolCall,
    UnifiedUsage,
)


class ClaudeChatCompletion(LLMChatCompletion):
    """Anthropic Claude 实现"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        model_settings: Optional[ModelSettings] = None,
    ):
        super().__init__(api_key, api_base, model, model_settings)
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key, base_url=self.api_base if self.api_base else None
        )

    def _convert_tools(self, tools: Optional[ToolFunctions]) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "name": tp.name,
                "description": tp.description,
                "input_schema": tp.parameters,
            }
            for tp in tools.tool_params
        ]

    async def _convert_mcp_tools(
        self, mcp_servers: dict[str, MCPServerType]
    ) -> list[dict]:
        result = []
        for server_name, server in mcp_servers.items():
            tools = await server.list_tools()
            for one_tool in tools:
                result.append(
                    {
                        "name": f"_mcp_{server_name}_{one_tool.name}",  # mcp 前缀避免冲突
                        "description": one_tool.description or "",
                        "input_schema": one_tool.inputSchema,
                    }
                )

        return result

    def _convert_messages(self, messages: list[UnifiedMessage]) -> list[dict[str, Any]]:
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                continue

            if msg.role == "user":
                content = []
                if msg.content:
                    if isinstance(msg.content, str):
                        content.append({"type": "text", "text": msg.content})
                    else:
                        for block in msg.content:
                            if isinstance(block, TextContent):
                                content.append({"type": "text", "text": block.text})
                            elif isinstance(block, ImageContent):
                                img_dict: dict[str, Any] = {"type": "image"}
                                # 简化的格式用于日志
                                image_url = block.image_url
                                if image_url.startswith("http"):
                                    img_dict["source"] = {
                                        "type": "url",
                                        "url": image_url,
                                    }
                                else:
                                    img_dict["source"] = {
                                        "type": "base64",
                                        "media_type": block.media_type,
                                        "data": image_url,
                                    }
                                content.append(img_dict)
                            elif isinstance(block, DocumentContent):
                                doc_dict: dict[str, Any] = {"type": "document"}
                                document_url = block.document_url
                                if document_url.startswith("http"):
                                    doc_dict["source"] = {
                                        "type": "url",
                                        "url": document_url,
                                    }
                                else:
                                    doc_dict["source"] = {
                                        "type": "base64",
                                        "media_type": block.media_type,
                                        "data": document_url,
                                    }
                                content.append(doc_dict)

                anthropic_messages.append({"role": "user", "content": content})

            elif msg.role == "assistant":
                content = []
                # Thinking
                if msg.reasoning_content:
                    signature = None
                    if msg.vendor_metadata:
                        signature = msg.vendor_metadata.get("thinking_signature")

                    if signature:
                        content.append(
                            {
                                "type": "thinking",
                                "thinking": msg.reasoning_content,
                                "signature": signature,
                            }
                        )

                # Text
                if msg.content:
                    content.append({"type": "text", "text": msg.content})

                # Tool Calls
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": json.loads(tc.arguments)
                                if tc.arguments
                                else {},
                            }
                        )

                anthropic_messages.append({"role": "assistant", "content": content})

            elif msg.role == "tool":
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }

                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    anthropic_messages[-1]["content"].append(tool_result_block)
                else:
                    anthropic_messages.append(
                        {"role": "user", "content": [tool_result_block]}
                    )

        return anthropic_messages

    async def _build_request_kwargs(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> dict[str, Any]:
        """构建 Anthropic API 请求参数，供流式和非流式共用"""
        # Extract system message
        system_prompt = next((m.content for m in messages if m.role == "system"), None)

        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        mcp_tools = await self._convert_mcp_tools(mcp_servers) if mcp_servers else []
        full_tools = anthropic_tools + mcp_tools

        # Prepare parameters
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.model_settings.max_tokens or 4096,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if anthropic_tools:
            kwargs["tools"] = full_tools

        if response_format:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                # 对于 API，直接传入 Pydantic 模型类型
                # SDK 会自动调用 transform_schema
                kwargs["output_format"] = response_format
                kwargs["betas"] = ["structured-outputs-2025-11-13"]
            else:
                logging.error(
                    "Anthropic Claude only supports Pydantic BaseModel as response_format."
                )

        # Handle thinking/reasoning settings
        if self.model_settings.extra_body:
            extra_body: Any = self.model_settings.extra_body
            if "thinking" in extra_body:
                kwargs["thinking"] = extra_body["thinking"]

        return kwargs

    def _convert_response(self, response) -> UnifiedResponse:
        """将 Anthropic Message 响应转换为 UnifiedResponse"""
        unified = UnifiedResponse()

        # 转换 usage
        if response.usage:
            unified.usage = UnifiedUsage(
                completion_tokens=response.usage.output_tokens,
                prompt_tokens=response.usage.input_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                model=response.model,
            )

        # 处理 content blocks
        content_text = ""
        reasoning_content = ""
        tool_calls: list[UnifiedToolCall] = []
        thinking_signature: Optional[str] = None

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking":
                reasoning_content += block.thinking
                if hasattr(block, "signature") and block.signature:
                    thinking_signature = block.signature
            elif block.type == "tool_use":
                tool_calls.append(
                    UnifiedToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input) if block.input else "{}",
                    )
                )

        if content_text:
            unified.content = content_text

        if reasoning_content:
            unified.reasoning_content = reasoning_content

        if tool_calls:
            unified.tool_calls = tool_calls

        if thinking_signature:
            unified.vendor_metadata = {"thinking_signature": thinking_signature}

        return unified

    async def acreate(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> UnifiedResponse:
        """发起非流式请求，返回 UnifiedResponse"""
        kwargs = await self._build_request_kwargs(
            messages=messages,
            tools=tools,
            mcp_servers=mcp_servers,
            response_format=response_format,
        )

        # 使用 beta API 以支持 structured outputs
        if "betas" in kwargs:
            betas = kwargs.pop("betas")
            response = await self.client.beta.messages.create(**kwargs, betas=betas)
        else:
            response = await self.client.messages.create(**kwargs)

        logging.debug(
            "Claude Response: %s\n",
            json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
        )

        return self._convert_response(response)

    async def acreate_stream(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> AsyncIterator[UnifiedChunk]:
        """发起流式请求并将 Anthropic 事件转换为 UnifiedChunk"""
        kwargs = await self._build_request_kwargs(
            messages=messages,
            tools=tools,
            mcp_servers=mcp_servers,
            response_format=response_format,
        )

        async def _generate():
            # Map index to tool_call_id for the current stream
            tool_calls_accumulator: dict[int, UnifiedToolCall] = {}

            async with self.client.beta.messages.stream(**kwargs) as stream:
                async for event in stream:
                    chunk = UnifiedChunk()

                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_id = event.content_block.id
                            tool_name = event.content_block.name

                            tool_calls_accumulator[event.index] = UnifiedToolCall(
                                id=tool_id, name=tool_name, arguments=""
                            )

                            # 返回当前累积的完整状态（深拷贝）
                            chunk.tool_calls = [
                                UnifiedToolCall(
                                    id=tc.id, name=tc.name, arguments=tc.arguments
                                )
                                for tc in tool_calls_accumulator.values()
                            ]
                        elif event.content_block.type == "thinking":
                            # Thinking start
                            pass

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            chunk.content = event.delta.text
                        elif event.delta.type == "thinking_delta":
                            chunk.reasoning_content = event.delta.thinking
                        elif event.delta.type == "input_json_delta":
                            if event.index in tool_calls_accumulator:
                                tool_calls_accumulator[
                                    event.index
                                ].arguments += event.delta.partial_json

                                # 返回当前累积的完整状态（深拷贝）
                                chunk.tool_calls = [
                                    UnifiedToolCall(
                                        id=tc.id,
                                        name=tc.name,
                                        arguments=tc.arguments,
                                    )
                                    for tc in tool_calls_accumulator.values()
                                ]

                    elif event.type == "content_block_stop":
                        # Check for signature in thinking block
                        if (
                            hasattr(event, "content_block")
                            and getattr(event.content_block, "type", "") == "thinking"
                        ):
                            signature = getattr(event.content_block, "signature", None)
                            if signature:
                                chunk.vendor_metadata = {
                                    "thinking_signature": signature
                                }

                    elif event.type == "message_delta":
                        if event.usage:
                            chunk.usage = UnifiedUsage(
                                completion_tokens=event.usage.output_tokens,
                                prompt_tokens=0,  # Usage info might be incomplete in delta
                                total_tokens=event.usage.output_tokens,
                            )

                    elif event.type == "message_start":
                        if event.message.usage:
                            chunk.usage = UnifiedUsage(
                                completion_tokens=event.message.usage.output_tokens,
                                prompt_tokens=0,  # Usage info might be incomplete in delta
                                total_tokens=event.message.usage.output_tokens,
                            )

                    yield chunk

        return _generate()
