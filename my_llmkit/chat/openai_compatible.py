# coding=utf-8
#
# OpenAI 兼容实现
#
import json
import logging
from typing import Any, AsyncIterator, Optional, Type, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolUnionParam
from pydantic import BaseModel


from ..mcp.mcp_client import MCPServerType
from .base import LLMChatCompletion
from .tools import ToolFunctions
from .model_settings import ModelSettings
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


class OpenAICompatibleChatCompletion(LLMChatCompletion):
    """OpenAI 兼容的实现，负责将 OpenAI 格式转换为统一格式"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        model_settings: Optional[ModelSettings] = None,
    ):
        super().__init__(api_key, api_base, model, model_settings)
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)

    def _convert_messages(self, messages: list[UnifiedMessage]) -> list[dict[str, Any]]:
        """将统一消息格式转换为 OpenAI 格式"""
        input_messages = []
        for msg in messages:
            msg_dict: dict[str, Any] = {"role": msg.role}
            # 处理 content - 支持字符串或内容块数组
            if msg.content is not None:
                if isinstance(msg.content, str):
                    msg_dict["content"] = msg.content
                else:
                    # 内容块数组，转换为通用格式（用于日志）
                    content_list = []
                    for block in msg.content:
                        if isinstance(block, TextContent):
                            content_list.append({"type": "text", "text": block.text})
                        elif isinstance(block, ImageContent):
                            # 简化的格式用于日志
                            image_url = block.image_url
                            if not image_url.startswith("http"):
                                image_url = (
                                    f"data:{block.media_type};base64,{image_url}"
                                )
                            img_dict: dict[str, Any] = {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }
                            content_list.append(img_dict)
                        elif isinstance(block, DocumentContent):
                            # OpenAI 仅支持 base64 格式的文档输入
                            if block.document_url.startswith("http"):
                                raise ValueError(
                                    "OpenAI 不支持通过 URL 输入文档，请使用 base64 格式或本地文件。"
                                    f"\n尝试的 URL: {block.document_url}"
                                )
                            if not block.filename:
                                raise ValueError(
                                    "OpenAI 文档输入需要提供 filename 参数"
                                )
                            doc_dict: dict[str, Any] = {
                                "type": "file",
                                "file": {
                                    "filename": block.filename,
                                    "file_data": f"data:{block.media_type};base64,{block.document_url}",
                                },
                            }
                            content_list.append(doc_dict)

                    msg_dict["content"] = content_list

            if msg.tool_calls:
                tool_calls_list = []
                for tc in msg.tool_calls:
                    tc_dict = {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    # gemini 的 thought_signature
                    if tc.extra_content:
                        if isinstance(tc.extra_content, str):
                            tc_dict["extra_content"] = json.loads(tc.extra_content)
                        else:
                            tc_dict["extra_content"] = tc.extra_content
                    tool_calls_list.append(tc_dict)
                msg_dict["tool_calls"] = tool_calls_list

            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.reasoning_content:
                msg_dict["reasoning_content"] = msg.reasoning_content
            if msg.vendor_metadata:
                msg_dict.update(msg.vendor_metadata)

            input_messages.append(msg_dict)

        return input_messages

    def _convert_tools(
        self, tools: Optional[ToolFunctions]
    ) -> list[ChatCompletionToolUnionParam]:
        """将统一工具格式转换为 OpenAI 格式"""
        if not tools:
            return []
        result = []
        for tp in tools.tool_params:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tp.name,
                        "description": tp.description,
                        "parameters": tp.parameters,
                    },
                }
            )
        return result

    async def _convert_mcp_tools(
        self, mcp_servers: dict[str, MCPServerType]
    ) -> list[dict]:
        """将 MCP 工具服务器转换为 OpenAI 格式的工具定义"""
        result = []
        for server_name, server in mcp_servers.items():
            tools = await server.list_tools()
            for one_tool in tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"_mcp_{server_name}_{one_tool.name}",  # mcp 前缀避免冲突
                            "description": one_tool.description or "",
                            "parameters": one_tool.inputSchema,
                        },
                    }
                )

        return result

    async def acreate_stream(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> AsyncIterator[UnifiedChunk]:
        """发起流式请求并将 OpenAI chunk 转换为 UnifiedChunk"""
        kwargs = await self._build_request_kwargs(
            messages=messages,
            tools=tools,
            mcp_servers=mcp_servers,
            response_format=response_format,
            stream=True,
        )

        response = await self.client.chat.completions.create(**kwargs)

        async def _generate() -> AsyncIterator[UnifiedChunk]:
            # 在生成器内部维护工具调用累积状态（使用 index 作为 key）
            tool_calls_accumulator: dict[int, UnifiedToolCall] = {}

            async for chunk in response:
                logging.debug(
                    "OpenAI Chunk: %s\n",
                    json.dumps(chunk.model_dump(), indent=2, ensure_ascii=False),
                )

                unified = self._convert_chunk(chunk, tool_calls_accumulator)
                yield unified

        return _generate()

    def _convert_chunk(
        self, chunk, tool_calls_accumulator: dict[int, UnifiedToolCall]
    ) -> UnifiedChunk:
        """将 OpenAI ChatCompletionChunk 转换为 UnifiedChunk"""
        unified = UnifiedChunk()

        # 转换 usage
        if chunk.usage:
            reasoning_tokens = None
            cached_tokens = None
            if chunk.usage.completion_tokens_details:
                reasoning_tokens = (
                    chunk.usage.completion_tokens_details.reasoning_tokens
                )
            prompt_details = chunk.usage.model_dump().get("prompt_tokens_details")
            if prompt_details:
                cached_tokens = prompt_details.get("cached_tokens")

            unified.usage = UnifiedUsage(
                completion_tokens=chunk.usage.completion_tokens,
                prompt_tokens=chunk.usage.prompt_tokens,
                total_tokens=chunk.usage.total_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                model=chunk.model,
            )

        if not chunk.choices:
            return unified

        delta = chunk.choices[0].delta

        # content
        if delta.content:
            unified.content = delta.content

        # reasoning (gemini 风格)
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            unified.reasoning_content = reasoning

        # reasoning_content (DeepSeek/Kimi 风格/doubao)
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            unified.reasoning_content = reasoning_content

        # reasoning_details (OpenRouter/ZenMux 风格的 thinking_signature)
        reasoning_details = getattr(delta, "reasoning_details", None)
        if reasoning_details:
            unified.vendor_metadata = {"reasoning_details": reasoning_details}

        # tool_calls - 使用 index 累积，支持增量式工具调用
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index

                # 如果是新的工具调用，初始化
                if idx not in tool_calls_accumulator:
                    tool_calls_accumulator[idx] = UnifiedToolCall(
                        id="", name="", arguments=""
                    )

                # 累积 id（只有第一个 chunk 有）
                if tc.id:
                    tool_calls_accumulator[idx].id = tc.id

                # 累积 name（只有第一个 chunk 有）
                if tc.function and tc.function.name:
                    tool_calls_accumulator[idx].name = tc.function.name

                # 累积 arguments（可能分多个 chunk）
                if tc.function and tc.function.arguments:
                    tool_calls_accumulator[idx].arguments += tc.function.arguments

                # gemini 的 thought_signature
                extra_content = getattr(tc, "extra_content", None)
                if extra_content:
                    tool_calls_accumulator[idx].extra_content = extra_content

            # 返回当前累积的完整状态（深拷贝避免引用问题）
            unified.tool_calls = [
                UnifiedToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    extra_content=tc.extra_content,
                )
                for tc in tool_calls_accumulator.values()
            ]

        return unified

    def _convert_response(self, response) -> UnifiedResponse:
        """将 OpenAI ChatCompletion 响应转换为 UnifiedResponse"""
        unified = UnifiedResponse()

        # 转换 usage
        if response.usage:
            reasoning_tokens = None
            cached_tokens = None
            if response.usage.completion_tokens_details:
                reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
            prompt_details = response.usage.model_dump().get("prompt_tokens_details")
            if prompt_details:
                cached_tokens = prompt_details.get("cached_tokens")

            unified.usage = UnifiedUsage(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                model=response.model,
            )

        if not response.choices:
            return unified

        message = response.choices[0].message

        # content
        if message.content:
            unified.content = message.content

        # reasoning (gemini 风格)
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            unified.reasoning_content = reasoning

        # reasoning_content (DeepSeek/Kimi 风格/doubao)
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            unified.reasoning_content = reasoning_content

        # reasoning_details (OpenRouter/ZenMux 风格的 thinking_signature)
        reasoning_details = getattr(message, "reasoning_details", None)
        if reasoning_details:
            unified.vendor_metadata = {"reasoning_details": reasoning_details}

        # tool_calls
        if message.tool_calls:
            unified.tool_calls = []
            for tc in message.tool_calls:
                extra_content = getattr(tc, "extra_content", None)
                unified.tool_calls.append(
                    UnifiedToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        extra_content=extra_content,
                    )
                )

        return unified

    async def _build_request_kwargs(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
        stream: bool = True,
    ) -> dict[str, Any]:
        """构建 OpenAI API 请求参数，供流式和非流式共用"""
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools)
        mcp_tools = await self._convert_mcp_tools(mcp_servers) if mcp_servers else []
        full_tools = openai_tools + mcp_tools

        logging.debug(
            "OpenAI Input Messages: %s\n",
            json.dumps(openai_messages, indent=2, ensure_ascii=False),
        )
        logging.debug(
            "Full Tools: %s\n", json.dumps(full_tools, indent=2, ensure_ascii=False)
        )

        # OpenAI 的 response_format
        api_response_format: Any = None
        if response_format:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                return_type = response_format
                json_schema = return_type.model_json_schema()
                json_schema["additionalProperties"] = False  # 禁止额外属性
                api_response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": return_type.__name__,
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            else:
                api_response_format = response_format

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "stream": stream,
        }

        if openai_tools:
            kwargs["tools"] = full_tools

        if self.model_settings.reasoning:
            kwargs["reasoning_effort"] = self.model_settings.reasoning.effort

        if self.model_settings.extra_body is not None:
            kwargs["extra_body"] = self.model_settings.extra_body

        if self.model_settings.extra_query is not None:
            kwargs["extra_query"] = self.model_settings.extra_query

        if self.model_settings.extra_headers is not None:
            kwargs["extra_headers"] = self.model_settings.extra_headers

        if self.model_settings.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.model_settings.frequency_penalty

        if self.model_settings.max_tokens is not None:
            kwargs["max_completion_tokens"] = self.model_settings.max_tokens

        if self.model_settings.metadata is not None:
            kwargs["metadata"] = self.model_settings.metadata

        if self.model_settings.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = self.model_settings.parallel_tool_calls

        if self.model_settings.top_p is not None:
            kwargs["top_p"] = self.model_settings.top_p

        if self.model_settings.presence_penalty is not None:
            kwargs["presence_penalty"] = self.model_settings.presence_penalty

        if stream and self.model_settings.include_usage is not None:
            kwargs["stream_options"] = {
                "include_usage": self.model_settings.include_usage
            }

        if self.model_settings.verbosity is not None:
            kwargs["verbosity"] = self.model_settings.verbosity

        # TODO: OpenAI 中的 tool_choice 参数支持更多选项
        # https://platform.openai.com/docs/api-reference/chat/create#chat_create-tool_choice
        if self.model_settings.tool_choice is not None:
            tool_choice = self.model_settings.tool_choice
            if isinstance(tool_choice, str):
                kwargs["tool_choice"] = tool_choice
            else:
                # MCPToolChoice object - convert to OpenAI format
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice.name},
                }

        if api_response_format is not None:
            kwargs["response_format"] = api_response_format

        return kwargs

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
            stream=False,
        )

        response = await self.client.chat.completions.create(**kwargs)
        logging.debug(
            "OpenAI Response: %s\n",
            json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
        )

        return self._convert_response(response)
