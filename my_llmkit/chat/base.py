# coding=utf-8
#
# 抽象基类
#
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Type, Union

from .model_settings import ModelSettings
from pydantic import BaseModel

from ..mcp.mcp_client import MCPServerType
from .events import Usage
from .runner import ChatCompletionResult, ChatCompletionStreamRunner
from .tools import ToolExecutor, ToolFunctions
from .types import UnifiedChunk, UnifiedMessage, UnifiedResponse


class LLMChatCompletion(ABC):
    """
    LLM 客户端抽象基类 - 使用统一类型，与任何 SDK 解耦。
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        model_settings: Optional[ModelSettings] = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.model_settings = model_settings or ModelSettings()

    @abstractmethod
    async def acreate_stream(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> AsyncIterator[UnifiedChunk]:
        """
        发起流式请求，返回统一格式的 UnifiedChunk 流。
        子类负责将厂商特定格式转换为 UnifiedChunk。
        """
        raise NotImplementedError

    @abstractmethod
    async def acreate(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> UnifiedResponse:
        """
        发起非流式请求，返回统一格式的 UnifiedResponse。
        子类负责将厂商特定格式转换为 UnifiedResponse。
        """
        raise NotImplementedError

    def run_stream(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        max_rounds: int = 10,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> ChatCompletionStreamRunner:
        return ChatCompletionStreamRunner(
            self,
            messages,
            tools=tools,
            mcp_servers=mcp_servers,
            max_rounds=max_rounds,
            response_format=response_format,
        )

    async def run(
        self,
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        max_rounds: int = 10,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ) -> ChatCompletionResult:
        """
        非流式多轮对话，直接返回完整结果。
        """
        inner_messages = messages.copy()
        usages: list[Usage] = []
        last_content: Optional[str] = None
        current_round = 0
        tool_executor = ToolExecutor(tools, mcp_servers)

        while current_round < max_rounds:
            logging.info("========== Round: %s =========", current_round)
            current_round += 1

            # 调用模型（非流式）
            response = await self.acreate(
                messages=inner_messages,
                tools=tools,
                mcp_servers=mcp_servers,
                response_format=response_format,
            )

            # 记录 usage
            if response.usage:
                usages.append(
                    Usage(
                        completion_tokens=response.usage.completion_tokens,
                        prompt_tokens=response.usage.prompt_tokens,
                        total_tokens=response.usage.total_tokens,
                        reasoning_tokens=response.usage.reasoning_tokens,
                        cached_tokens=response.usage.cached_tokens,
                        model=response.usage.model,
                    )
                )

            # 构建 assistant 消息
            assistant_message = UnifiedMessage(
                role="assistant",
                content=response.content if response.content else None,
                reasoning_content=response.reasoning_content
                if response.reasoning_content
                else None,
                tool_calls=response.tool_calls if response.tool_calls else None,
                vendor_metadata=response.vendor_metadata,
            )
            inner_messages.append(assistant_message)

            # 如果没有工具调用，结束循环
            if not response.tool_calls:
                last_content = response.content
                break

            # 执行工具调用
            for tool_call in response.tool_calls:
                tool_result, _ = await tool_executor.execute(tool_call)
                tool_message = UnifiedMessage(
                    role="tool",
                    content=tool_result,
                    tool_call_id=tool_call.id,
                )
                inner_messages.append(tool_message)

        return ChatCompletionResult(
            last_content=last_content,
            usages=usages,
            messages=inner_messages,
            response_format=response_format
            if isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
            else None,
        )
