# coding=utf-8
#
# Runner 与 Result
#
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Type, Union, TYPE_CHECKING

from pydantic import BaseModel

from ..mcp.mcp_client import MCPServerType
from .events import Usage
from .processor import ChatCompletionStreamProcessor
from .tools import ToolFunctions
from .types import UnifiedMessage

if TYPE_CHECKING:
    from .base import LLMChatCompletion


@dataclass
class ChatCompletionResult:
    """非流式多轮对话的完整结果"""

    last_content: Optional[str] = None
    usages: list[Usage] = field(default_factory=list)
    messages: list[UnifiedMessage] = field(default_factory=list)
    response_format: Optional[Type[BaseModel]] = None

    @property
    def output_result(self) -> Optional[Union[BaseModel, dict]]:
        """解析结构化输出"""
        if not self.last_content:
            return None
        try:
            data = json.loads(self.last_content)
            if (
                self.response_format
                and isinstance(self.response_format, type)
                and issubclass(self.response_format, BaseModel)
            ):
                return self.response_format.model_validate(data)
            return data
        except Exception as e:
            logging.error(f"Error parsing result: {e}")
            return None


class ChatCompletionStreamRunner:
    def __init__(
        self,
        client: "LLMChatCompletion",
        messages: list[UnifiedMessage],
        tools: Optional[ToolFunctions] = None,
        mcp_servers: Optional[dict[str, MCPServerType]] = None,
        max_rounds: int = 10,
        response_format: Optional[Union[Type[BaseModel], dict]] = None,
    ):
        self.client = client
        self.messages = messages
        self.tools = tools
        self.mcp_servers = mcp_servers
        self.response_format = response_format
        self.max_rounds = max_rounds
        self.last_content = None
        self.usages: list[Usage] = []

    async def stream_event(self):
        inner_messages = self.messages.copy()
        current_round = 0

        while current_round < self.max_rounds:
            logging.info("========== Round: %s =========", current_round)
            current_round += 1

            # 调用模型
            response = await self.client.acreate_stream(
                messages=inner_messages,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                response_format=self.response_format,
            )

            processor = ChatCompletionStreamProcessor(
                response,
                messages=inner_messages,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
            )

            async for event in processor.stream_event():
                yield event

            if processor.usage:
                self.usages.append(processor.usage)

            if not processor.tool_calls:
                self.last_content = processor.content
                break

    @property
    def output_result(self) -> Optional[Union[BaseModel, dict]]:
        if not self.last_content:
            return None
        try:
            data = json.loads(self.last_content)
            if (
                self.response_format
                and isinstance(self.response_format, type)
                and issubclass(self.response_format, BaseModel)
            ):
                return self.response_format.model_validate(data)

            return data
        except Exception as e:
            logging.error(f"Error parsing result: {e}")
            return None
