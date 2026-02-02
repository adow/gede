# coding=utf-8
#
# 事件定义
#
from typing import Optional, Literal
from pydantic import BaseModel


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    model: Optional[str] = None


class ChatCompleteionStreamContentEvent:
    type: Literal["content"] = "content"
    content: str

    def __init__(self, content: str):
        self.content = content


class ChatCompletionStreamReasoningContentEvent:
    type: Literal["reasoning_content"] = "reasoning_content"
    content: str

    def __init__(self, content: str):
        self.content = content


class ChatCompletionStreamToolCallStartEvent:
    type: Literal["tool_call_start"] = "tool_call_start"
    function_name: str
    function_args: Optional[str] = None

    def __init__(self, function_name: str, function_args: Optional[str] = None):
        self.function_name = function_name
        self.function_args = function_args


class ChatCompletionStreamToolCallResultEvent:
    type: Literal["tool_call_result"] = "tool_call_result"
    function_name: str
    function_result: Optional[str] = None

    def __init__(self, function_name: str, function_result: Optional[str] = None):
        self.function_name = function_name
        self.function_result = function_result


class ChatCompletionStreamUsageEvent:
    type: Literal["usage"] = "usage"
    usage: Usage

    def __init__(self, usage: Usage):
        self.usage = usage
