# coding=utf-8
#
# chat 包对外门面，保持 my_llmkit.chat 导入路径不变
#
from dotenv import load_dotenv

load_dotenv()

from .types import (
    UnifiedToolCall,
    UnifiedUsage,
    UnifiedChunk,
    UnifiedResponse,
    ContentBlock,
    TextContent,
    ImageContent,
    DocumentContent,
    UnifiedMessage,
    UnifiedToolParam,
)
from .events import (
    Usage,
    ChatCompleteionStreamContentEvent,
    ChatCompletionStreamReasoningContentEvent,
    ChatCompletionStreamToolCallStartEvent,
    ChatCompletionStreamToolCallResultEvent,
    ChatCompletionStreamUsageEvent,
)
from .tools import ToolFunction, ToolFunctions, ToolExecutor, make_mcp_tool_name
from .processor import ChatCompletionStreamProcessor
from .runner import ChatCompletionResult, ChatCompletionStreamRunner
from .base import LLMChatCompletion
from .openai_compatible import OpenAICompatibleChatCompletion
from .claude import ClaudeChatCompletion

__all__ = [
    "UnifiedToolCall",
    "UnifiedUsage",
    "UnifiedChunk",
    "UnifiedResponse",
    "ContentBlock",
    "TextContent",
    "ImageContent",
    "DocumentContent",
    "UnifiedMessage",
    "UnifiedToolParam",
    "Usage",
    "ChatCompleteionStreamContentEvent",
    "ChatCompletionStreamReasoningContentEvent",
    "ChatCompletionStreamToolCallStartEvent",
    "ChatCompletionStreamToolCallResultEvent",
    "ChatCompletionStreamUsageEvent",
    "ToolFunction",
    "ToolFunctions",
    "ToolExecutor",
    "make_mcp_tool_name",
    "ChatCompletionStreamProcessor",
    "ChatCompletionResult",
    "ChatCompletionStreamRunner",
    "LLMChatCompletion",
    "OpenAICompatibleChatCompletion",
    "ClaudeChatCompletion",
]
