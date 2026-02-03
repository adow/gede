# coding=utf-8
#
# display.py - Message rendering and display management
#

from typing import Optional, Union
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

# 导入事件类型
from my_llmkit.chat.events import (
    ChatCompletionStreamContentEvent,
    ChatCompletionStreamReasoningContentEvent,
    ChatCompletionStreamToolCallStartEvent,
    ChatCompletionStreamToolCallResultEvent,
    ChatCompletionStreamUsageEvent,
)

# 定义流式事件的联合类型
StreamEvent = Union[
    ChatCompletionStreamContentEvent,
    ChatCompletionStreamReasoningContentEvent,
    ChatCompletionStreamToolCallStartEvent,
    ChatCompletionStreamToolCallResultEvent,
    ChatCompletionStreamUsageEvent,
]


class MessageRenderer:
    """处理聊天消息的显示渲染

    负责管理加载状态、流式内容渲染等显示逻辑，
    将显示逻辑从业务逻辑中分离出来。
    """

    # 样式配置
    STYLES = {
        "assistant_label": "bold deep_sky_blue1",
        "reasoning": "dim",
        "reasoning_label": "dim",
        "content": "",
        "loading": "dim",
    }

    def __init__(self, console: Console):
        """初始化消息渲染器

        Args:
            console: Rich Console实例，用于输出显示
        """
        self.console = console
        self._live: Optional[Live] = None
        self._last_event_type = ""
        self._first_output = True

    def show_loading(self, text: str = "Assistant is thinking"):
        """显示加载状态

        Args:
            text: 加载提示文本
        """
        loading_text = Text(text, style=self.STYLES["loading"])
        spinner = Spinner("dots", text=loading_text)
        self._live = Live(
            spinner, console=self.console, refresh_per_second=4, transient=True
        )
        self._live.start()
        self._first_output = True

    def stop_loading(self):
        """停止加载状态显示"""
        if self._live and self._live.is_started:
            self._live.stop()

    def render_event(self, event: StreamEvent) -> Optional[str]:
        """渲染流式事件

        根据事件类型渲染相应的内容，并管理状态转换。

        Args:
            event: 流式事件对象，可以是以下类型之一：
                - ChatCompleteionStreamContentEvent: 回答内容事件
                - ChatCompletionStreamReasoningContentEvent: 推理内容事件
                - ChatCompletionStreamToolCallStartEvent: 工具调用开始事件
                - ChatCompletionStreamToolCallResultEvent: 工具调用结果事件
                - ChatCompletionStreamUsageEvent: 使用统计事件

        Returns:
            返回渲染的内容文本，如果没有内容则返回None
        """
        # 首次输出时停止加载动画并显示助手标签
        if self._first_output:
            self.stop_loading()
            self.console.print(
                f"[{self.STYLES['assistant_label']}]Assistant: [/{self.STYLES['assistant_label']}]",
                end="",
            )
            self._first_output = False

        # 检测事件类型是否变化
        new_event_type = self._last_event_type != event.type
        self._last_event_type = event.type

        # 根据事件类型分发渲染
        if event.type == "reasoning_content":
            return self._render_reasoning(event, new_event_type)
        elif event.type == "content":
            return self._render_content(event, new_event_type)
        elif event.type == "tool_call_start":
            return self._render_tool_call_start(event, new_event_type)
        elif event.type == "tool_call_result":
            return self._render_tool_call_result(event, new_event_type)
        elif event.type == "usage":
            return self._render_usage(event, new_event_type)

        return None

    def _render_reasoning(
        self, event: ChatCompletionStreamReasoningContentEvent, is_new_section: bool
    ) -> str:
        """渲染推理内容

        Args:
            event: 推理事件
            is_new_section: 是否是新的推理段落

        Returns:
            推理内容文本
        """
        if is_new_section:
            self.console.print()
            self.console.print("[Reasoining]", style=self.STYLES["reasoning_label"])

        self.console.print(event.content, end="", style=self.STYLES["reasoning"])
        return event.content

    def _render_content(
        self, event: ChatCompletionStreamContentEvent, is_new_section: bool
    ) -> str:
        """渲染回答内容

        Args:
            event: 内容事件
            is_new_section: 是否是新的内容段落

        Returns:
            回答内容文本
        """
        if is_new_section:
            self.console.print()

        self.console.print(event.content, end="", style=self.STYLES["content"])
        return event.content

    def _render_tool_call_start(
        self, event: ChatCompletionStreamToolCallStartEvent, is_new_section: bool
    ) -> Optional[str]:
        """渲染工具调用开始事件

        Args:
            event: 工具调用开始事件
            is_new_section: 是否是新的事件类型

        Returns:
            工具调用信息（如果有）
        """
        # 当前实现为占位符，可以根据需要扩展
        return None

    def _render_tool_call_result(
        self, event: ChatCompletionStreamToolCallResultEvent, is_new_section: bool
    ) -> Optional[str]:
        """渲染工具调用结果事件

        Args:
            event: 工具调用结果事件
            is_new_section: 是否是新的事件类型

        Returns:
            工具调用结果信息（如果有）
        """
        # 当前实现为占位符，可以根据需要扩展
        return None

    def _render_usage(
        self, event: ChatCompletionStreamUsageEvent, is_new_section: bool
    ) -> Optional[str]:
        """渲染使用统计事件

        Args:
            event: 使用统计事件
            is_new_section: 是否是新的事件类型

        Returns:
            使用统计信息（如果有）
        """
        # 当前实现为占位符，可以根据需要扩展
        return None

    def finish_message(self):
        """完成消息渲染，输出换行"""
        self.console.print()
        self._reset_state()

    def _reset_state(self):
        """重置渲染器状态"""
        self._last_event_type = ""
        self._first_output = True
        if self._live:
            self._live = None
