# coding=utf-8
#
# display.py - Message rendering and display management
#

from typing import Optional, Union
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.panel import Panel

# 导入事件类型
from my_llmkit.chat import (
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
        "tool": "dim",
        "usage": "dim",
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
                - ChatCompletionStreamContentEvent: 回答内容事件
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
        tool_description = (
            f"{event.function_name}: {event.function_args}"
            if event.function_args
            else event.function_name
        )
        self.console.print()
        self.console.print(
            Panel(f"🧰 {tool_description}", expand=False), style=self.STYLES["tool"]
        )
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
        self.console.print()
        usage = event.usage
        self.console.print(
            f"Input {usage.prompt_tokens}, Reasoning {usage.reasoning_tokens}, Output {usage.completion_tokens}, Total {usage.total_tokens}, Cached {usage.cached_tokens} ",
            style=self.STYLES["usage"],
        )
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


class NotificationRenderer:
    """处理系统提示、错误、通知等简单信息的显示

    与 MessageRenderer 职责分离，专注于简单的通知消息渲染，
    不涉及复杂的流式渲染和状态管理。
    """

    # 样式配置
    STYLES = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "dim": "dim",
    }

    def __init__(self, console: Console):
        """初始化通知渲染器

        Args:
            console: Rich Console实例，用于输出显示
        """
        self.console = console

    def info(self, message: str):
        """显示提示信息

        Args:
            message: 提示信息文本
        """
        self.console.print(f"ℹ️  {message}", style=self.STYLES["info"])

    def success(self, message: str):
        """显示成功信息

        Args:
            message: 成功信息文本
        """
        self.console.print(f"✓ {message}", style=self.STYLES["success"])

    def warning(self, message: str):
        """显示警告信息

        Args:
            message: 警告信息文本
        """
        self.console.print(f"⚠️  {message}", style=self.STYLES["warning"])

    def error(self, message: str):
        """显示错误信息

        Args:
            message: 错误信息文本
        """
        self.console.print(f"✗ {message}", style=self.STYLES["error"])

    def dim(self, message: str):
        """显示弱化的提示信息

        Args:
            message: 提示信息文本
        """
        self.console.print(message, style=self.STYLES["dim"])


# tests


def test_notification():
    """测试 NotificationRenderer 的各种显示效果"""
    console = Console()
    notification = NotificationRenderer(console)

    console.print("\n[bold cyan]测试 NotificationRenderer 各种消息类型：[/bold cyan]\n")

    # 测试各种消息类型
    notification.info("这是一条提示信息 - 用于显示一般性的提示")
    notification.success("这是一条成功信息 - 用于显示操作成功")
    notification.warning("这是一条警告信息 - 用于显示需要注意的事项")
    notification.error("这是一条错误信息 - 用于显示错误或失败")
    notification.dim("这是一条弱化的提示信息 - 用于显示次要提示")

    console.print("\n[bold cyan]实际使用场景示例：[/bold cyan]\n")

    # 实际使用场景示例
    notification.dim("Tip: Type '\\' for multi-line input, or just type your message.")
    notification.info("正在加载配置文件...")
    notification.success("配置文件加载成功")
    notification.warning("检测到旧版本配置，建议更新")
    notification.error("找不到模型路径对应的 Provider: openai/gpt-4")

    console.print()


def test_message_renderer():
    """测试 MessageRenderer 的显示效果"""
    import asyncio

    console = Console()
    renderer = MessageRenderer(console)

    console.print("\n[bold cyan]测试 MessageRenderer 流式渲染：[/bold cyan]\n")

    # 模拟流式事件
    async def simulate_stream():
        renderer.show_loading("Assistant is thinking")
        await asyncio.sleep(1)

        # 模拟推理内容
        reasoning_text = "让我分析一下这个问题..."
        for i in range(0, len(reasoning_text), 3):
            chunk = reasoning_text[i : i + 3]
            event = ChatCompletionStreamReasoningContentEvent(content=chunk)
            renderer.render_event(event)
            await asyncio.sleep(0.05)

        # 模拟回答内容
        answer_text = "这是一个示例回答，展示流式渲染的效果。"
        for i in range(0, len(answer_text), 2):
            chunk = answer_text[i : i + 2]
            event = ChatCompletionStreamContentEvent(content=chunk)
            renderer.render_event(event)
            await asyncio.sleep(0.05)

        renderer.render_event(
            ChatCompletionStreamToolCallStartEvent(function_name="now")
        )

        renderer.finish_message()

    asyncio.run(simulate_stream())
    console.print()


def tests():
    # 测试 NotificationRenderer
    test_notification()

    # 测试 MessageRenderer
    test_message_renderer()


if __name__ == "__main__":
    """运行所有测试"""
    tests()
