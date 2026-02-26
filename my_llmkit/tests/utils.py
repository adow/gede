import logging
from pydantic import BaseModel
from typing import Optional, Any
from my_llmkit.chat import (
    ToolFunctions,
    ChatCompletionStreamRunner,
    Usage,
)


class RunStreamResult(BaseModel):
    content: str
    reasoning: Optional[str] = None
    usages: Optional[list[Usage]] = None
    tool_calls: list[dict[str, Any]]


async def run_stream(runner: ChatCompletionStreamRunner):
    content_buffer = ""
    reasoning_content_buffer = ""
    tool_calls = []
    usages: list[Usage] = []

    async for event in runner.stream_event():
        if event.type == "content":
            print(event.content, end="", flush=True)
            content_buffer += event.content
        elif event.type == "reasoning_content":
            print(event.content, end="", flush=True)
            reasoning_content_buffer += event.content
        elif event.type == "tool_call_start":
            print(f"\n[Tool Call Start] {event.function_name}\n")
        elif event.type == "tool_call_result":
            print(
                f"\n[Tool Call Result] {event.function_name}: {event.function_result}\n"
            )
            tool_calls.append(
                {"name": event.function_name, "result": event.function_result}
            )
        elif event.type == "usage":
            print(f"\n[Usage] {event.usage.model_dump_json()}\n")
            usages.append(event.usage)

    logging.info("full reasoning: %s", reasoning_content_buffer)
    logging.info("last_content: %s", content_buffer)
    logging.info("usages: %s", usages)
    logging.info("tool_calls: %s", tool_calls)

    return RunStreamResult(
        content=content_buffer.strip(),
        reasoning=reasoning_content_buffer.strip(),
        usages=usages,
        tool_calls=tool_calls,
    )


async def run_stream_and_collect(client, messages, tools=None, response_format=None):
    """
    Helper function to run the chat stream and collect results.
    """
    tool_functions = ToolFunctions(*tools) if tools else None

    result = client.run(
        messages=messages, tools=tool_functions, response_format=response_format
    )

    content_buffer = ""
    reasoning_content_buffer = ""
    usage_info = None
    tool_calls = []

    async for event in result.stream_event():
        if event.type == "content":
            print(event.content, end="", flush=True)
            content_buffer += event.content
        elif event.type == "reasoning_content":
            print(event.content, end="", flush=True)
            reasoning_content_buffer += event.content
        elif event.type == "tool_call_start":
            print(f"\n[Tool Call Start] {event.function_name}\n")
        elif event.type == "tool_call_result":
            print(
                f"\n[Tool Call Result] {event.function_name}: {event.function_result}\n"
            )
            tool_calls.append(
                {"name": event.function_name, "result": event.function_result}
            )
        elif event.type == "usage":
            print(f"\n[Usage] {event.usage.model_dump_json()}\n")
            usage_info = event.usage

    logging.info("reasoning: %s", reasoning_content_buffer)
    logging.info("answer: %s", content_buffer)

    return {
        "content": content_buffer,
        "reasoning": reasoning_content_buffer,
        "usage": usage_info,
        "tool_calls": tool_calls,
    }
