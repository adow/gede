# coding=utf-8
#
# gede.py
#

import os
import json
import asyncio
import unicodedata
import argparse
from contextlib import AsyncExitStack

from agents.mcp import MCPServer
from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from agents import Agent, Runner, OpenAIChatCompletionsModel, Tool, set_tracing_disabled

from prompt_toolkit.shortcuts import CompleteStyle
from rich import print
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from pyfiglet import figlet_format

from .top import logger, console, VERSION, gede_dir
from . import config
from .commands import do_command, CommandConext, get_command_hints
from .chatcore import ChatModel
from .llm.model_info import read_model_info_dict
from .llm.providers import get_llm_model, get_model_path_value_list
from .llm.tools.tools import get_tools
from .profiles import get_profile


def clean_unicode_text(text):
    """Clean problematic Unicode characters from text"""
    # Remove surrogate pair characters
    text = "".join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))
    # Normalize Unicode characters
    text = unicodedata.normalize("NFC", text)
    return text


async def chat(context: CommandConext):
    # Start chat
    chat_data = context.current_chat
    logger.debug("model_settings: %s", chat_data.model_settings)
    tools: list[Tool] = []
    mcp_servers: list[MCPServer] = []
    if context.tools:
        if chat_data.model.model_supports and "Tools" in chat_data.model.model_supports:
            tools = get_tools(*context.tools)
            logger.debug(f"Using {len(tools)} tools")
            # MCP servers can only be used when tool calling is supported
            mcp_servers = context.mcp_manager.get_running_servers()
            logger.debug("Supports MCP servers")
        else:
            logger.debug("Tools are not supported")

    agent = Agent[CommandConext](
        name="gede",
        model=OpenAIChatCompletionsModel(
            model=chat_data.model.model.model_id,
            openai_client=chat_data.model.client,
        ),
        model_settings=chat_data.model_settings,
        tools=tools,
        mcp_servers=mcp_servers,
    )
    # Create loading prompt
    loading_text = Text("Assistant is thinking", style="bold deep_sky_blue1")
    spinner = Spinner("dots", text=loading_text)

    # prepare input messages
    input_messages = chat_data.get_messages_to_talk()
    # logger.debug('input_messages: "%s"', input_messages)
    result = Runner.run_streamed(agent, input_messages, max_turns=10, context=context)

    live = Live(spinner, console=console, refresh_per_second=4, transient=True)
    live.start()
    response_started = False

    # When using reasoning models, need to output thinking status
    is_thinking_message = False  # Whether it is a reasoning message
    is_thinking_hint_printed = False  # Whether thinking hint has been printed
    is_answer_hint_printed = False  # Whether answer hint has been printed

    # run and stream events
    try:
        async for event in result.stream_events():
            # logger
            if event.type == "run_item_stream_event":
                logger.debug(
                    "run_item_stream_event: %s, %s", event.name, event.item.type
                )
                if event.name == "reasoning_item_created":
                    # Update loading prompt to reasoning status
                    reasoning_text = Text(
                        "Assistant is reasoning", style="bold deep_sky_blue1"
                    )
                    spinner.text = reasoning_text
                    live.update(spinner)
                # elif event.name == 'tool_called':
                #     context.console.print('tool called', style='info')
                # elif event.name == 'tool_output':
                #     pass

                # Output tool call
                if event.item.type == "tool_call_item":
                    # Handle both Pydantic models and dicts
                    if isinstance(event.item.raw_item, dict):
                        logger.debug("tool_call_item: %s", json.dumps(event.item.raw_item, indent=2, ensure_ascii=False))
                        tool_dict = event.item.raw_item
                    else:
                        logger.debug("tool_call_item: %s", event.item.raw_item.model_dump_json(indent=2, exclude_none=True))
                        tool_dict = event.item.raw_item.model_dump()
                    
                    name = tool_dict.get("name") if isinstance(tool_dict, dict) else getattr(tool_dict, "name", None)
                    arguments = tool_dict.get("arguments", "") if isinstance(tool_dict, dict) else getattr(tool_dict, "arguments", "")

                    if name:
                        tool_description = f"{name}:{arguments}" if arguments else name
                        console.print()
                        console.print(Panel(f"🧰 {tool_description}", expand=False), style="warning")

                elif event.item.type == "tool_call_output_item":
                    # Output tool call result
                    logger.debug(
                        "tool_call_output_item: %s",
                        json.dumps(event.item.raw_item, indent=2, ensure_ascii=False),
                    )

            elif event.type == "agent_updated_stream_event":
                logger.debug("agent_updated_stream_event, %s", event.new_agent.name)
            if event.type == "raw_response_event":
                # Stop loading prompt and start normal output when first response is received
                if not response_started:
                    live.stop()
                    console.print(
                        "[bold deep_sky_blue1]Assistant: [/bold deep_sky_blue1]", end=""
                    )
                    response_started = True

                # thinking
                if isinstance(
                    event.data, ResponseReasoningTextDeltaEvent
                ) or isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                    if not is_thinking_hint_printed:
                        console.print("\n[Thinking]", end="\n", style="info")
                        is_thinking_message = True
                        is_thinking_hint_printed = True
                        is_answer_hint_printed = False

                    console.print(event.data.delta, end="", style="info")

                    # debug
                    # console.print("Thinking ", end="", style="info")
                    # console.print(event.data.delta)
                # answer
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    # console.print(event.data.model_dump(), end="")
                    # ResponseTextDeltaEvent may still be output during thinking
                    if not event.data.delta:
                        continue
                    if is_thinking_message and not is_answer_hint_printed:
                        console.print("\n[Answer]", end="\n")
                        is_answer_hint_printed = True

                    console.print(event.data.delta, end="")

                    # debug
                    # console.print("Answer ", end="", style="info")
                    # console.print(event.data.delta)

                else:  # Output other event content
                    # debug
                    # console.print(
                    #     f"<RawResponseEventDataType: {type(event.data)}>",
                    #     end="",
                    #     style="info",
                    # )
                    if logger.level <= 10:  # DEBUG level
                        console.print(
                            f"\n[DEBUG] Event type: {type(event.data)}", style="red"
                        )
                        if hasattr(event.data, "model_dump"):
                            try:
                                console.print(
                                    json.dumps(
                                        event.data.model_dump(),
                                        indent=2,
                                        ensure_ascii=False,
                                    ),
                                    style="red",
                                )
                            except:
                                console.print(str(event.data), style="red")
                        else:
                            console.print(str(event.data), style="red")
                        console.print()
                    pass

    finally:
        if not response_started:
            live.stop()

    # logger.debug(result.final_output)
    full_response_message = result.final_output or ""
    chat_data.append_assistant_message(full_response_message)
    console.print()

    for one_response in result.raw_responses:
        usage = one_response.usage
        console.print(
            f"Input {usage.input_tokens}, Output {usage.output_tokens} Total {usage.total_tokens}, Input Messages {len(input_messages)} ",
            style="info",
        )

    print()  # Add newline after response completes


def input_history():
    cache_dir = os.path.join(gede_dir(), "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    filename = os.path.join(cache_dir, "input_history.txt")
    history = FileHistory(filename)
    return history


def create_prompt_style():
    """Create prompt style"""
    return Style.from_dict(
        {
            "username": "#87d7ff bold",
            "private": "#ffa500 bold",
            "symbol": "#00aaaa",
        }
    )


async def get_input_message(
    context: CommandConext,
    history: History,
    completer: WordCompleter,
    session: PromptSession,
    style: Style,
):
    prompt_text_public = HTML("<username>You</username><symbol>: </symbol>")
    prompt_text_private = HTML("<private>You (Private)</private><symbol>: </symbol>")

    # Get input in single-line mode first
    with patch_stdout():
        message = await session.prompt_async(
            prompt_text_private
            if context.current_chat.is_private
            else prompt_text_public,
            completer=completer,
            style=style,
            multiline=False,  # Default single-line mode
        )

    # If input is backslash, switch to multi-line mode
    if message.strip() == "\\":
        console.print("[dim]Multi-line mode. Press Esc+Enter to submit.[/dim]")
        with patch_stdout():
            message = await session.prompt_async(
                "... ",
                style=style,
                multiline=True,  # Multi-line mode
                prompt_continuation="... ",  # Continuation prompt
            )

    message = message.strip()
    message = clean_unicode_text(message)
    return message


async def run_main(
    model_path=None,
    instruction=None,
    is_private=False,
    reasoning_effort=None,
    web_search=None,
    tools: list[str] = [],
    mcp: list[str] = [],
    trace: bool = False,
):
    if not trace:
        set_tracing_disabled(True)

    history = InMemoryHistory()
    completer = WordCompleter(get_command_hints(), ignore_case=True, sentence=True)
    style = create_prompt_style()

    current_chat = ChatModel(is_private=is_private)
    # cli arguments
    if model_path:
        current_chat.model_path = model_path
    if instruction:
        current_chat.set_instruction(instruction)
    if reasoning_effort:
        try:
            current_chat.set_model_reasoning(effort=reasoning_effort)
            console.print(f"Set reasoning effort to {reasoning_effort}", style="info")
        except Exception as e:
            console.print(f"Warning: {e}", style="warning")
    if web_search:
        try:
            current_chat.set_model_web_search(web_search)
            console.print(f"Set web search: {web_search}", style="info")
        except Exception as e:
            console.print(f"Warning: {e}", style="warning")

    session = PromptSession()
    context = CommandConext(
        console=console,
        message="",
        current_chat=current_chat,
        tools=tools,
        prompt_session=session,
    )

    if is_private:
        console.rule("NEW CHAT (Private)")
    else:
        console.rule("NEW CHAT")
    context.print_chat_info()
    console.print(f"[bold]System:[/bold] {current_chat.instruction}", style="dim")
    console.print()

    async with AsyncExitStack() as stack:
        context.stack = stack  # Assign stack to context

        for server in context.mcp_manager.server_items.values():
            # Only process auto_select servers
            if server.auto_select or server.server.name in mcp:
                await stack.enter_async_context(server.server)
                logger.debug(f'MCP server "{server.server.name}" connected')
                await server.select_server(context.console)

        console.print(
            "[dim]Tip: Type '\\' for multi-line input, or just type your message.[/dim]"
        )
        # run
        while True:
            message = await get_input_message(
                context=context,
                history=history,
                completer=completer,
                session=session,
                style=style,
            )

            if not message:
                console.print(
                    "Input cannot be empty. Please try again.", style="warning"
                )
                continue
            console.print()
            # Process command
            context.message = message
            should_continue = await do_command(context)
            # After command execution, do not continue
            if not should_continue:
                console.print()
                continue

            try:
                context.current_chat.append_user_message(context.message)
                await chat(context)
                # asyncio.run(chat(context))
            except UnicodeEncodeError as e:
                console.print()
                logger.exception(
                    f"Chat error: {e.__class__.__module__}.{e.__class__.__name__}\n{e}"
                )
                context.current_chat.messages.pop()  # Remove the last user message
                console.print()
                continue
            except Exception as e:
                console.print()
                logger.exception(
                    f"Chat error: {e.__class__.__module__}.{e.__class__.__name__}\n{e}"
                )
                console.print()
                continue


def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM.")
    # Parameter --profile
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Use specified configuration profile (default: default)",
    )
    # Parameter --log-level
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    # Parameter --model
    parser.add_argument(
        "--model",
        type=str,
        help="Specify default model (format: provider_id:model_id, e.g.: openai:gpt-4o)",
    )
    # Parameter --instruction
    parser.add_argument(
        "--instruction",
        type=str,
        help="Set system prompt (equivalent to executing /set-instruction command)",
    )
    # Parameter --private
    parser.add_argument(
        "--private",
        action="store_true",
        help="Start private session (equivalent to executing /new-private command)",
    )
    # Parameter --reasoning-effort
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high", "off", "auto"],
        help="Set reasoning mode (equivalent to executing /set-model-reasoning command)",
    )
    # Parameter --web-search
    parser.add_argument(
        "--web-search",
        type=str,
        choices=["on", "off", "auto"],
        help="Enable or disable model's built-in web search (equivalent to executing /set-model-web-search command)",
    )
    # Parameter --tools
    parser.add_argument(
        "--tools",
        type=str,
        help="Set enabled tools list (multiple tools separated by commas, e.g.: --tools web_search,now,read_page)",
    )
    # Parameter --trace
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable trace mode for analyzing detailed execution information of agent calls",
    )
    # Parameter --mcp
    parser.add_argument(
        "--mcp",
        type=str,
        help="Enable MCP servers, multiple servers separated by commas",
    )

    args = parser.parse_args()

    # Load model list from cache first
    get_model_path_value_list()

    # Read profiles and merge CLI arguments (CLI overrides profiles)
    profile = get_profile(args.profile)
    # Derive final parameter values
    final_model = args.model or profile.model
    final_instruction = args.instruction or profile.instruction
    final_is_private = args.private or (profile.private or False)
    final_reasoning_effort = args.reasoning_effort or profile.reasoning_effort
    final_web_search = args.web_search or profile.web_search
    # tools: merge CLI and profile
    args_tools = [t.strip() for t in args.tools.split(",")] if args.tools else []
    profile_tools = profile.tools or []
    final_tools = list(set(args_tools + profile_tools))
    # mcp: merge CLI and profile
    args_mcp = [m.strip() for m in args.mcp.split(",")] if args.mcp else []
    profile_mcp = profile.mcp or []
    final_mcp = list(set(args_mcp + profile_mcp))

    # trace: CLI takes priority
    final_trace = args.trace or (profile.trace or False)
    # Log level: CLI takes priority, otherwise use profile, otherwise INFO
    final_log_level = (args.log_level or (profile.log_level or "INFO")).upper()
    logger.setLevel(final_log_level)

    # Validate final model parameters (if any)
    if final_model:
        try:
            if ":" not in final_model:
                console.print(
                    "Model format is incorrect. Should be 'provider_id:model_id' format, e.g.: openai:gpt-4o",
                    style="danger",
                )
                return
            provider_id, model_id = final_model.split(":", 1)
            # Validate if model exists
            get_llm_model(provider_id, model_id)
        except Exception as e:
            console.print(
                f"Cannot find specified model '{final_model}'. Please check if provider and model ID are correct.{e}",
                style="danger",
            )
            return

    console.print(
        Panel(
            figlet_format("Gede", font="slant"),
            title=f"Version: {VERSION}",
            subtitle="Type /help for commands",
            expand=False,
        )
    )
    if final_trace:
        # Try to use arize-phoenix-otel if available, otherwise rely on OpenAI's default tracing
        try:
            # pip install arize-phoenix-otel
            # pip install openinference-instrumentation-openai-agents
            from phoenix.otel import register

            # Get Phoenix endpoint from environment variable
            phoenix_endpoint = os.getenv(
                "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"
            )
            # Configure the Phoenix tracer
            tracer_provider = register(
                project_name="gede",  # Default is 'default'
                endpoint=phoenix_endpoint,
                auto_instrument=True,  # Auto-instrument your app based on installed dependencies
            )
            console.print("[dim]Trace: Using Arize Phoenix[/dim]", style="info")
        except ImportError:
            # Arize Phoenix not installed, will use OpenAI's default tracing
            console.print(
                "[dim]Trace: Using OpenAI default tracing (set OPENAI_API_KEY to enable)[/dim]",
                style="info",
            )
    else:
        set_tracing_disabled(True)
    asyncio.run(
        run_main(
            model_path=final_model,
            instruction=final_instruction,
            is_private=final_is_private,
            reasoning_effort=final_reasoning_effort,
            web_search=final_web_search,
            tools=final_tools,
            mcp=final_mcp,
            trace=final_trace,
        )
    )


if __name__ == "__main__":
    main()
