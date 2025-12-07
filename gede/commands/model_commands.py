# coding=utf-8
#
# model_commands.py
# Model-related commands
#

import json
from typing import Optional, cast, Literal, Any

from rich.panel import Panel
from rich.prompt import Prompt
from openai.types.shared import Reasoning, ReasoningEffort
from agents import ModelSettings

from .base import CommandBase
from ..llm.providers import get_model_path_value_list
from ..top import logger
from ..chatcore import WebSearchType


class SelectLLMCommand(CommandBase):
    def do_command(self) -> bool:
        import inquirer

        cmd = "/select-llm"
        if self.message.startswith(cmd):
            args = self.message[len(cmd) :].strip()
            no_cache = "--no-cache" in args
            path_list = get_model_path_value_list(no_cache=no_cache)

            provider = args.replace("--no-cache", "").strip()
            if provider:
                path_list = [one for one in path_list if provider in one[1]]
            if not path_list:
                return False
            question = [
                inquirer.List(
                    "LLM",
                    message="Select LLM Model",
                    choices=path_list,
                    default=self.context.current_chat.model_path,
                    carousel=True,
                )
            ]
            answers = inquirer.prompt(question)
            if answers and "LLM" in answers:
                model_path = answers["LLM"]
                self.context.current_chat.model_path = model_path
                # Reset user model settings when switching models
                self.context.current_chat.user_model_settings = ModelSettings()
                self.console.print(
                    f"Using {self.context.current_chat.model.provider_name}:{self.context.current_chat.model.model.name} now",
                    style="info",
                )
            else:
                self.console.print("No LLM model selected.", style="warning")
            return False

        return True

    @property
    def doc_title(self) -> str:
        return "/select-llm [PROVIDER]\nSwitch to a different AI model"

    @property
    def doc_description(self) -> str:
        return """Select an AI model from available providers (OpenAI, Anthropic, etc.). Use --no-cache to refresh the model list from providers. If PROVIDER is specified (e.g., 'openai'), only models from that provider will be shown. The new model will be used for subsequent responses."""

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/select-llm"


class SetMessageNumCommand(CommandBase):
    def do_command(self) -> bool:
        cmd = "/set-message-num"
        if self.message.startswith(cmd):
            args = self.message[len(cmd) :].strip()
            if not args.isdigit():
                self.console.print("Please input a valid number.", style="warning")
                return False
            num = int(args)
            self.context.current_chat.message_num_in_context = num
            self.console.print(f"Set message number in context to {num}", style="info")
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/set-message-num NUMBER\nControl chat history length"

    @property
    def doc_description(self) -> str:
        return """Limit how many recent messages the AI considers when generating responses. Set to 0 to include all messages in the conversation. Reducing this number can save tokens and improve response time."""

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/set-message-num"


class SetModelSettingsCommand(CommandBase):
    def do_command(self) -> bool:
        cmd = "/set-model-settings"
        if self.message.startswith(cmd):
            args = self.message[len(cmd) :].strip()
            try:
                (key, value) = args.split(" ", maxsplit=1)
                key = key.strip().lower()
                value = value.strip()
                if key == "temperature":
                    self.context.current_chat.user_model_settings.temperature = float(
                        value
                    )
                    return False
                if key == "top_p":
                    self.context.current_chat.user_model_settings.top_p = float(value)
                    return False
                if key == "frequency_penalty":
                    self.context.current_chat.user_model_settings.frequency_penalty = (
                        float(value)
                    )
                if key == "presence_penalty":
                    self.context.current_chat.user_model_settings.presence_penalty = (
                        float(value)
                    )
                    return False
                if key == "max_tokens":
                    self.context.current_chat.user_model_settings.max_tokens = int(
                        value
                    )
                    return False
                if key == "verbosity" and value in ["low", "medium", "high"]:
                    self.context.current_chat.user_model_settings.verbosity = cast(
                        Literal["low", "medium", "high"], value
                    )
                    return False
                if key == "metadata":
                    self.context.current_chat.user_model_settings.metadata = json.loads(
                        value
                    )
                    return False

                if key == "include_usage":
                    self.context.current_chat.user_model_settings.include_usage = (
                        value
                        in [
                            "true",
                            "1",
                            "yes",
                        ]
                    )
                    return False

                if key == "extra_query":
                    self.context.current_chat.user_model_settings.extra_query = (
                        json.loads(value)
                    )
                    return False

                if key == "extra_body":
                    self.context.current_chat.user_model_settings.extra_body = (
                        json.loads(value)
                    )
                    return False

                if key == "extra_headers":
                    self.context.current_chat.user_model_settings.extra_headers = (
                        json.loads(value)
                    )
                    return False

                if key == "extra_args":
                    self.context.current_chat.user_model_settings.extra_args = (
                        json.loads(value)
                    )
                    return False
                if key == "reasoning_effort" and value in [
                    "minimal",
                    "low",
                    "medium",
                    "high",
                ]:
                    value = cast(ReasoningEffort, value)
                    if self.context.current_chat.user_model_settings.reasoning is None:
                        self.context.current_chat.user_model_settings.reasoning = (
                            Reasoning(effort=value)
                        )
                    else:
                        self.context.current_chat.user_model_settings.reasoning.effort = value

                    return False
                if key == "reasoning_summary" and value in [
                    "auto",
                    "concise",
                    "detailed",
                ]:
                    value = cast(Literal["auto", "concise", "detailed"], value)
                    if self.context.current_chat.user_model_settings.reasoning is None:
                        self.context.current_chat.user_model_settings.reasoning = (
                            Reasoning(summary=value)
                        )
                    else:
                        self.context.current_chat.user_model_settings.reasoning.summary = value
                    return False

                self.console.print(
                    f"Unknown model settings key ({key})", style="warning"
                )
            except Exception as e:
                logger.exception("Set model settings error: %s", e)
                self.console.print(f"Set model settings error: {e}", style="danger")

            return False

        return True

    @property
    def doc_title(self) -> str:
        return "/set-model-settings KEY VALUE\nCustomize model parameters"

    @property
    def doc_description(self) -> str:
        return """Fine-tune model behavior by adjusting parameters like temperature, top_p, max_tokens, etc. This overrides default settings for the current chat only. Common parameters: temperature (0-2), top_p (0-1), max_tokens, frequency_penalty (-2 to 2), presence_penalty (-2 to 2). Also supports reasoning_effort (minimal/low/medium/high) and reasoning_summary for reasoning models."""

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/set-model-settings"


class GetModelSettingsCommand(CommandBase):
    def do_command(self) -> bool:
        if self.message == "/get-model-settings":
            settings = self.context.current_chat.model_settings.to_json_dict()
            self.console.print(
                Panel(
                    json.dumps(settings, indent=2, ensure_ascii=False),
                    title="Model Settings",
                ),
                style="info",
            )
            return False
        return True

    @property
    def doc_title(self) -> str:
        return "/get-model-settings \nShow current model settings for the current chat"

    @property
    def doc_description(self) -> str:
        return """Display the current model settings being used for the chat, including any user-specific overrides"""

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/get-model-settings"


class SetModelReasoningCommand(CommandBase):
    def do_command(self) -> bool:
        command = "/set-model-reasoning"
        if self.message.startswith(command):
            args = self.message[len(command) :].strip() or "off"
            args = args.lower()
            allow_levels = ["minimal", "low", "medium", "high", "off", "auto"]

            if args not in allow_levels:
                self.console.print(
                    f"Invalid reasoning effort level. Choose from {','.join(allow_levels)}.",
                    style="warning",
                )
                return False
            effort = cast(ReasoningEffort, args)
            try:
                self.context.current_chat.set_model_reasoning(effort=effort)
                self.console.print(f"Set reasoning effort to {effort}", style="info")
            except Exception as e:
                self.console.print(e, style="danger")
            return False
        return True

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/set-model-reasoning"

    @property
    def doc_title(self) -> str:
        return "/set-model-reasoning <LEVEL>\nControl reasoning depth (o1 models)"

    @property
    def doc_description(self) -> str:
        return """Enable reasoning mode for supported models (like o1). LEVEL controls thinking depth: minimal, low, medium, high, or auto. Use 'off' to disable. Deeper reasoning uses more tokens but produces more thorough answers."""


class SetModelWebSearchCommand(CommandBase):
    def do_command(self) -> bool:
        command = "/set-model-web-search"
        if self.message.startswith(command):
            args = self.message[len(command) :].strip() or "off"
            args = args.lower()
            allow_types = ["on", "off", "auto"]
            if args not in allow_types:
                self.console.print(
                    f"Invalid web search type. Choose from {','.join(allow_types)}.",
                    style="warning",
                )
                return False
            web_search_type = cast(WebSearchType, args)
            try:
                self.context.current_chat.set_model_web_search(web_search_type)
                self.console.print(f"Set web search:{web_search_type}", style="info")
            except Exception as e:
                self.console.print(e, style="danger")
            return False

        return True

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/set-model-web-search"

    @property
    def doc_title(self) -> str:
        return "/set-model-web-search <on|off|auto>\nToggle web search capability"

    @property
    def doc_description(self) -> str:
        return """Allow the AI to search the web for current information. 'on' enables web search, 'off' disables it, and 'auto' lets the model decide when to search. Web search helps with recent events and factual queries but increases response time."""
