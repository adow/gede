# coding=utf-8
#
# model_commands.py
# Model-related commands
#

import json
import logging
from typing import Optional, cast, Literal, Any

from rich.panel import Panel
from rich.prompt import Prompt
from openai.types.shared import Reasoning, ReasoningEffort

from .base import CommandBase
from ..llm.providers2.providers import (
    get_provider_from_model_path,
    MODEL_DATA,
    get_model_path_value_list,
    PROVIDERS,
    ProviderCache,
    ModelCache,
    save_models_to_file,
    PATH_VALUE_LIST,
)
from ..chatcore import WebSearchType
from my_llmkit.chat.model_settings import ModelSettings

logger = logging.getLogger(__name__)


class SelectLLMCommand(CommandBase):
    async def do_command_async(self) -> bool:
        import inquirer

        cmd = "/select-llm"
        if self.message.startswith(cmd):
            args = self.message[len(cmd) :].strip()
            path_list = get_model_path_value_list()

            provider = args.replace("--no-cache", "").strip()
            if provider:
                path_list = [one for one in path_list if provider in one[1]]
            if not path_list:
                self.context.notification_display.warning("No LLM models available.")
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
                self.context.notification_display.info(f"Using {model_path} now")
            else:
                self.context.notification_display.warning("No LLM model selected.")
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


class ManageProviderModelsCommand(CommandBase):
    async def do_command_async(self) -> bool:
        import inquirer

        cmd = "/model-manage"
        if not self.message.startswith(cmd):
            return True

        provider_prefix = self.message[len(cmd) :].strip()
        if not provider_prefix:
            available_providers = ", ".join(
                [provider.provider_id for provider in PROVIDERS]
            )
            self.context.notification_display.warning(
                f"Please input provider, usage: /model-manage [PROVIDER]. Available providers: {available_providers}"
            )
            return False

        matched_providers = [
            provider
            for provider in PROVIDERS
            if provider.provider_id.startswith(provider_prefix)
        ]
        if not matched_providers:
            self.context.notification_display.warning(
                f"Provider not found with prefix: {provider_prefix}"
            )
            return False

        # Prefix conflict strategy: use the first matched provider by PROVIDERS order
        provider = matched_providers[0]
        provider.models = []
        await provider.load_models()
        if not provider.models:
            self.context.notification_display.warning(
                f"No available models found for provider: {provider.provider_id}"
            )
            return False

        provider_cache = next(
            (one for one in MODEL_DATA if one.provider_id == provider.provider_id), None
        )
        default_selected = (
            [model.model_id for model in provider_cache.models] if provider_cache else []
        )

        choices: list[tuple[str, str]] = []
        model_name_dict: dict[str, str] = {}
        for model in provider.models:
            model_id = model.model_id
            model_name = model.model_name or model_id
            model_name_dict[model_id] = model_name
            choices.append((model_name, model_id))

        answers = inquirer.prompt(
            [
                inquirer.Checkbox(
                    "models",
                    message=f"Select models to enable for {provider.provider_id} (use SPACE to select/deselect, ENTER to confirm)",
                    choices=choices,
                    default=default_selected,
                )
            ]
        )
        if not answers or "models" not in answers:
            self.context.notification_display.warning("Model selection canceled.")
            return False

        selected_model_ids: list[str] = answers["models"]
        selected_models = [
            ModelCache(
                model_id=model_id,
                name=model_name_dict.get(model_id, model_id),
                model_path=f"{provider.provider_id}:{model_id}",
            )
            for model_id in selected_model_ids
        ]

        if provider_cache:
            provider_cache.models = selected_models
        else:
            MODEL_DATA.append(
                ProviderCache(
                    provider_id=provider.provider_id,
                    name=provider.name,
                    models=selected_models,
                )
            )

        PATH_VALUE_LIST.clear()
        save_models_to_file()

        total_enabled = sum(len(one_provider.models) for one_provider in MODEL_DATA)
        self.context.notification_display.info(
            f"{provider.provider_id} enabled {len(selected_model_ids)} models"
        )
        self.context.notification_display.info(f"一共启用了 {total_enabled} 个模型")
        return False

    @property
    def doc_title(self) -> str:
        return "/model-manage [PROVIDER]\nManage enabled models for one provider"

    @property
    def doc_description(self) -> str:
        return """Manage enabled models for a provider with prefix matching. PROVIDER is required and matched against provider_id by prefix. This command loads all available provider models, shows a multi-select checkbox list, and persists selected models to local enabled-model cache."""

    @property
    def command_hint(self) -> Optional[str | tuple[str, ...]]:
        return "/model-manage"


class SetMessageNumCommand(CommandBase):
    async def do_command_async(self) -> bool:
        cmd = "/set-message-num"
        if self.message.startswith(cmd):
            args = self.message[len(cmd) :].strip()
            if not args.isdigit():
                self.context.notification_display.warning(
                    "Please input a valid number."
                )
                return False
            num = int(args)
            self.context.current_chat.message_num_in_context = num
            self.context.notification_display.info(f"Set message num to {num}")
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
    async def do_command_async(self) -> bool:
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

                self.context.notification_display.warning(
                    f"Unknown model settings key ({key})"
                )
            except Exception as e:
                logger.exception("Set model settings error: %s", e)
                self.context.notification_display.error(
                    f"Set model settings error: {e}"
                )

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
            self.context.print_model_settings()
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
    async def do_command_async(self) -> bool:
        command = "/set-model-reasoning"
        if self.message.startswith(command):
            args = self.message[len(command) :].strip() or "off"
            args = args.lower()
            allow_levels = ["minimal", "low", "medium", "high", "off", "auto"]

            if args not in allow_levels:
                self.context.notification_display.warning(
                    f"Invalid reasoning effort level. Choose from {','.join(allow_levels)}."
                )
                return False
            effort = cast(ReasoningEffort, args)
            try:
                # TODO: set model reasoning
                # self.context.current_chat.set_model_reasoning(effort=effort)
                self.context.notification_display.info(
                    f"Set reasoning effort to {effort}"
                )
            except Exception as e:
                self.context.notification_display.error(
                    f"Set reasoning effort error: {e}"
                )
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
