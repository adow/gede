# coding=utf-8

from dataclasses import dataclass, field
from typing import Literal, Optional, TypeAlias, cast, Any

from openai import AsyncClient
from openai.types.shared import Reasoning, ReasoningEffort
from agents import ModelSettings

from ..top import logger

ReasoningEffortType: TypeAlias = ReasoningEffort | Literal["off", "auto"]
WebSearchType: TypeAlias = Literal["on", "off", "auto"]

gpt_reasoning_model_id_prefix = [
    "openai/o1",
    "openai/o3",
    "openai/gpt-o4",
    "openai/gpt-5",
    "openai/gpt-oss",
    "o1",
    "o3",
    "o4",
    "gpt-5",
    "gpt-oss",
]
grok_reasoning_model_id_prefix = [
    "grok-code",
    "x-ai/grok-code",
    "grok-4",
    "x-ai/grok-4",
]
claude_reasoning_model_id_prefix = [
    "claude-3-7-sonnet",
    "claude-sonnet-4",
    "claude-opus-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
]
gemini_reasoning_model_id_prefix = [
    "gemini-2.5",
    "google/gemini-2.5",
    "gemini-3",
    "google/gemini-3",
]


gpt_web_search_model_id_prefix = [
    "gpt-4o-search",
    "gpt-4o-mini-search",
    "openai/gpt-4o-search",
    "openai/gpt-4o-mini-search",
]
grok_web_search_model_id_prefix = ["grok-", "x-ai/grok-"]

gpt_tool_model_id_prefix = [
    "openai/o1",
    "openai/o3",
    "openai/gpt-o4",
    "openai/gpt-5",
    "openai/gpt-oss",
    "o1",
    "o3",
    "o4",
    "gpt-5",
    "gpt-oss",
]

grok_tool_model_id_prefix = [
    "grok-",
    "x-ai/grok-",
]

claude_tool_model_id_prefix = [
    "claude-3-7-sonnet",
    "claude-sonnet-4",
    "claude-opus-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
]

gemini_tool_model_id_prefix = [
    "gemini-2.5",
    "google/gemini-2.5",
    "gemini-2.0",
    "google/gemini-2.0",
    "gemini-3",
    "google/gemini-3",
]


@dataclass
class LLMModel:
    model_id: str
    name: str


@dataclass
class LLMProviderBase:
    provider_id: str
    name: str
    client: AsyncClient
    models: list[LLMModel] = field(default_factory=list)

    async def load_models_async(self):
        pass

    def load_models(self):
        pass

    # model settings

    def model_settings_for_model_id(self, model_id: str) -> ModelSettings:
        base_settings = ModelSettings(include_usage=True)
        return base_settings

    def set_model_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        """
        Set reasoning model parameters

        When implementing, input_settings should be used as the base, then override or add corresponding reasoning configuration based on model_id and reasoning_type, return the modified input_settings
        """
        # grok
        if any(
            model_id.startswith(prefix) for prefix in grok_reasoning_model_id_prefix
        ):
            return self._set_grok_model_reasoning(
                model_id, input_settings, reasoning_type
            )

        # openai
        if any(model_id.startswith(prefix) for prefix in gpt_reasoning_model_id_prefix):
            return self._set_openai_model_reasoning(
                model_id, input_settings, reasoning_type
            )

        # claude
        if any(
            model_id.startswith(prefix) for prefix in claude_reasoning_model_id_prefix
        ):
            return self._set_claude_mode_reasoning(
                model_id, input_settings, reasoning_type
            )
        # gemini
        if any(
            model_id.startswith(prefix) for prefix in gemini_reasoning_model_id_prefix
        ):
            return self._set_gemini_mode_reasoning(
                model_id, input_settings, reasoning_type
            )
        raise Exception(f"Model {model_id} does not support reasoning.")

    def _set_grok_model_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        if model_id.startswith("grok-4") or model_id.startswith("x-ai/grok-4"):
            # grok-4 does not support turning off reasoning
            logger.warning("grok-4 does not support setting reasoning effort.")
            return input_settings
        if reasoning_type in ["off", "auto"]:
            input_settings.reasoning = None
        elif reasoning_type in ["high"]:
            input_settings.reasoning = Reasoning(effort="high")
        else:
            input_settings.reasoning = Reasoning(effort="low")
        return input_settings

    def _set_openai_model_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        if reasoning_type in ["auto", "off"]:
            input_settings.reasoning = None
        else:
            effort = cast(ReasoningEffort, reasoning_type)
            input_settings.reasoning = Reasoning(effort=effort)
        return input_settings

    def _set_gemini_mode_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        extra_body: Any = input_settings.extra_body or {}
        if reasoning_type in ["auto", "off"]:
            if "google" in extra_body:
                del extra_body["google"]
        else:
            extra_body["google"] = {"thinking_config": {"include_thoughts": True}}
        input_settings.extra_body = extra_body
        return input_settings

    def _set_claude_mode_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        extra_body: Any = input_settings.extra_body or {}
        if reasoning_type == "auto":
            if "thinking" in extra_body:
                del extra_body["thinking"]
                input_settings.extra_body = extra_body
                return input_settings
        if reasoning_type == "off":
            extra_body["thinking"] = {"type": "disabled"}
            input_settings.extra_body = extra_body
            return input_settings
        budget_tokens = 2000
        if reasoning_type == "minimal":
            budget_tokens = 1000
        elif reasoning_type == "low":
            budget_tokens = 2000
        elif reasoning_type == "medium":
            budget_tokens = 5000
        elif reasoning_type == "high":
            budget_tokens = 10000
        extra_body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        input_settings.extra_body = extra_body
        return input_settings

    def set_model_web_search(
        self,
        model_id: str,
        input_settings: ModelSettings,
        web_search_type: WebSearchType = "auto",
    ):
        """
        Set web search model parameters

        When implementing, input_settings should be used as the base, then override or add corresponding web search configuration based on model_id and web_search_type, return the modified input_settings
        """
        # openai
        if any(
            model_id.startswith(prefix) for prefix in gpt_web_search_model_id_prefix
        ):
            input_settings.extra_args = {"web_search_options": {}}
            logger.warning("gpt-4o does not support web search settings.")
            return input_settings
        # grok
        if any(
            model_id.startswith(prefix) for prefix in grok_web_search_model_id_prefix
        ):
            extra_body: Any = input_settings.extra_body or {}
            search_parameters = {
                "mode": web_search_type,
                "return_citations": True,
            }
            extra_body["search_parameters"] = search_parameters
            input_settings.extra_body = extra_body
            return input_settings
        raise Exception(f"Model {model_id} does not support web search.")

    # supports

    def supports_reasoning_for_model_id(self, model_id: str) -> bool:
        # claude
        if any(
            model_id.startswith(prefix) for prefix in claude_reasoning_model_id_prefix
        ):
            return True

        # grok
        if any(
            model_id.startswith(prefix) for prefix in grok_reasoning_model_id_prefix
        ):
            return True

        # openai
        if any(model_id.startswith(prefix) for prefix in gpt_reasoning_model_id_prefix):
            return True

        # gemini
        if any(
            model_id.startswith(prefix) for prefix in gemini_reasoning_model_id_prefix
        ):
            return True

        return False

    def supports_web_search_for_model_id(self, model_id: str) -> bool:
        # openai
        if any(
            model_id.startswith(prefix) for prefix in gpt_web_search_model_id_prefix
        ):
            return True

        # grok
        if any(
            model_id.startswith(prefix) for prefix in grok_web_search_model_id_prefix
        ):
            return True

        return False

    def supports_tools_for_model_id(self, model_id: str) -> bool:
        # claude
        if any(model_id.startswith(prefix) for prefix in claude_tool_model_id_prefix):
            return True
        # grok
        if any(model_id.startswith(prefix) for prefix in grok_tool_model_id_prefix):
            return True
        # openai
        if any(
            model_id.startswith(prefix) and "research" not in model_id
            for prefix in gpt_tool_model_id_prefix
        ):
            return True
        # gemini
        if any(model_id.startswith(prefix) for prefix in gemini_tool_model_id_prefix):
            return True
        return False
