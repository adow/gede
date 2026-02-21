# coding=utf-8
#
# reasoning.py
# set model reasoning
#
#

import logging
from typing import Literal, TypeAlias, cast, Any
from openai.types.shared import Reasoning, ReasoningEffort

from my_llmkit.chat.model_settings import ModelSettings

logger = logging.getLogger(__name__)

ReasoningEffortType: TypeAlias = ReasoningEffort | Literal["off", "auto"]


def make_gpt_reasoning(
    model_settings: ModelSettings,
    reasoning_effort: ReasoningEffortType = "auto",
):
    if reasoning_effort in ["auto", "off"]:
        model_settings.reasoning = None
    else:
        effort = cast(ReasoningEffort, reasoning_effort)
        model_settings.reasoning = Reasoning(effort=effort)
    return model_settings


def make_grok_reasoning(
    model_id: str,
    model_settings: ModelSettings,
    reasoning_effort: ReasoningEffortType = "auto",
):
    if model_id not in ["grok-3-mini", "x-ai/grok-3-mini"]:
        return model_settings
    effort = cast(ReasoningEffort, reasoning_effort)
    model_settings.reasoning = Reasoning(effort=effort)
    return model_settings


def make_claude_reasoning(
    model_settings: ModelSettings, reasoning_effort: ReasoningEffortType = "auto"
):
    extra_body: Any = model_settings.extra_body or {}
    if reasoning_effort == "auto":
        if "thinking" in extra_body:
            del extra_body["thinking"]
            model_settings.extra_body = extra_body
            return model_settings
    if reasoning_effort == "off":
        extra_body["thinking"] = {"type": "disabled"}
        model_settings.extra_body = extra_body
        return model_settings
    budget_tokens = 2000
    if reasoning_effort == "minimal":
        budget_tokens = 1000
    elif reasoning_effort == "low":
        budget_tokens = 2000
    elif reasoning_effort == "medium":
        budget_tokens = 5000
    elif reasoning_effort == "high":
        budget_tokens = 10000
    extra_body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
    model_settings.extra_body = extra_body
    return model_settings


def make_gemini_reasnoing(
    model_settings: ModelSettings, reasoning_effort: ReasoningEffortType = "auto"
):
    # extra_body: Any = model_settings.extra_body or {}
    # if reasoning_effort in ["auto", "off"]:
    #     if "google" in extra_body:
    #         del extra_body["google"]
    # else:
    #     extra_body["google"] = {"thinking_config": {"include_thoughts": True}}
    # model_settings.extra_body = extra_body
    # return model_settings
    if reasoning_effort in ["auto", "off"]:
        model_settings.reasoning = None
    else:
        effort = cast(ReasoningEffort, reasoning_effort)
        model_settings.reasoning = Reasoning(effort=effort)
    return model_settings
