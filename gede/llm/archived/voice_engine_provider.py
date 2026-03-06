# coding=utf-8
#
#  voice_engine_provider.py
#
#

import os
from typing import Optional, Any
from agents import ModelSettings
from openai import AsyncClient
from openai.types.shared import ReasoningEffort
from .llm_provider import LLMProviderBase, LLMModel, ReasoningEffortType

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_API_BASE = os.getenv("DOUBAO_BASE_URL", "")
DOUBAO_BOT_URL = os.getenv("DOUBAO_BOT_URL", "")


class VoiceEngineProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="voice_engine",
            name="Voice Engine",
            client=AsyncClient(
                api_key=DOUBAO_API_KEY, base_url=DOUBAO_API_BASE, timeout=60
            ),
            models=[
                LLMModel(model_id="doubao-1-5-pro-32k", name="Doubao-1.5-pro-32k"),
                LLMModel(model_id="doubao-1-5-pro-256k", name="Doubao-1.5-pro-256k"),
                LLMModel(model_id="doubao-seed-1-6", name="Doubao-Seed-1.6"),
                LLMModel(
                    model_id="doubao-seed-1-6-thinking",
                    name="Doubao-Seed-1.6-thinking",
                ),
            ],
        )

    # model_settings

    def model_settings_for_model_id(self, model_id: str):
        base_settings: ModelSettings = super().model_settings_for_model_id(model_id)

        settings = ModelSettings()
        if "doubao-1-5" in model_id:
            settings.max_tokens = 12000
        if "doubao-seed-1-6" in model_id:
            settings.max_tokens = 30000

        base_settings = base_settings.resolve(settings)

        return base_settings

    def reasoning_model_settings_for_model_id(
        self, model_id: str, effort: ReasoningEffort = "medium"
    ) -> Optional[ModelSettings]:
        settings = ModelSettings()
        settings.extra_body = {
            "thinking": {
                # "type": "disabled",  # Do not use deep thinking capability
                "type": "enabled",  # Use deep thinking capability
                # "type": "auto",  # Model judges whether to use deep thinking capability
            }
        }
        return settings

    def set_model_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        extra_body: Any = input_settings.extra_body or {}
        if model_id != "doubao-seed-1-6":
            raise Exception("model does not support reasoning: " + model_id)
        if reasoning_type == "auto":
            extra_body["thinking"] = {"type": "auto"}
        elif reasoning_type == "off":
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {"type": "enabled"}
        input_settings.extra_body = extra_body
        return input_settings

    # supports

    def supports_reasoning_for_model_id(self, model_id: str) -> bool:
        if model_id.startswith("doubao-seed-1-6"):
            return True
        return False

    def supports_web_search_for_model_id(self, model_id: str) -> bool:
        return False

    def supports_tools_for_model_id(self, model_id: str) -> bool:
        return True
