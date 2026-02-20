# coding=utf-8
#
#

import os
import logging
from typing import Optional
import httpx

from my_llmkit.models import get_model_info, ModelInfo
from my_llmkit.chat import LLMChatCompletion, OpenAICompatibleChatCompletion
from my_llmkit.chat.model_settings import ModelSettings

from .base import LLMProviderBase
from .reasoning import (
    ReasoningEffortType,
    make_gpt_reasoning,
    make_grok_reasoning,
    make_claude_reasoning,
    make_gemini_reasnoing,
)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
API_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class OpenRouterProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="openrouter",
            name="OpenRouter",
        )

    def get_chat_client(
        self, model_id: str, model_settings: Optional[ModelSettings] = None
    ) -> LLMChatCompletion:
        return OpenAICompatibleChatCompletion(
            api_key=API_KEY,
            model=model_id,
            api_base=API_BASE_URL,
            model_settings=model_settings,
        )

    @property
    def default_models(self) -> Optional[list[str]]:
        return [
            "openai/gpt-5.2",
            "x-ai/grok-4-fast",
            "google/gemini-3-pro-preview",
        ]

    async def load_models(self):
        if not API_KEY:
            logger.warning("OpenRouter API key is not set, skipping model loading.")
            return
        url = f"{API_BASE_URL}/models"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=10)
                if response.status_code != 200:
                    return None
                result = response.json()
                data = result.get("data", [])
                for one in data:
                    model_id = one.get("id", "").lower()
                    # logger.info(f"OpenRouter model: {model_id}, {name}")
                    model_path = f"{self.provider_id}:{model_id}"
                    model_info = await get_model_info(model_path)
                    if not model_info:
                        logger.warning(f"OpenRouter model info not found: {model_path}")
                        continue
                    self.models.append(model_info)
                logger.debug(f"OpenRouter models loaded: {len(self.models)}")
            except Exception as e:
                logger.error(f"OpenRouter load models error: {e}")

    def make_reasoning_setting(
        self, model_id: str, reasoning_effort: ReasoningEffortType
    ) -> ModelSettings:
        settings = ModelSettings()
        if model_id.startswith("x-ai"):
            settings = make_grok_reasoning(
                model_id=model_id,
                model_settings=settings,
                reasoning_effort=reasoning_effort,
            )
        elif model_id.startswith("anthropic"):
            settings = make_claude_reasoning(
                model_settings=settings, reasoning_effort=reasoning_effort
            )
        elif model_id.startswith("google"):
            make_gemini_reasnoing(
                model_settings=settings, reasoning_effort=reasoning_effort
            )
        elif model_id.startswith("openai"):
            settings = make_gpt_reasoning(
                model_settings=settings, reasoning_effort=reasoning_effort
            )
        return settings
