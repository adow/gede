# coding=utf-8
#
#

import os
import logging
from typing import Optional, Any
from openai import AsyncClient

from my_llmkit.models import get_model_info
from my_llmkit.chat.model_settings import ModelSettings
from my_llmkit.chat import LLMChatCompletion, OpenAICompatibleChatCompletion

from .base import LLMProviderBase
from .reasoning import ReasoningEffortType, make_gemini_reasnoing

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY", "")
API_BASE_URL = os.getenv(
    "GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)


class GoogleProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="google",
            name="Google",
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
        return ["gemini-2.5-pro", "gemini-3.1-pro-preview"]

    async def load_models(self):
        if not API_KEY:
            logger.warning("Google API key is not set, skipping model loading.")
            return

        try:
            client = AsyncClient(api_key=API_KEY, base_url=API_BASE_URL, timeout=120)
            models_list = await client.models.list()
            for one in models_list.data:
                model_id = (one.id or "").lower()
                if not model_id:
                    continue
                # if not model_id.startswith("gemini"):
                #     continue
                if model_id.startswith("models/"):
                    model_id = model_id[len("models/") :]
                model_path = f"{self.provider_id}:{model_id}"
                model_info = await get_model_info(model_path)
                if not model_info:
                    logger.warning(f"Google model info not found: {model_path}")
                    continue
                self.models.append(model_info)
            logger.debug(f"Google models loaded: {len(self.models)}")
        except Exception as e:
            logger.error(f"Google load models error: {e}")

    def make_reasoning_setting(
        self, model_id: str, reasoning_effort: ReasoningEffortType
    ) -> ModelSettings:
        settings = ModelSettings()
        # if not model_id.startswith("gemini"):
        #     return settings

        # extra_body: Any = settings.extra_body or {}
        # if reasoning_effort in [None, "auto", "off"]:
        #     settings.extra_body = extra_body
        #     return settings

        # budget_tokens = 800
        # if reasoning_effort == "minimal":
        #     budget_tokens = 256
        # elif reasoning_effort == "low":
        #     budget_tokens = 512
        # elif reasoning_effort == "medium":
        #     budget_tokens = 800
        # elif reasoning_effort == "high":
        #     budget_tokens = 2048

        # extra_body["google"] = {
        #     "thinking_config": {
        #         "thinking_budget": budget_tokens,
        #         "include_thoughts": True,
        #     }
        # }
        # settings.extra_body = extra_body
        # return settings
        return make_gemini_reasnoing(
            model_settings=settings, reasoning_effort=reasoning_effort
        )
