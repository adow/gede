# coding=utf-8
#
#

import os
import logging
from typing import Optional
from openai import AsyncClient

from my_llmkit.models import get_model_info
from my_llmkit.chat.model_settings import ModelSettings
from my_llmkit.chat import LLMChatCompletion, OpenAICompatibleChatCompletion

from .base import LLMProviderBase
from .reasoning import ReasoningEffortType

logger = logging.getLogger(__name__)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
API_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


class AlibabaProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="alibaba-cn",
            name="Alibaba(China)",
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
        return ["qwen3.5-plus"]

    async def load_models(self):
        if not API_KEY:
            logger.warning("Alibaba API key is not set, skipping model loading.")
            return

        try:
            client = AsyncClient(api_key=API_KEY, base_url=API_BASE_URL, timeout=120)
            models_list = await client.models.list()
            for one in models_list.data:
                model_id = (one.id or "").lower()
                if not model_id:
                    continue
                if not (model_id.startswith("qwen") or model_id.startswith("qwq")):
                    continue
                model_path = f"{self.provider_id}:{model_id}"
                model_info = await get_model_info(model_path)
                if not model_info:
                    logger.warning(f"Alibaba model info not found: {model_path}")
                    continue
                self.models.append(model_info)
            logger.debug(f"Alibaba models loaded: {len(self.models)}")
        except Exception as e:
            logger.error(f"Alibaba load models error: {e}")

    def make_reasoning_setting(
        self, model_id: str, reasoning_effort: ReasoningEffortType
    ) -> ModelSettings:
        settings = ModelSettings()
        if not model_id.startswith("qwen3"):
            return settings

        if reasoning_effort in [None, "auto"]:
            return settings

        if reasoning_effort == "off":
            settings.extra_body = {"enable_thinking": False}
            return settings

        budget_tokens = 2000
        if reasoning_effort == "minimal":
            budget_tokens = 1000
        elif reasoning_effort == "low":
            budget_tokens = 2000
        elif reasoning_effort == "medium":
            budget_tokens = 5000
        elif reasoning_effort == "high":
            budget_tokens = 10000

        settings.extra_body = {
            "enable_thinking": True,
            "thinking_budget": budget_tokens,
        }
        return settings
