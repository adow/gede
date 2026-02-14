# coding=utf-8
#
#

import os
import logging
from typing import Optional
from openai import AsyncClient

from my_llmkit.models import get_model_info
from my_llmkit.chat import LLMChatCompletion, OpenAICompatibleChatCompletion

from .base import LLMProviderBase

logger = logging.getLogger(__name__)

API_KEY = os.getenv("MOONSHOT_API_KEY", "")
API_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1/")


class MoonshotProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="moonshot",
            name="Moonshot",
        )

    def get_chat_client(self, model_id: str) -> LLMChatCompletion:
        return OpenAICompatibleChatCompletion(
            api_key=API_KEY, model=model_id, api_base=API_BASE_URL
        )

    @property
    def default_models(self) -> Optional[list[str]]:
        return ["kimi-k2.5", "kimi-k2-thinking"]

    async def load_models(self):
        if not API_KEY:
            logger.warning("Moonshot API key is not set, skipping model loading.")
            return

        try:
            client = AsyncClient(api_key=API_KEY, base_url=API_BASE_URL, timeout=120)
            models_list = await client.models.list()
            for one in models_list.data:
                model_id = (one.id or "").lower()
                if not model_id:
                    continue
                model_path = f"{self.provider_id}:{model_id}"
                model_info = await get_model_info(model_path)
                if not model_info:
                    logger.warning(f"Moonshot model info not found: {model_path}")
                    continue
                self.models.append(model_info)
            logger.debug(f"Moonshot models loaded: {len(self.models)}")
        except Exception as e:
            logger.error(f"Moonshot load models error: {e}")
