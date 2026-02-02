# coding=utf-8
#
#

import os
from typing import Optional
import httpx
from my_llmkit.models import get_model_info, ModelInfo

from .base import LLMProviderBase
from ...top import logger

API_KEY = os.getenv("ZENMUX_API_KEY")
API_BASE_URL = os.getenv("ZENMUX_BASE_URL", "https://zenmux.ai/api/v1")


class ZenMuxProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="zenmux",
            name="ZenMux",
        )

    @property
    def default_models(self) -> Optional[list[str]]:
        return [
            "openai/gpt-5.2",
            "anthropic/claude-sonnet-4.5",
            "x-ai/grok-4-fast",
            "google/gemini-3-pro-preview",
        ]

    async def load_models(self):
        if not API_KEY:
            logger.warning("ZenMux API key is not set, skipping model loading.")
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
                        logger.warning(f"ZenMux model info not found: {model_path}")
                        continue
                    self.models.append(model_info)
                logger.debug(f"ZenMux models loaded: {len(self.models)}")
            except Exception as e:
                logger.error(f"ZenMux load models error: {e}")
