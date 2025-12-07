# coding=utf-8
#
#

import os

import httpx
from openai import AsyncClient
from .llm_provider import LLMProviderBase, LLMModel
from ..top import logger

API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")


class AnthropicProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="anthropic",
            name="Anthropic",
            client=AsyncClient(api_key=API_KEY, base_url=BASE_URL, timeout=120),
        )

    async def load_models_async(self):
        if not API_KEY:
            logger.warning("Anthropic API key is not set, skipping model loading.")
            return
        url = f"{BASE_URL}/models"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url, headers={"Authorization": "Bearer " + API_KEY}, timeout=10
                )
                if response.status_code != 200:
                    return None
                result = response.json()
                data = result.get("data", [])
                models: list[LLMModel] = []
                for one in data:
                    model_id = one.get("id", "").lower()
                    name = one.get("name")
                    if model_id.startswith("claude"):
                        model = LLMModel(model_id=model_id, name=name)
                        models.append(model)
                logger.debug(f"Anthropic models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"Anthropic load models error: {e}")
                return None

    def load_models(self):
        """
        Load Anthropic models
        """
        if not API_KEY:
            logger.warning("Anthropic API key is not set, skipping model loading.")
            return
        logger.info("Loading Anthropic models...")
        url = f"{BASE_URL}/models"
        with httpx.Client() as client:
            try:
                response = client.get(
                    url, headers={"Authorization": "Bearer " + API_KEY}, timeout=10
                )
                if response.status_code != 200:
                    logger.error(
                        f"Anthropic load models failed: {response.status_code}, {response.text}"
                    )
                    return None
                result = response.json()
                data = result.get("data", [])
                models: list[LLMModel] = []
                for one in data:
                    model_id = one.get("id", "").lower()
                    if model_id.startswith("claude"):
                        model = LLMModel(model_id=model_id, name=model_id)
                        models.append(model)
                logger.debug(f"Anthropic models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"Anthropic load models error: {e}")
                return None
