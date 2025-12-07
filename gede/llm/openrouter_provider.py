# coding=utf-8
#
# openrouter_provider.py
#

import os

import httpx
from openai import AsyncClient
from agents import ModelSettings
from . import common_model_settings
from .llm_provider import LLMProviderBase, LLMModel
from ..top import logger


API_KEY = os.getenv("OPENROUTER_API_KEY")
API_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class OpenRouterProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="openrouter",
            name="OpenRouter",
            client=AsyncClient(api_key=API_KEY, base_url=API_BASE_URL, timeout=60),
        )
        self.models = [
            LLMModel(model_id="x-ai/grok-3", name="Grok-3"),
            LLMModel(model_id="x-ai/grok-4", name="Grok-4"),
        ]

    async def load_models_async(self):
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
                models: list[LLMModel] = []
                for one in data:
                    model_id = one.get("id", "").lower()
                    name = one.get("name")
                    # logger.info(f"OpenRouter model: {model_id}, {name}")
                    if (
                        model_id.startswith("openai")
                        or model_id.startswith("x-ai")
                        or model_id.startswith("anthropic")
                        or model_id.startswith("meta-llama")
                        or model_id.startswith("google")
                    ):
                        model = LLMModel(model_id=model_id, name=name)
                        models.append(model)
                logger.debug(f"OpenRouter models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"OpenRouter load models error: {e}")

    def load_models(self):
        if not API_KEY:
            logger.warning("OpenRouter API key is not set, skipping model loading.")
            return
        logger.info("Loading OpenRouter models...")
        url = f"{API_BASE_URL}/models"
        try:
            with httpx.Client() as client:
                response = client.get(url, timeout=10)
            if response.status_code != 200:
                return
            result = response.json()
            data = result.get("data", [])
            models: list[LLMModel] = []
            for one in data:
                model_id = one.get("id", "").lower()
                name = one.get("name")
                if (
                    model_id.startswith("openai")
                    or model_id.startswith("x-ai")
                    or model_id.startswith("anthropic")
                    or model_id.startswith("meta-llama")
                    or model_id.startswith("google")
                ):
                    models.append(LLMModel(model_id=model_id, name=name))
            logger.debug(f"OpenRouter models loaded: {len(models)}")
            self.models = models
        except Exception as e:
            logger.error(f"OpenRouter load models error: {e}")

    def model_settings_for_model_id(self, model_id: str):
        base_settings = super().model_settings_for_model_id(model_id)
        openrouter_base_settings = ModelSettings(
            extra_headers={
                "X-Title": "gede",
                "HTTP-Referer": "https://gede.slashusr.xyz",
            }
        )
        base_settings = base_settings.resolve(openrouter_base_settings)

        return base_settings
