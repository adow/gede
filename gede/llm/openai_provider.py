# coding=utf-8
#
#

import os

import httpx
from openai import AsyncClient
from agents import ModelSettings

from .llm_provider import LLMProviderBase, LLMModel
from ..top import logger

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


class OpenAIProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="openai",
            name="OpenAI",
            client=AsyncClient(api_key=API_KEY, base_url=BASE_URL, timeout=120),
        )

    async def load_models_async(self):
        if not API_KEY:
            logger.warning("OpenAI API key is not set, skipping model loading.")
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
                    if (
                        model_id.startswith("gpt")
                        or model_id.startswith("o1")
                        or model_id.startswith("o3")
                        or model_id.startswith("o4")
                        or model_id.startswith("openai/o1")
                        or model_id.startswith("openai/o3")
                        or model_id.startswith("openai/o4")
                        or model_id.startswith("openai/gpt-oss")
                        or model_id.startswith("openai/gpt-5")
                    ):
                        model = LLMModel(model_id=model_id, name=name)
                        models.append(model)
                logger.debug(f"OpenAI models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"OpenAI load models error: {e}")
                return None

    def load_models(self):
        """
        Load OpenAI models
        """
        if not API_KEY:
            logger.warning("OpenAI API key is not set, skipping model loading.")
            return
        logger.info("Loading OpenAI models...")
        url = f"{BASE_URL}/models"
        with httpx.Client() as client:
            try:
                response = client.get(
                    url, headers={"Authorization": "Bearer " + API_KEY}, timeout=10
                )
                if response.status_code != 200:
                    logger.error(
                        f"OpenAI load models failed: {response.status_code}, {response.text}"
                    )
                    return None
                result = response.json()
                data = result.get("data", [])
                models: list[LLMModel] = []
                for one in data:
                    model_id = one.get("id", "").lower()
                    if (
                        model_id.startswith("gpt")
                        or model_id.startswith("o1")
                        or model_id.startswith("o3")
                        or model_id.startswith("o4")
                        or model_id.startswith("openai/o1")
                        or model_id.startswith("openai/o3")
                        or model_id.startswith("openai/o4")
                        or model_id.startswith("openai/gpt-oss")
                        or model_id.startswith("openai/gpt-5")
                    ):
                        model = LLMModel(model_id=model_id, name=model_id)
                        models.append(model)
                logger.debug(f"OpenAI models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"OpenAI load models error: {e}")
                return None
