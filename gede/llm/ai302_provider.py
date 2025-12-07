# coding=utf-8
#
#

import os

import httpx
from openai import AsyncClient
from .llm_provider import LLMProviderBase, LLMModel
from ..top import logger
from . import common_model_settings

API_KEY = os.getenv("AI302_API_KEY", "")
BASE_URL = os.getenv("AI302_BASE_URL", "https://api.302ai.cn/v1")


class AI302Provider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="302ai",
            name="302ai",
            client=AsyncClient(api_key=API_KEY, base_url=BASE_URL, timeout=120),
        )

    async def load_models_async(self):
        if not API_KEY:
            logger.warning("302AI API key is not set, skipping model loading.")
            return
        url = f"{BASE_URL}/models?llm=1"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url, headers={"Authorization": "Bearer " + API_KEY}, timeout=120
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
                        or model_id.startswith("grok")
                        or model_id.startswith("claude")
                        or model_id.startswith("meta-llama")
                        or model_id.startswith("gemini")
                    ):
                        model = LLMModel(model_id=model_id, name=name)
                        models.append(model)
                logger.debug(f"302AI models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"302AI load models error: {e}")
                return None

    def load_models(self):
        if not API_KEY:
            logger.warning("302AI API key is not set, skipping model loading.")
            return
        logger.info("Loading 302AI models...")
        url = f"{BASE_URL}/models?llm=1"
        try:
            with httpx.Client() as client:
                response = client.get(
                    url, headers={"Authorization": "Bearer " + API_KEY}, timeout=120
                )
            if response.status_code != 200:
                logger.error(
                    "302AI load models failed, status code: {response.status_code}"
                )
                return
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
                    or model_id.startswith("grok")
                    or model_id.startswith("claude")
                    or model_id.startswith("meta-llama")
                    or model_id.startswith("gemini")
                ):
                    models.append(LLMModel(model_id=model_id, name=name))
            logger.debug(f"302AI models loaded: {len(models)}")
            self.models = models
        except Exception as e:
            logger.error(f"302AI load models error: {e}")
