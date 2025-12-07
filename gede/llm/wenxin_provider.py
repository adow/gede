# coding=utf-8
#
#

import os
from typing import Optional, Any

import httpx
from openai import AsyncClient
from agents import ModelSettings
from .llm_provider import LLMProviderBase, LLMModel, WebSearchType
from ..top import logger

API_KEY = os.getenv("WENXIN_API_KEY", "")
BASE_URL = os.getenv("WENXIN_BASE_URL", "https://qianfan.baidubce.com/v2")

web_search_models = [
    "ernie-x1-turbo-32k",
    "ernie-4.5-turbo-32k",
    "ernie-4.5-turbo-128k",
]


class WenxinProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="wenxin",
            name="Wenxin",
            client=AsyncClient(api_key=API_KEY, base_url=BASE_URL, timeout=120),
        )

    async def load_models_async(self):
        if not API_KEY:
            logger.warning("Wenxin API key is not set, skipping model loading.")
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
                    if model_id.startswith("erine"):
                        model = LLMModel(model_id=model_id, name=name)
                        models.append(model)
                logger.debug(f"Wenxin models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"Wenxin load models error: {e}")
                return None

    def load_models(self):
        if not API_KEY:
            logger.warning("Wenxin API key is not set, skipping model loading.")
            return
        logger.info("Loading Wenxin models...")
        url = f"{BASE_URL}/models"
        with httpx.Client() as client:
            try:
                response = client.get(
                    url, headers={"Authorization": "Bearer " + API_KEY}, timeout=10
                )
                if response.status_code != 200:
                    logger.error(
                        f"Wenxin load models failed: {response.status_code}, {response.text}"
                    )
                    return None
                result = response.json()
                data = result.get("data", [])
                models: list[LLMModel] = []
                for one in data:
                    model_id = one.get("id", "").lower()
                    if model_id.startswith("ernie"):
                        model = LLMModel(model_id=model_id, name=model_id)
                        models.append(model)
                logger.debug(f"Wenxin models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"Wenxin load models error: {e}")
                return None

    # model settings

    def model_settings_for_model_id(self, model_id: str) -> ModelSettings:
        base_settings = super().model_settings_for_model_id(model_id)
        settings = ModelSettings()
        if model_id == "ernie-4.5-turbo-128k":
            settings.max_tokens = 12000
        if model_id == "ernie-x1-turbo-32k":
            settings.max_tokens = 27000

        base_settings = base_settings.resolve(settings)
        return base_settings

    def web_search_model_settings_for_model_id(
        self, model_id: str
    ) -> Optional[ModelSettings]:
        if model_id in web_search_models:
            settings = ModelSettings()
            web_search = {
                "enable": True,
            }
            if model_id in ["ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k"]:
                web_search["enable_citation"] = True
                web_search["enable_trace"] = True
            settings.extra_body = {"web_search": web_search}
            return settings
        return None

    def set_model_web_search(
        self,
        model_id: str,
        input_settings: ModelSettings,
        web_search_type: WebSearchType = "auto",
    ):
        extra_body: Any = input_settings.extra_body or {}
        if web_search_type == "auto":
            if "web_search" in extra_body:
                del extra_body["web_search"]
            input_settings.extra_body = extra_body
            return input_settings
        if model_id not in web_search_models:
            raise Exception(f"Model {model_id} does not support web search.")
        if web_search_type == "on":
            extra_body["web_search"] = {"enable": True}
            if model_id in ["ernie-4.5-turbo-32k", "ernie-4.5-turbo-128k"]:
                extra_body["web_search"]["enable_citation"] = True
                extra_body["web_search"]["enable_trace"] = True
            input_settings.extra_body = extra_body
            return input_settings
        if web_search_type == "off":
            extra_body["web_search"] = {"enable": False}
            if "enable_citation" in extra_body["web_search"]:
                del extra_body["web_search"]["enable_citation"]
            if "enable_trace" in extra_body["web_search"]:
                del extra_body["web_search"]["enable_trace"]
            input_settings.extra_body = extra_body
            return input_settings

    # supports

    def supports_reasoning_for_model_id(self, model_id: str) -> bool:
        if model_id.startswith("ernie-x1"):
            return True
        return False

    def supports_web_search_for_model_id(self, model_id: str) -> bool:
        if model_id in web_search_models:
            return True
        return False

    def supports_tools_for_model_id(self, model_id: str) -> bool:
        return True
