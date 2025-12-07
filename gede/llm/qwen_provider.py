# coding=utf-8
#
#

import os
from typing import Optional, Any

import httpx
from openai import AsyncClient
from agents import ModelSettings
from openai.types.shared import ReasoningEffort
from .llm_provider import LLMProviderBase, LLMModel, ReasoningEffortType, WebSearchType
from ..top import logger

API_KEY = os.getenv("QWEN_API_KEY", "")
BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_models = [
    "qwen-plus",
    "qwen-turbo",
    "qwen-flash",
    "qwen3",
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen3-0.6b",
]

web_search_models = ["qwen-plus", "qwen-turbo", "qwen-max"]


class QwenProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="qwen",
            name="Qwen",
            client=AsyncClient(api_key=API_KEY, base_url=BASE_URL, timeout=120),
        )

    async def load_models_async(self):
        if not API_KEY:
            logger.warning("Qwen API key is not set, skipping model loading.")
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
                    if model_id.startswith("qwen"):
                        model = LLMModel(model_id=model_id, name=name)
                        models.append(model)
                logger.debug(f"Qwen models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"Qwen load models error: {e}")
                return None

    def load_models(self):
        """
        Load Qwen models
        """
        if not API_KEY:
            logger.warning("Qwen API key is not set, skipping model loading.")
            return
        logger.info("Loading Qwen models...")
        url = f"{BASE_URL}/models"
        with httpx.Client() as client:
            try:
                response = client.get(
                    url, headers={"Authorization": "Bearer " + API_KEY}, timeout=10
                )
                if response.status_code != 200:
                    logger.error(
                        f"Qwen load models failed: {response.status_code}, {response.text}"
                    )
                    return None
                result = response.json()
                data = result.get("data", [])
                models: list[LLMModel] = []
                for one in data:
                    model_id = one.get("id", "").lower()
                    if model_id.startswith("qwen"):
                        model = LLMModel(model_id=model_id, name=model_id)
                        models.append(model)
                logger.debug(f"Qwen models loaded: {len(models)}")
                self.models = models
            except Exception as e:
                logger.error(f"Qwen load models error: {e}")
                return None

    # model settings
    def reasoning_model_settings_for_model_id(
        self, model_id: str, effort: ReasoningEffort = "medium"
    ) -> Optional[ModelSettings]:
        if model_id in reasoning_models:
            settings = ModelSettings()
            budget_tokens = 2000
            if effort == "minimal":
                budget_tokens = 1000
            elif effort == "low":
                budget_tokens = 2000
            elif effort == "medium":
                budget_tokens = 5000
            elif effort == "high":
                budget_tokens = 10000
            extra_body = {
                "enable_thinking": True,
                "thinking_budget": budget_tokens,
            }
            settings.extra_body = extra_body
            return settings

    def web_search_model_settings_for_model_id(
        self, model_id: str
    ) -> Optional[ModelSettings]:
        if model_id in web_search_models:
            settings = ModelSettings()
            extra_body = {
                "enable_search": True,  # Enable web search parameter
                "search_options": {
                    "forced_search": True,  # Force web search parameter
                    "search_strategy": "max",  # Model will search 10 internet information
                    "enable_source": True,
                    "enable_citation": True,
                },
            }
            settings.extra_body = extra_body
            return settings
        return None

    def set_model_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        extra_body: Any = input_settings.extra_body or {}
        if reasoning_type == "auto":
            del extra_body["enable_thinking"]
            del extra_body["thinking_budget"]
            input_settings.extra_body = extra_body
            return input_settings

        if model_id not in reasoning_models:
            raise Exception(f"Reasoning not supported for model: {model_id}")

        if reasoning_type == "off":
            extra_body["enable_thinking"] = False
            del extra_body["thinking_budget"]
            input_settings.extra_body = extra_body
            return input_settings

        extra_body["enable_thinking"] = True
        budget_tokens = 2000
        if reasoning_type == "minimal":
            budget_tokens = 1000
        elif reasoning_type == "low":
            budget_tokens = 2000
        elif reasoning_type == "medium":
            budget_tokens = 5000
        elif reasoning_type == "high":
            budget_tokens = 10000
        extra_body["thinking_budget"] = budget_tokens
        input_settings.extra_body = extra_body
        return input_settings

    def set_model_web_search(
        self,
        model_id: str,
        input_settings: ModelSettings,
        web_search_type: WebSearchType = "auto",
    ):
        extra_body: Any = input_settings.extra_body or {}
        if web_search_type == "auto":
            if "enable_search" in extra_body:
                del extra_body["enable_search"]
            if "search_options" in extra_body:
                del extra_body["search_options"]
            input_settings.extra_body = extra_body
            return input_settings
        if model_id not in web_search_models:
            raise Exception(f"Web search not supported for model: {model_id}")
        if web_search_type == "off":
            extra_body["enable_search"] = False
            if "search_options" in extra_body:
                del extra_body["search_options"]
            input_settings.extra_body = extra_body
            return input_settings
        if web_search_type == "on":
            extra_body["enable_search"] = True
            extra_body["search_options"] = {
                "forced_search": True,
                "search_strategy": "max",
                "enable_source": True,
                "enable_citation": True,
            }
            input_settings.extra_body = extra_body
            return input_settings

    # supports

    def supports_reasoning_for_model_id(self, model_id: str) -> bool:
        if model_id in reasoning_models:
            return True
        return super().supports_reasoning_for_model_id(model_id)

    def supports_web_search_for_model_id(self, model_id: str) -> bool:
        if model_id in web_search_models:
            return True
        return super().supports_web_search_for_model_id(model_id)
