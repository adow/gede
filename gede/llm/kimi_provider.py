# coding=utf-8
# kimi
#

import os

import httpx
from openai import AsyncClient, Client
from agents import ModelSettings
from .llm_provider import LLMProviderBase, LLMModel, ReasoningEffortType
from ..top import logger

API_KEY = os.getenv("MOONSHOT_API_KEY", "")
BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1/")


kimi_reasoning_model_id_prefix = ["kimi-k2-", "kimi-k2-thinking-"]


class KimiProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="kimi",
            name="Kimi",
            client=AsyncClient(api_key=API_KEY, base_url=BASE_URL, timeout=120),
        )

    async def load_models_async(self):
        if not API_KEY:
            logger.warning(
                "MOONSHOT_AP_KEY API key is not set, skipping model loading."
            )
            return
        logger.info("Loading Moonshot models...")
        try:
            models_list = await self.client.models.list()
            models: list[LLMModel] = []
            for one in models_list.data:
                model_id = one.id
                name = model_id
                model = LLMModel(model_id=model_id, name=name)
                models.append(model)
            self.models = models
            logger.debug(f"Kimi models loaded: {len(models)}")
        except Exception as e:
            logger.error(f"Kimi load models error: {e}")
            return None

    def load_models(self):
        if not API_KEY:
            logger.warning(
                "MOONSHOT_AP_KEY API key is not set, skipping model loading."
            )
            return
        logger.info("Loading Moonshot models...")
        try:
            client = Client(api_key=API_KEY, base_url=BASE_URL, timeout=120)
            models_list = client.models.list()
            models: list[LLMModel] = []
            for one in models_list.data:
                model_id = one.id
                name = model_id
                model = LLMModel(model_id=model_id, name=name)
                models.append(model)
            self.models = models
            logger.debug(f"Kimi models loaded: {len(models)}")
        except Exception as e:
            logger.error(f"Kimi load models error: {e}")
            return None

    def model_settings_for_model_id(self, model_id: str) -> ModelSettings:
        base_settings = super().model_settings_for_model_id(model_id)
        if any(
            model_id.startswith(prefix) for prefix in kimi_reasoning_model_id_prefix
        ):
            base_settings.temperature = 1.0
        return base_settings

    def set_model_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        if any(
            model_id.startswith(prefix) for prefix in kimi_reasoning_model_id_prefix
        ):
            logger.warning("Kimi does not support setting reasoning effort.")
        return input_settings

    def supports_reasoning_for_model_id(self, model_id: str) -> bool:
        if any(
            model_id.startswith(prefix) for prefix in kimi_reasoning_model_id_prefix
        ):
            return True
        return super().supports_reasoning_for_model_id(model_id)

    def supports_tools_for_model_id(self, model_id: str) -> bool:
        return True
