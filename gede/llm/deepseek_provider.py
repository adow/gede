# coding=utf-8
#
#

import os

from agents import ModelSettings
import httpx
from openai import AsyncClient
from .llm_provider import LLMProviderBase, LLMModel, ReasoningEffortType
from ..top import logger

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


class DeepSeekProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="deepseek",
            name="DeepSeek",
            client=AsyncClient(api_key=API_KEY, base_url=BASE_URL, timeout=120),
            models=[
                LLMModel(model_id="deepseek-chat", name="deepseek-chat"),
                LLMModel(model_id="deepseek-reasoner", name="deepseek-reasoner"),
            ],
        )

    def supports_reasoning_for_model_id(self, model_id: str) -> bool:
        if model_id.startswith("deepseek-reasoner"):
            return True
        return super().supports_reasoning_for_model_id(model_id)

    def supports_tools_for_model_id(self, model_id: str) -> bool:
        return True

    def set_model_reasoning(
        self,
        model_id: str,
        input_settings: ModelSettings,
        reasoning_type: ReasoningEffortType = "auto",
    ):
        raise Exception("DeepSeek does not support set_model_reasoning")
