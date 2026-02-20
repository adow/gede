# coding=utf-8
#
#

from abc import ABC, abstractmethod
from typing import Optional

from my_llmkit.models import get_model_info, ModelInfo
from my_llmkit.chat import LLMChatCompletion
from my_llmkit.chat.model_settings import ModelSettings
from .reasoning import (
    ReasoningEffortType,
)


class LLMProviderBase:
    provider_id: str
    name: str

    models: list[ModelInfo] = []

    def __init__(self, provider_id: str, name: str):
        self.provider_id = provider_id
        self.name = name

    @abstractmethod
    def get_chat_client(
        self, model_id: str, model_settings: Optional[ModelSettings] = None
    ) -> LLMChatCompletion:
        raise NotImplementedError

    @property
    def default_models(self) -> list[str] | None:
        return None

    def default_model_settings(self, model_id: str) -> ModelSettings:
        return ModelSettings(
            include_usage=True,
            extra_headers={
                "X-Title": "gede",
                "HTTP-Referer": "https://gede.slashusr.xyz",
            },
        )

    @abstractmethod
    async def load_models(self):
        raise NotImplementedError

    def make_reasoning_setting(
        self, model_id: str, reasoning_effort: ReasoningEffortType
    ) -> ModelSettings:
        raise NotImplementedError
