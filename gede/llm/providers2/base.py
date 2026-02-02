# coding=utf-8
#
#

from abc import ABC, abstractmethod
from my_llmkit.models import get_model_info, ModelInfo
from my_llmkit.chat import LLMChatCompletion


class LLMProviderBase:
    provider_id: str
    name: str

    models: list[ModelInfo] = []

    def __init__(self, provider_id: str, name: str):
        self.provider_id = provider_id
        self.name = name

    @abstractmethod
    def get_chat_client(self, model_id: str) -> LLMChatCompletion:
        raise NotImplementedError

    @property
    def default_models(self) -> list[str] | None:
        return None

    @abstractmethod
    async def load_models(self):
        raise NotImplementedError
