# coding=utf-8
#
#

import os
import logging
from typing import Optional, Any, cast
from openai import AsyncClient
from openai.types.shared import Reasoning, ReasoningEffort

from my_llmkit.models import get_model_info
from my_llmkit.chat.model_settings import ModelSettings
from my_llmkit.chat import LLMChatCompletion, OpenAICompatibleChatCompletion

from .base import LLMProviderBase
from .reasoning import ReasoningEffortType

logger = logging.getLogger(__name__)

API_KEY = os.getenv("DOUBAO_API_KEY", "")
API_BASE_URL = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")


class VoiceEngineProvider(LLMProviderBase):
    def __init__(self):
        super().__init__(
            provider_id="voice_engine",
            name="Voice Engine",
        )

    def get_chat_client(
        self, model_id: str, model_settings: Optional[ModelSettings] = None
    ) -> LLMChatCompletion:
        return OpenAICompatibleChatCompletion(
            api_key=API_KEY,
            model=model_id,
            api_base=API_BASE_URL,
            model_settings=model_settings,
        )

    @property
    def default_models(self) -> Optional[list[str]]:
        return [
            "doubao-seed-2-0-pro-260215",
        ]

    async def load_models(self):
        if not API_KEY:
            logger.warning("Voice Engine API key is not set, skipping model loading.")
            return
        try:
            client = AsyncClient(api_key=API_KEY, base_url=API_BASE_URL, timeout=120)
            models_list = await client.models.list()
            for one in models_list.data:
                model_id = (one.id or "").lower()
                if not model_id:
                    continue
                model_path = f"{self.provider_id}:{model_id}"
                model_info = await get_model_info(model_path)
                if not model_info:
                    logger.warning(f"Voice Engine model info not found: {model_path}")
                    continue
                self.models.append(model_info)
            logger.debug(f"Voice Engine models loaded: {len(self.models)}")
        except Exception as e:
            logger.error(f"Voice Engine load models error: {e}")

    def make_reasoning_setting(
        self, model_id: str, reasoning_effort: ReasoningEffortType
    ) -> ModelSettings:
        settings = ModelSettings()
        extra_body: Any = settings.extra_body or {}
        if reasoning_effort in [None, "auto"]:
            extra_body["thinking"] = {"type": "auto"}
        elif reasoning_effort in ["off", "minimal"]:
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {"type": "enabled"}
            effort = cast(ReasoningEffort, reasoning_effort)
            settings.reasoning = Reasoning(effort=effort)
        settings.extra_body = extra_body
        return settings
